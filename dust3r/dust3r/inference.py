# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from mast3r.losses_evidential import evidential_nig_loss, predictive_mean_var, evidential_nig_loss_3d


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def _get_first(view, keys):
    for k in keys:
        if isinstance(view, dict) and k in view:
            return view[k]
    return None


def _get_gt_xyz(view1, view2, device):
    pts1 = _get_first(view1, ("pts3d",))
    pts2 = _get_first(view2, ("pts3d",))
    cam1 = _get_first(view1, ("camera_pose",))

    if pts1 is None or pts2 is None or cam1 is None:
        return None

    pts1 = pts1.to(device, non_blocking=True)
    pts2 = pts2.to(device, non_blocking=True)
    cam1 = cam1.to(device, non_blocking=True)

    world_to_cam1 = inv(cam1)
    gt_xyz1 = geotrf(world_to_cam1, pts1)
    gt_xyz2 = geotrf(world_to_cam1, pts2)

    if gt_xyz1.ndim == 4 and gt_xyz1.shape[-1] == 3:
        gt_xyz1 = gt_xyz1.permute(0, 3, 1, 2)
    if gt_xyz2.ndim == 4 and gt_xyz2.shape[-1] == 3:
        gt_xyz2 = gt_xyz2.permute(0, 3, 1, 2)

    return gt_xyz1, gt_xyz2


def _compute_uq_loss(view1, view2, pred1, pred2, args):
    if args is None:
        return None

    lam_geom = getattr(args, 'lambda_uq', 0.0)
    lam_feat_evi = getattr(args, 'lambda_evi_feat', 0.0)
    lam_mean_gt = getattr(args, 'lambda_mean_gt', 0.0)
    lam_mean_distill = getattr(args, 'lambda_mean_distill', 0.0)
    lam_var_distill = getattr(args, 'lambda_var_distill', 0.0)
    lam_uq_xyz = getattr(args, "lambda_uq_xyz", 0.0)

    if max(lam_geom, lam_feat_evi, lam_mean_gt, lam_mean_distill, lam_var_distill, lam_uq_xyz) <= 0:
        return None

    depth1 = _get_first(view1, ('depth', 'depth_gt', 'depthmap', 'depth1', 'metric_depth'))
    depth2 = _get_first(view2, ('depth', 'depth_gt', 'depthmap', 'depth2', 'metric_depth'))
    valid1 = _get_first(view1, ('valid_depth', 'valid1', 'valid_mask', 'mask'))
    valid2 = _get_first(view2, ('valid_depth', 'valid2', 'valid_mask', 'mask'))
    if depth1 is None or depth2 is None or valid1 is None or valid2 is None:
        return None

    def _to_mask(mask, target):
        if mask is None:
            return None
        mask = mask.to(target.device, non_blocking=True).float()
        if mask.ndim == target.ndim - 1:
            mask = mask.unsqueeze(1)
        return mask

    def _masked_mean(x, mask):
        if mask is not None:
            x = x * mask
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(x.numel(), dtype=x.dtype, device=x.device)
        return x.sum() / denom

    candidate_devices = [
        _get_first(pred1, ("depth_gamma", "gamma", "depth_gamma_feat")),
        _get_first(pred1, ("pts3d",)),
        _get_first(pred2, ("depth_gamma", "gamma", "depth_gamma_feat")),
        _get_first(pred2, ("pts3d",)),
        depth1,
    ]
    target_device = None
    for cand in candidate_devices:
        if torch.is_tensor(cand):
            target_device = cand.device
            break
    target_device = target_device or depth1.device

    depth1 = depth1.to(target_device, non_blocking=True)
    depth2 = depth2.to(target_device, non_blocking=True)
    valid1 = _to_mask(valid1, depth1)
    valid2 = _to_mask(valid2, depth2)

    losses = {}
    stats = {}

    # Geometry-level evidential loss (teacher)
    if getattr(args, "uq_mode", "geom") in ("geom", "both") and lam_geom > 0:
        gamma1 = _get_first(pred1, ('depth_gamma', 'gamma'))
        nu1 = _get_first(pred1, ('depth_nu', 'nu'))
        alpha1 = _get_first(pred1, ('depth_alpha', 'alpha'))
        beta1 = _get_first(pred1, ('depth_beta', 'beta'))

        gamma2 = _get_first(pred2, ('depth_gamma', 'gamma'))
        nu2 = _get_first(pred2, ('depth_nu', 'nu'))
        alpha2 = _get_first(pred2, ('depth_alpha', 'alpha'))
        beta2 = _get_first(pred2, ('depth_beta', 'beta'))

        if all(x is not None for x in (gamma1, nu1, alpha1, beta1, gamma2, nu2, alpha2, beta2)):
            loss_uq_1 = evidential_nig_loss(depth1, gamma1, nu1, alpha1, beta1,
                                            mask=valid1,
                                            lambda_evi=getattr(args, 'lambda_evi', 1e-3))
            loss_uq_2 = evidential_nig_loss(depth2, gamma2, nu2, alpha2, beta2,
                                            mask=valid2,
                                            lambda_evi=getattr(args, 'lambda_evi', 1e-3))
            losses['loss_uq_geom'] = 0.5 * (loss_uq_1 + loss_uq_2)
            stats.update({
                'loss_uq_geom_1': float(loss_uq_1.detach()),
                'loss_uq_geom_2': float(loss_uq_2.detach()),
                'loss_uq_geom': float(losses['loss_uq_geom'].detach()),
            })

            depth_mu1_t, depth_var1_t = predictive_mean_var(gamma1, nu1, alpha1, beta1)
            depth_mu2_t, depth_var2_t = predictive_mean_var(gamma2, nu2, alpha2, beta2)
            stats.update({
                'depth_mu_geom_1_mean': float(depth_mu1_t.detach().mean()),
                'depth_mu_geom_2_mean': float(depth_mu2_t.detach().mean()),
                'depth_var_geom_1_mean': float(depth_var1_t.detach().mean()),
                'depth_var_geom_2_mean': float(depth_var2_t.detach().mean()),
            })

    # Feature-level evidential loss (student)
    if getattr(args, "uq_mode", "geom") in ("feat", "both") and (
        lam_feat_evi > 0 or lam_mean_gt > 0 or lam_mean_distill > 0 or lam_var_distill > 0
    ):
        gamma1_f = _get_first(pred1, ("depth_gamma_feat",))
        nu1_f = _get_first(pred1, ("depth_nu_feat",))
        alpha1_f = _get_first(pred1, ("depth_alpha_feat",))
        beta1_f = _get_first(pred1, ("depth_beta_feat",))

        gamma2_f = _get_first(pred2, ("depth_gamma_feat",))
        nu2_f = _get_first(pred2, ("depth_nu_feat",))
        alpha2_f = _get_first(pred2, ("depth_alpha_feat",))
        beta2_f = _get_first(pred2, ("depth_beta_feat",))

        have_feat = all(x is not None for x in (gamma1_f, nu1_f, alpha1_f, beta1_f, gamma2_f, nu2_f, alpha2_f, beta2_f))
        if have_feat:
            depth_mean1_feat = _get_first(pred1, ("depth_mean_feat",))
            depth_mean2_feat = _get_first(pred2, ("depth_mean_feat",))

            if lam_feat_evi > 0:
                loss_uq_f1 = evidential_nig_loss(depth1, gamma1_f, nu1_f, alpha1_f, beta1_f,
                                                 mask=valid1,
                                                 lambda_evi=getattr(args, 'lambda_evi', 1e-3))
                loss_uq_f2 = evidential_nig_loss(depth2, gamma2_f, nu2_f, alpha2_f, beta2_f,
                                                 mask=valid2,
                                                 lambda_evi=getattr(args, 'lambda_evi', 1e-3))
                losses['loss_uq_feat'] = 0.5 * (loss_uq_f1 + loss_uq_f2)
                stats.update({
                    'loss_uq_feat_1': float(loss_uq_f1.detach()),
                    'loss_uq_feat_2': float(loss_uq_f2.detach()),
                    'loss_uq_feat': float(losses['loss_uq_feat'].detach()),
                })

            depth_mu1_feat, depth_var1_feat = predictive_mean_var(gamma1_f, nu1_f, alpha1_f, beta1_f)
            depth_mu2_feat, depth_var2_feat = predictive_mean_var(gamma2_f, nu2_f, alpha2_f, beta2_f)

            student_mu1 = depth_mean1_feat if depth_mean1_feat is not None else depth_mu1_feat
            student_mu2 = depth_mean2_feat if depth_mean2_feat is not None else depth_mu2_feat

            if lam_mean_gt > 0:
                l_gt1 = _masked_mean((student_mu1 - depth1) ** 2, valid1)
                l_gt2 = _masked_mean((student_mu2 - depth2) ** 2, valid2)
                losses['loss_mean_gt'] = 0.5 * (l_gt1 + l_gt2)
                stats.update({
                    'loss_mean_gt_1': float(l_gt1.detach()),
                    'loss_mean_gt_2': float(l_gt2.detach()),
                    'loss_mean_gt': float(losses['loss_mean_gt'].detach()),
                })

            teacher_mu1 = _get_first(pred1, ("depth_mu", "depth_mean", "depth_gamma", "gamma"))
            teacher_mu2 = _get_first(pred2, ("depth_mu", "depth_mean", "depth_gamma", "gamma"))
            teacher_var1 = _get_first(pred1, ("depth_var",))
            teacher_var2 = _get_first(pred2, ("depth_var",))

            if lam_mean_distill > 0 and teacher_mu1 is not None and teacher_mu2 is not None:
                l_md1 = _masked_mean((student_mu1 - teacher_mu1) ** 2, valid1)
                l_md2 = _masked_mean((student_mu2 - teacher_mu2) ** 2, valid2)
                losses['loss_mean_distill'] = 0.5 * (l_md1 + l_md2)
                stats.update({
                    'loss_mean_distill_1': float(l_md1.detach()),
                    'loss_mean_distill_2': float(l_md2.detach()),
                    'loss_mean_distill': float(losses['loss_mean_distill'].detach()),
                })

            if lam_var_distill > 0 and teacher_var1 is not None and teacher_var2 is not None:
                l_vd1 = _masked_mean(torch.abs(depth_var1_feat - teacher_var1), valid1)
                l_vd2 = _masked_mean(torch.abs(depth_var2_feat - teacher_var2), valid2)
                losses['loss_var_distill'] = 0.5 * (l_vd1 + l_vd2)
                stats.update({
                    'loss_var_distill_1': float(l_vd1.detach()),
                    'loss_var_distill_2': float(l_vd2.detach()),
                    'loss_var_distill': float(losses['loss_var_distill'].detach()),
                })

            stats.update({
                'depth_mu_feat_1_mean': float(depth_mu1_feat.detach().mean()),
                'depth_mu_feat_2_mean': float(depth_mu2_feat.detach().mean()),
                'depth_var_feat_1_mean': float(depth_var1_feat.detach().mean()),
                'depth_var_feat_2_mean': float(depth_var2_feat.detach().mean()),
            })
            if depth_mean1_feat is not None:
                stats['depth_mean_feat_1_mean'] = float(depth_mean1_feat.detach().mean())
            if depth_mean2_feat is not None:
                stats['depth_mean_feat_2_mean'] = float(depth_mean2_feat.detach().mean())

    if lam_uq_xyz > 0:
        gamma1_xyz = _get_first(pred1, ("xyz_gamma",))
        nu1_xyz = _get_first(pred1, ("xyz_nu",))
        alpha1_xyz = _get_first(pred1, ("xyz_alpha",))
        beta1_xyz = _get_first(pred1, ("xyz_beta",))

        gamma2_xyz = _get_first(pred2, ("xyz_gamma",))
        nu2_xyz = _get_first(pred2, ("xyz_nu",))
        alpha2_xyz = _get_first(pred2, ("xyz_alpha",))
        beta2_xyz = _get_first(pred2, ("xyz_beta",))

        have_xyz = all(
            x is not None
            for x in (gamma1_xyz, nu1_xyz, alpha1_xyz, beta1_xyz, gamma2_xyz, nu2_xyz, alpha2_xyz, beta2_xyz)
        )
        if have_xyz:
            gt_xyz = _get_gt_xyz(view1, view2, target_device)
            if gt_xyz is not None:
                y_xyz1, y_xyz2 = gt_xyz
                lam_evi_xyz = getattr(args, "lambda_evi_xyz", 1e-3)

                loss_xyz1 = evidential_nig_loss_3d(
                    y_xyz1, gamma1_xyz, nu1_xyz, alpha1_xyz, beta1_xyz,
                    mask=valid1, lambda_evi=lam_evi_xyz,
                )
                loss_xyz2 = evidential_nig_loss_3d(
                    y_xyz2, gamma2_xyz, nu2_xyz, alpha2_xyz, beta2_xyz,
                    mask=valid2, lambda_evi=lam_evi_xyz,
                )
                losses["loss_uq_xyz"] = 0.5 * (loss_xyz1 + loss_xyz2)
                stats.update({
                    "loss_uq_xyz_1": float(loss_xyz1.detach()),
                    "loss_uq_xyz_2": float(loss_xyz2.detach()),
                    "loss_uq_xyz": float(losses["loss_uq_xyz"].detach()),
                })

    if not losses:
        return None

    return losses, stats


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None, args=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            base_loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    loss_details = {}
    if base_loss is None:
        loss = torch.tensor(0.0, device=device)
    elif isinstance(base_loss, tuple):
        loss, loss_details = base_loss
    else:
        loss = base_loss
        if loss.ndim == 0:
            loss_details = {'loss_base': float(loss)}

    uq_res = _compute_uq_loss(view1, view2, pred1, pred2, args)
    if uq_res is not None:
        uq_losses, uq_stats = uq_res
        if 'loss_uq_geom' in uq_losses:
            loss = loss + getattr(args, 'lambda_uq', 1.0) * uq_losses['loss_uq_geom']
        if 'loss_uq_feat' in uq_losses:
            loss = loss + getattr(args, 'lambda_evi_feat', 0.0) * uq_losses['loss_uq_feat']
        if 'loss_mean_gt' in uq_losses:
            loss = loss + getattr(args, 'lambda_mean_gt', 0.0) * uq_losses['loss_mean_gt']
        if 'loss_mean_distill' in uq_losses:
            loss = loss + getattr(args, 'lambda_mean_distill', 0.0) * uq_losses['loss_mean_distill']
        if 'loss_var_distill' in uq_losses:
            loss = loss + getattr(args, 'lambda_var_distill', 0.0) * uq_losses['loss_var_distill']
        lam_uq_xyz = getattr(args, "lambda_uq_xyz", 0.0)
        if "loss_uq_xyz" in uq_losses and lam_uq_xyz > 0:
            loss = (1.0 - lam_uq_xyz) * loss + lam_uq_xyz * uq_losses["loss_uq_xyz"]
            loss_details["loss_uq_xyz"] = float(uq_losses["loss_uq_xyz"].detach())
        loss_details.update(uq_stats)

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=(loss, loss_details))
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
