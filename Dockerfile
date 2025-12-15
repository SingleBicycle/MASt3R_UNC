FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0t64 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    build-essential \
    python3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace/repos/mast3r
COPY . .


RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r dust3r/requirements.txt \
 && pip install --no-cache-dir "numpy<1.27" \
 && pip install --no-cache-dir -r dust3r/requirements_optional.txt \
 && pip install --no-cache-dir cython \
 && pip install --no-cache-dir "faiss-cpu>=1.8.0"


RUN git clone https://github.com/jenicek/asmk /tmp/asmk \
 && cd /tmp/asmk/cython && cythonize *.pyx \
 && cd /tmp/asmk && pip install . \
 && rm -rf /tmp/asmk

RUN cd dust3r/croco/models/curope \
 && python setup.py build_ext --inplace \
 || echo "RoPE CUDA build failed or skipped, continuing..."


WORKDIR /workspace


#docker run --gpus all -it --ipc=host  --name mast3r-dev   -v /DATA/zihao/projects/crashtwin/repos:/workspace/repos   -v /DATA/zihao/projects/crashtwin/data:/workspace/data   mast3r-full   /bin/bash