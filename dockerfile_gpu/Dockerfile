FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TZ=Etc/UTC

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# TensorFlow + DeepCell stack
RUN pip install \
    numpy==1.23.5 \
    tifffile==2023.7.10 \
    scikit-image==0.19.3 \
    tensorflow==2.8.4 \
    deepcell==0.12.6

COPY preload_model.py /app/preload_model.py
COPY segment_tiles.py /app/segment_tiles.py

# Optional: preload DeepCell assets into image cache
RUN python /app/preload_model.py || true

ENTRYPOINT ["python", "/app/segment_tiles.py"]
