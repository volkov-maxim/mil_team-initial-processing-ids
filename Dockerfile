FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential g++ python3.12 python3-pip python3.12-venv python3.12-distutils git ffmpeg python3-dev python3.12-dev libsm6 libxext6 libgl1 libglib2.0-0 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "app.main"]