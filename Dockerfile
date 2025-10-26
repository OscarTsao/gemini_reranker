FROM nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-distutils python3-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY . .

RUN python3 -m pip install -e .

CMD ["bash"]
