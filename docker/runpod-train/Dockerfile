# FROM nvidia/cuda:12.4.1-base-ubuntu22.04
FROM ubuntu:jammy

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc


# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | shv
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Clone repository and install LeRobot in a single layer
COPY . /lerobot
WORKDIR /lerobot

# Create virtual environment and install 
RUN uv pip install --upgrade --no-cache-dir pip
RUN uv pip install .

# # Install huggingface cli
# RUN /opt/venv/bin/pip install -U "huggingface_hub[cli]"
