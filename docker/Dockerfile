FROM nvidia/cuda:11.4.0-base-ubuntu20.04 AS builder
FROM python:3.7 AS core

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        xvfb \
        ffmpeg \
        xorg-dev \
        libsdl2-dev \
        swig \
        cmake \
        unzip \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libosmesa6-dev \
        patchelf \
        gcc \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        libglfw3 \
        libglew-dev

# Download mujoco
RUN mkdir /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -zxf mujoco.tar.gz -C "/root/.mujoco" && \
    rm mujoco.tar.gz

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so

# Copy to image and set workdir
COPY . /home/
WORKDIR /home/

# Upgrade pip3
RUN pip install --upgrade pip

# Run install
# RUN pip3 install .[nomujoco] && pip3 install -r requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Remove the requirements and Docker file - this is no longer needed
RUN rm /home/requirements.txt
RUN rm /home/Dockerfile

# Install hybrid-rl and shadowhand-gym and prepare to run the experiments
RUN chmod +x "./update.sh"
RUN ./update.sh

ENTRYPOINT ["/bin/bash"]