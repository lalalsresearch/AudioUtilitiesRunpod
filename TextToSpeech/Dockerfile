# Base image with PyTorch, CUDA, and cuDNN
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS base

# Install system dependencies
RUN apt update && apt install -y \
    git \
    ffmpeg \
    espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Create a new stage for cloning and setting up the repository
FROM base AS build

WORKDIR /app

# Clone Zonos repository separately for better caching
RUN git clone https://github.com/Zyphra/Zonos.git /app


# Install dependencies with UV (better caching)
RUN uv pip install --system -e . && \
    uv pip install --system -e .[compile]

# Create a build-requirement stage for requirements
FROM build AS build-requirement

WORKDIR /app

# Copy additional requirements and install Python dependencies
COPY TextToSpeech/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY TextToSpeech/ ./
COPY utils/ utils/