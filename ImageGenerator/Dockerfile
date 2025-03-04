# Use a base image from Runpod
FROM runpod/base:0.6.2-cuda12.1.0 AS base

# Install CUDA dependencies
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y ffmpeg cuda-toolkit cudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CUDA-compatible ONNX Runtime
RUN python3 -m pip install --upgrade pip && \
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Create a new stage for dependencies
FROM base AS dependencies

WORKDIR /app

# Copy and install Python dependencies separately for better caching
COPY ImageGenerator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Final stage to assemble everything
FROM dependencies AS final

WORKDIR /app

# Copy only the application code (avoiding unnecessary rebuilds)
COPY ImageGenerator/image_generator_runpod.py image_generator_runpod.py
COPY ImageGenerator/image_generator.py image_generator.py
COPY utils/ utils/

CMD ["python3", "image_generator_runpod.py"]