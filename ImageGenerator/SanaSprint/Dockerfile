# Use official PyTorch image as the base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY ImageGenerator/SanaSprint/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# Copy project files
COPY utils/ utils/
COPY ImageGenerator/SanaSprint/ ./

CMD ["python3", "image_generator_beam.py"]