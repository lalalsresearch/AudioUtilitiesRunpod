FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip once
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install dependencies
COPY YoutubeDownloader/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY utils/ utils/
COPY YoutubeDownloader/ YoutubeDownloader/