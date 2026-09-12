# Multi-Stage Production Dockerfile for SLM Fine-Tuning & Distillation Pipeline
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Requirements & Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Source Code
COPY . .

# Expose FastAPI Serving Port
EXPOSE 8000

# Default Entrypoint for Inference Serving Layer
CMD ["python3", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
