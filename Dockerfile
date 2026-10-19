# FallGuard Production Containerfile
FROM python:3.11-slim

# System dependencies: install ffmpeg for 20s opt-in incident video clip joining and clean apt lists
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy source tree
COPY . .

# Expose backend port
EXPOSE 8000

# Keep a single uvicorn worker as specified
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
