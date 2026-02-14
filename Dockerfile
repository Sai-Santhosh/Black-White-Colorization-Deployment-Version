# Black & White Video Colorization - Production Docker
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/input data/output .model_cache

ENV API_HOST=0.0.0.0
ENV API_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
