"""
Production configuration for Black & White Video Colorization Pipeline.
Environment variables override defaults for deployment flexibility.
"""
import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE", str(BASE_DIR / ".model_cache")))

# Model configuration
MODEL_URL = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
MODEL_CHECKSUM = "9b330a0b"

# Processing configuration
DEFAULT_FRAME_SIZE = (256, 256)
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "10000"))  # Safety limit
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

# Device configuration
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "7860"))

# Hugging Face Spaces
HF_SPACE = os.getenv("SPACE_ID") is not None


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    for d in (DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
