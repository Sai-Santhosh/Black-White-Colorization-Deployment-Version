"""
Image preprocessing and postprocessing utilities for Lab color space.
Handles grayscale/BGR conversion for OpenCV compatibility.
"""
import logging
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color

logger = logging.getLogger(__name__)


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format. Handles BGR (OpenCV) and grayscale.

    Args:
        img: Input image (BGR, grayscale, or RGB)

    Returns:
        RGB numpy array (H, W, 3)
    """
    if img.ndim == 2:
        return np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_img(img_path: str) -> np.ndarray:
    """Load image from path and return RGB array."""
    out_np = np.asarray(Image.open(img_path).convert("RGB"))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], (1, 1, 3))
    return out_np


def resize_img(img: np.ndarray, HW: Tuple[int, int] = (256, 256), resample: int = 3) -> np.ndarray:
    """Resize image to target dimensions."""
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(
    img_input: np.ndarray,
    HW: Tuple[int, int] = (256, 256),
    resample: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess image for colorization model.

    Handles BGR (OpenCV) and grayscale inputs. Returns original-size L and
    resized L as tensors for the model.

    Args:
        img_input: Input frame (BGR or grayscale from OpenCV)
        HW: Resize dimensions for model input
        resample: PIL resample method

    Returns:
        (tens_orig_l, tens_rs_l) - L channel tensors
    """
    img_rgb = _ensure_rgb(img_input)
    img_rgb_rs = resize_img(img_rgb, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.from_numpy(img_l_orig).float()[None, None, :, :]
    tens_rs_l = torch.from_numpy(img_l_rs).float()[None, None, :, :]

    return tens_orig_l, tens_rs_l


def postprocess_tens(
    tens_orig_l: torch.Tensor,
    out_ab: torch.Tensor,
    mode: str = "bilinear",
) -> np.ndarray:
    """
    Convert model output (ab channels) + L to RGB image.

    Args:
        tens_orig_l: Original L channel (1, 1, H, W)
        out_ab: Model output ab channels (1, 2, H, W)
        mode: Interpolation mode for size mismatch

    Returns:
        RGB numpy array (H, W, 3) in [0, 1]
    """
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    rgb = color.lab2rgb(out_lab_orig.data.cpu().numpy()[0].transpose((1, 2, 0)))
    return np.clip(rgb, 0, 1).astype(np.float32)
