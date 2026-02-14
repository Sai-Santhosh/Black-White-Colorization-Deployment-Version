"""
Production-grade video colorization pipeline.
Processes black & white videos frame-by-frame with proper error handling.
"""
import logging
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..config import DEFAULT_FRAME_SIZE, MAX_VIDEO_FRAMES, SUPPORTED_VIDEO_FORMATS
from ..core import load_colorizer, postprocess_tens, preprocess_img

logger = logging.getLogger(__name__)


class VideoColorizer:
    """
    End-to-end video colorization pipeline.
    Reads B&W video, colorizes each frame, writes output video.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
        model_cache_dir: Optional[Path] = None,
    ) -> None:
        self.frame_size = frame_size
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = load_colorizer(
            pretrained=True,
            device=self.device,
            cache_dir=model_cache_dir,
        )

    def _validate_input(self, input_path: Path) -> None:
        """Validate input file exists and is supported."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        suffix = input_path.suffix.lower()
        if suffix not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
            )

    def _get_video_properties(self, cap: cv2.VideoCapture) -> dict:
        """Extract video properties for output writer."""
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        return {"fps": fps, "width": width, "height": height, "frame_count": frame_count}

    def _colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Colorize a single frame. Frame is BGR from OpenCV."""
        tens_orig_l, tens_rs_l = preprocess_img(frame, HW=self.frame_size)
        tens_orig_l = tens_orig_l.to(self.device)
        tens_rs_l = tens_rs_l.to(self.device)

        with torch.no_grad():
            out_ab = self.model(tens_rs_l)

        rgb = postprocess_tens(tens_orig_l.cpu(), out_ab.cpu())
        bgr = (rgb[:, :, ::-1] * 255).astype(np.uint8)
        return bgr

    def process_frames(
        self,
        input_path: Path,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, int, Optional[int]], None, None]:
        """
        Generator that yields colorized frames.

        Yields:
            (frame, frame_index, total_frames)
        """
        self._validate_input(input_path)
        cap = cv2.VideoCapture(str(input_path))

        try:
            props = self._get_video_properties(cap)
            total = min(
                props["frame_count"] or MAX_VIDEO_FRAMES,
                max_frames or MAX_VIDEO_FRAMES,
            )
            frame_idx = 0

            while frame_idx < (max_frames or MAX_VIDEO_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    colorized = self._colorize_frame(frame)
                    yield colorized, frame_idx, props["frame_count"] or None
                except Exception as e:
                    logger.warning("Frame %d failed: %s", frame_idx, e)
                    # Fallback: return original in color (cv2 will convert)
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    yield frame, frame_idx, props["frame_count"] or None

                frame_idx += 1
        finally:
            cap.release()

    def colorize_video(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Colorize video and write to output file.

        Args:
            input_path: Path to input video
            output_path: Path for output. Auto-generated if None.
            max_frames: Limit frames (for testing). None = process all.
            progress_callback: Optional fn(frame_idx, total) for progress updates

        Returns:
            Path to output video
        """
        self._validate_input(input_path)
        cap = cv2.VideoCapture(str(input_path))
        props = self._get_video_properties(cap)
        cap.release()

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_colorized{input_path.suffix}"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            props["fps"],
            (props["width"], props["height"]),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        try:
            total = props["frame_count"] or 0
            for frame, idx, _ in self.process_frames(input_path, max_frames=max_frames):
                writer.write(frame)
                if progress_callback and total > 0:
                    progress_callback(idx + 1, total)
        finally:
            writer.release()

        logger.info("Colorized video saved to %s", output_path)
        return output_path


def colorize_video_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Convenience function to colorize a video file.

    Args:
        input_path: Input video path
        output_path: Output path (optional)
        use_gpu: Use GPU if available
        max_frames: Max frames to process (optional)
        progress_callback: Optional progress callback

    Returns:
        Path to output video
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else None
    colorizer = VideoColorizer(use_gpu=use_gpu)
    return colorizer.colorize_video(
        input_path,
        output_path=output_path,
        max_frames=max_frames,
        progress_callback=progress_callback,
    )
