"""
Black & White Video Colorization - Production Gradio App
Deployable on Hugging Face Spaces and locally.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import torch

from config import (
    DEFAULT_FRAME_SIZE,
    USE_GPU,
)
from pipeline import VideoColorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize colorizer (lazy load on first use to avoid startup delay)
_colorizer: Optional[VideoColorizer] = None


def get_colorizer() -> VideoColorizer:
    """Lazy-load the colorizer to speed up app startup."""
    global _colorizer
    if _colorizer is None:
        use_gpu = USE_GPU and torch.cuda.is_available()
        logger.info("Initializing VideoColorizer (GPU=%s)", use_gpu)
        _colorizer = VideoColorizer(
            use_gpu=use_gpu,
            frame_size=DEFAULT_FRAME_SIZE,
            model_cache_dir=None,
        )
    return _colorizer


def colorize_video_gradio(
    video_file: Optional[str],
    max_frames: Optional[int],
    progress=gr.Progress(),
):
    """
    Gradio interface for video colorization.

    Returns:
        (output_video_path, status_message)
    """
    if video_file is None:
        return None, "❌ Please upload a black & white video first."

    # Handle Gradio returning dict with "path" key in some versions
    if isinstance(video_file, dict):
        video_file = video_file.get("path") or video_file.get("name", "")
    input_path = Path(str(video_file))

    if not input_path.exists():
        return None, f"❌ File not found: {input_path}"

    suffix = input_path.suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return None, f"❌ Unsupported format: {suffix}. Use MP4, AVI, MOV, MKV, or WebM."

    # Create temp output
    output_dir = Path(tempfile.mkdtemp())
    output_path = output_dir / f"colorized_{input_path.name}"

    try:
        colorizer = get_colorizer()

        def update_progress(current: int, total: int) -> None:
            if total > 0:
                progress((current, total), desc=f"Colorizing frame {current}/{total}")

        output_path = colorizer.colorize_video(
            input_path,
            output_path=output_path,
            max_frames=max_frames if max_frames and max_frames > 0 else None,
            progress_callback=update_progress,
        )

        if output_path.exists():
            return str(output_path), f"✅ Colorization complete! Processed video ready for download."
        return None, "❌ Output file was not created."

    except FileNotFoundError as e:
        logger.exception("File not found")
        return None, f"❌ Error: {e}"
    except Exception as e:
        logger.exception("Colorization failed")
        return None, f"❌ Colorization failed: {str(e)}"


# Custom CSS for premium dark UI
CUSTOM_CSS = """
/* Main container - cinematic dark theme */
.gradio-container {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Header styling */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
}

/* Primary button - vibrant gradient */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

.primary:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    transform: translateY(-1px);
}

/* File upload area */
.upload-area {
    border: 2px dashed rgba(102, 126, 234, 0.5) !important;
    border-radius: 12px !important;
}

/* Video preview cards */
.video-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2) !important;
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Footer info */
.footer-info {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin-top: 2rem;
}
"""


def create_interface() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="B&W Video Colorization | AI-Powered",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🎬 Black & White Video Colorization
            **AI-powered colorization pipeline** — Transform classic B&W footage into vivid color.
            """
        )

        gr.Markdown(
            "Upload a black & white video (MP4, AVI, MOV, MKV, WebM). "
            "The model uses deep learning to restore realistic colors frame-by-frame."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📤 Upload B&W Video",
                    format="mp4",
                )
                max_frames_input = gr.Number(
                    label="Max Frames (0 = process all)",
                    value=0,
                    minimum=0,
                    precision=0,
                    info="Limit frames for faster testing",
                )
                colorize_btn = gr.Button(
                    "✨ Colorize Video",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="🎨 Colorized Output",
                    format="mp4",
                )
                status_output = gr.Markdown(
                    value="Upload a video and click **Colorize** to begin.",
                )

        # Examples section (optional - add sample if available)
        gr.Markdown("---")
        gr.Markdown(
            """
            ### How it works
            1. **Upload** — Drag & drop your B&W video
            2. **Process** — AI colorizes each frame using Lab color space
            3. **Download** — Get your colorized video

            *Built with PyTorch • Zhang et al. colorization architecture*
            """
        )

        colorize_btn.click(
            fn=colorize_video_gradio,
            inputs=[video_input, max_frames_input],
            outputs=[video_output, status_output],
        )

    return demo


def main() -> None:
    """Run the Gradio app."""
    from config import API_HOST, API_PORT, ensure_dirs

    ensure_dirs()

    demo = create_interface()

    demo.launch(
        server_name=API_HOST,
        server_port=API_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
