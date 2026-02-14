"""
Command-line interface for video colorization.
Use for batch processing or scripting.
"""
import argparse
import logging
import sys
from pathlib import Path

from pipeline import colorize_video_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Colorize black & white videos"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video path",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output video path (default: <input>_colorized.<ext>)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "-n", "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames (for testing)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        output = colorize_video_file(
            args.input,
            output_path=args.output,
            use_gpu=args.gpu,
            max_frames=args.max_frames,
        )
        print(f"Done. Output: {output}")
        return 0
    except Exception as e:
        logging.exception("Colorization failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
