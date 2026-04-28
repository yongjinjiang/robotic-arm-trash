from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demos"
VIDEO_DIR = PROJECT_ROOT / "videos"


def get_project_summary() -> str:
    return (
        "robotic-arm-trash: small Gymnasium/MuJoCo experiments for a "
        "Reacher-style robotic arm task.\n"
        f"Demo scripts: {DEMO_DIR}\n"
        f"Generated videos: {VIDEO_DIR}\n"
        "Run demos with `uv run python demos/reacher_random_demo.py` or "
        "`uv run python demos/record_reacher_video.py`."
    )


def main() -> None:
    print(get_project_summary())


if __name__ == "__main__":
    main()
