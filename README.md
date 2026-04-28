# robotic-arm-trash

Small Gymnasium/MuJoCo experiments for a Reacher-style robotic arm environment.

This repository currently focuses on two things:
- a lightweight Python package/CLI entry point
- demo scripts that run a random-policy Reacher environment and optionally record video output

## Requirements

- Python 3.11+
- `uv` for environment management
- A machine setup capable of running Gymnasium's MuJoCo-based `Reacher-v5` environment

## Quick start

```bash
uv sync
uv run robotic-arm-trash
```

That command prints a short project summary and tells you where the demo scripts live.

## Demo scripts

The repo includes runnable demos under `demos/`:

- `demos/reacher_random_demo.py`
  - runs a random policy in `Reacher-v5`
- `demos/record_reacher_video.py`
  - records rollout videos into `videos/`

Run them with:

```bash
uv run python demos/reacher_random_demo.py
uv run python demos/record_reacher_video.py
```

## Tests

This repo now contains a minimal real pytest suite under `tests/`.

Run tests with:

```bash
uv run pytest -q
```

## Notes

- Generated rollout videos are ignored by git and are not intended to be committed.
- If you run into rendering issues on headless machines, you may need to configure MuJoCo/OpenGL settings for your environment.
