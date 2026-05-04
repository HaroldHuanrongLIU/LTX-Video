# Repository Guidelines

## Project Structure & Module Organization

`ltx_video/` contains the importable package. Core inference entry points live in `ltx_video/inference.py`, pipelines in `ltx_video/pipelines/`, model definitions in `ltx_video/models/`, schedulers in `ltx_video/schedulers/`, and shared helpers in `ltx_video/utils/`. The top-level `inference.py` is the CLI wrapper used by the README examples. Model and pipeline presets are YAML files under `configs/`. Tests live in `tests/`, with sample media in `tests/utils/`. Documentation media and README assets are in `docs/_static/`.

## Build, Test, and Development Commands

Create a local environment and install editable dependencies:

```bash
python -m venv env
source env/bin/activate
python -m pip install -e ".[inference,test]"
```

Run the full test suite with `pytest`. For a focused check, target a file such as `pytest tests/test_scheduler.py`. Use `pytest -m "not slow"` when skipping slow config-based inference tests. Inspect CLI options with `python inference.py --help`, or run inference with a config, for example:

```bash
python inference.py --prompt "A video of a cat" --height 256 --width 320 --num_frames 33 --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml
```

## Coding Style & Naming Conventions

Python code targets Python 3.10+ and follows Black formatting. Ruff is used for linting and autofixes through pre-commit. Use 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and descriptive dataclass/config field names. Keep config filenames consistent with existing presets, for example `configs/ltxv-2b-0.9.8-distilled.yaml`.

## Testing Guidelines

Tests use `pytest` and should be named `test_*.py` with functions named `test_*`. Prefer synthetic fixtures, temporary paths, and small frame counts as shown in `tests/conftest.py` and `tests/test_inference.py`. Mark hardware-heavy or full-config checks with `@pytest.mark.slow`; avoid making routine tests depend on external checkpoints unless the test explicitly covers download or integration behavior.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects, often scoped by area, such as `Readme: Add ...` or `Inference: Integrate ...`. Keep subjects concise and explain behavior changes in the body when needed. Pull requests should describe the affected model, pipeline, config, or test area; list the commands run; note CUDA/MPS or FP8 hardware assumptions; and include generated media or screenshots when changing visible inference output.

## Security & Configuration Tips

Do not commit model weights, generated videos, tokens, or local environment folders. Keep large binary assets in the intended tracked locations only, and prefer configurable paths over hard-coded machine-specific paths.
