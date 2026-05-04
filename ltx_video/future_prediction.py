from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from benchmark.surgwmbench import BaselineSpec, run_cli


def predict_native_ltx_frames(
    dataset_root: Path,
    window: object,
    args: argparse.Namespace,
) -> List[np.ndarray]:
    request = build_native_request(dataset_root, window, args)
    request_path = write_native_request(request, args)
    pipeline_config = resolve_pipeline_config(args)
    if pipeline_config is None:
        raise RuntimeError(
            "LTX-Video native prediction requires a real pipeline config YAML via "
            "--checkpoint or LTX_VIDEO_PIPELINE_CONFIG, not the adapter metadata "
            f"checkpoint: {args.checkpoint}. Wrote native request to {request_path}."
        )

    output_dir = args.output.parent / "native_ltx_outputs" / f"{safe_id(window.clip_id)}_h{args.prediction_horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "inference.py",
        "--prompt",
        str(request["prompt"]),
        "--conditioning_media_paths",
        str(request["conditioning_media_paths"][0]),
        "--conditioning_start_frames",
        "0",
        "--height",
        str(args.image_size),
        "--width",
        str(args.image_size),
        "--num_frames",
        str(request["ltx_num_frames"]),
        "--frame_rate",
        "8",
        "--seed",
        str(args.seed),
        "--pipeline_config",
        str(pipeline_config),
        "--output_path",
        str(output_dir),
    ]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LTX-Video native inference failed. "
            f"Request: {request_path}. Command: {' '.join(command)}. "
            f"stderr tail: {completed.stderr[-2000:]}"
        )
    video_path = newest_video(output_dir)
    if video_path is None:
        raise RuntimeError(
            f"LTX-Video native inference completed but did not write an mp4 to {output_dir}. "
            f"Request: {request_path}. stdout tail: {completed.stdout[-2000:]}"
        )
    return read_future_frames(video_path, int(args.prediction_horizon), int(args.image_size))


def build_native_request(dataset_root: Path, window: object, args: argparse.Namespace) -> Dict[str, Any]:
    first_frame_path = dataset_root / window.context_frame_paths[-1]
    ltx_num_frames = ltx_frame_count_for_horizon(int(args.prediction_horizon))
    prompt = os.environ.get(
        "LTX_VIDEO_PROMPT",
        "A surgical instrument continues moving smoothly in an endoscopic surgical scene.",
    )
    return {
        "dataset_name": "SurgWMBench",
        "baseline": "ltx_video",
        "model": "LTX-Video",
        "clip_id": window.clip_id,
        "data_track": window.data_track,
        "prediction_task": args.prediction_task,
        "context_frames": args.context_frames,
        "prediction_horizon": args.prediction_horizon,
        "conditioning_media_paths": [str(first_frame_path)],
        "conditioning_start_frames": [0],
        "prompt": prompt,
        "height": args.image_size,
        "width": args.image_size,
        "ltx_num_frames": ltx_num_frames,
        "pipeline_config": str(resolve_pipeline_config(args)) if resolve_pipeline_config(args) else None,
        "native_entrypoint": "inference.py",
        "notes": (
            "LTX-Video is used as an image-conditioned future-frame generator. "
            "The adapter conditions on the last context anchor frame and slices the "
            "generated sequence to the requested future horizon."
        ),
    }


def write_native_request(request: Dict[str, Any], args: argparse.Namespace) -> Path:
    request_dir = args.output.parent / "native_requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{safe_id(str(request['clip_id']))}_h{args.prediction_horizon}.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    return request_path


def resolve_pipeline_config(args: argparse.Namespace) -> Optional[Path]:
    configured = os.environ.get("LTX_VIDEO_PIPELINE_CONFIG")
    checkpoint = Path(str(args.checkpoint))
    candidate = Path(configured) if configured else checkpoint
    if candidate.suffix not in {".yaml", ".yml"}:
        return None
    if candidate.is_absolute() and candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate
    if candidate.exists():
        return candidate
    return None


def ltx_frame_count_for_horizon(horizon: int) -> int:
    requested_total = horizon + 1
    return ((requested_total - 2) // 8 + 1) * 8 + 1


def newest_video(output_dir: Path) -> Optional[Path]:
    videos = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def read_future_frames(video_path: Path, horizon: int, image_size: int) -> List[np.ndarray]:
    try:
        import imageio.v3 as imageio_v3
    except Exception as exc:  # pragma: no cover - depends on native LTX deps.
        raise RuntimeError("Reading LTX-Video native outputs requires imageio.") from exc
    frames = imageio_v3.imread(video_path)
    if frames.shape[0] <= horizon:
        raise RuntimeError(
            f"LTX-Video output has {frames.shape[0]} frames, which is not enough for horizon {horizon}"
        )
    output: List[np.ndarray] = []
    for frame in frames[1 : horizon + 1]:
        frame_uint8 = np.asarray(frame, dtype=np.uint8)
        if frame_uint8.shape[:2] != (image_size, image_size):
            frame_uint8 = np.asarray(Image.fromarray(frame_uint8).resize((image_size, image_size)))
        output.append(frame_uint8[:, :, :3])
    return output


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


SPEC = BaselineSpec(
    baseline="ltx_video",
    model="LTX-Video",
    native_entrypoint="inference.py",
    native_train_entrypoint=None,
    native_frame_predictor=(
        "native image-conditioned future-frame prediction is wired through inference.py "
        "when --checkpoint or LTX_VIDEO_PIPELINE_CONFIG points to an LTX pipeline YAML."
    ),
    native_frame_predictor_fn=predict_native_ltx_frames,
    notes=(
        "LTX-Video is the closest native image/video-conditioned generator among the "
        "remaining baselines. The adapter conditions on the last context anchor frame; "
        "trajectory outputs still use the shared deterministic coordinate head unless "
        "a trajectory-conditioned LTX control path is added."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
