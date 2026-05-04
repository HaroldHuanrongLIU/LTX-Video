from __future__ import annotations

from typing import Optional, Sequence

from benchmark.surgwmbench import BaselineSpec, run_cli


SPEC = BaselineSpec(
    baseline="ltx_video",
    model="LTX-Video",
    native_entrypoint="inference.py",
    native_train_entrypoint=None,
    native_frame_predictor=(
        "upstream inference.py supports image/video conditioning; full SurgWMBench "
        "future-frame evaluation requires selecting an LTX pipeline config/checkpoint "
        "and mapping context clips to conditioning media."
    ),
    notes=(
        "LTX-Video is the closest native image/video-conditioned generator among the "
        "remaining baselines. The adapter smoke path is CPU-only; native execution "
        "should use inference.py with a configured pipeline YAML and checkpoint assets."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
