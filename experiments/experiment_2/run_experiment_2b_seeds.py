#!/usr/bin/env python3
"""Train and evaluate preregistered Experiment 2B seeds 42--46 in order."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
SEEDS = (42, 43, 44, 45, 46)
TRAINING_MARKERS = (
    "run_metadata.json",
    "initial_shared/config.json",
    "HIGH_pre_npo/config.json",
    "LOW_pre_npo/config.json",
    "PLACEBO_pre_npo/config.json",
    "HIGH_final/config.json",
    "LOW_final/config.json",
    "PLACEBO_final/config.json",
)
EVALUATION_MARKERS = (
    "evaluation/per_sample_scores.csv",
    "evaluation/primary_contrast.csv",
    "evaluation/evaluation_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train then evaluate Experiment 2B seeds 42 through 46. "
            "Complete seeds are skipped and partial outputs fail closed."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ruli-root", type=Path, default=Path("/workspace/Ruli"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "results" / "intervention_manifest.json",
    )
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=SCRIPT_DIR / "results" / "experiment_2b",
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _all_markers_exist(root: Path, markers: tuple[str, ...]) -> bool:
    return all((root / marker).is_file() for marker in markers)


def _any_markers_exist(root: Path, markers: tuple[str, ...]) -> bool:
    return any((root / marker).exists() for marker in markers)


def _cleanup_completed_trainer_work(seed_root: Path) -> bool:
    """Remove only Trainer scratch belonging to a marker-complete seed."""

    resolved_seed = seed_root.resolve()
    if not _all_markers_exist(resolved_seed, TRAINING_MARKERS):
        raise RuntimeError(
            "Refusing Trainer cleanup because training markers are incomplete: "
            f"{resolved_seed}"
        )
    scratch = (resolved_seed / "trainer_work").resolve()
    if scratch.parent != resolved_seed or scratch.name != "trainer_work":
        raise RuntimeError(f"Refusing unexpected Trainer cleanup target: {scratch}")
    if not scratch.exists():
        return False
    if not scratch.is_dir():
        raise RuntimeError(f"Trainer work target is not a directory: {scratch}")
    shutil.rmtree(scratch)
    if scratch.exists():
        raise RuntimeError(f"Trainer work cleanup did not finish: {scratch}")
    print(f"[CLEANUP] Removed completed-seed Trainer scratch: {scratch}", flush=True)
    return True


def _run(command: list[str], log_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{timestamp}] $ {' '.join(command)}\n")
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    output_base = args.output_base.resolve()
    log_dir = (args.log_dir or output_base / "logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    shadow_path = args.shadow_path or (
        args.ruli_root
        / "core"
        / "attack"
        / "attack_inferences"
        / "WikiText103"
        / "shadow_9_attack_random_npo_gpt2.pth"
    )
    target_data_path = args.target_data_path or (
        args.ruli_root
        / "text"
        / "data"
        / "WikiText-103-local"
        / "gpt2"
        / "selective_dataset_prefixed_smoke_700"
    )
    common = [
        "--ruli-root",
        str(args.ruli_root.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--shadow-path",
        str(shadow_path.resolve()),
        "--target-data-path",
        str(target_data_path.resolve()),
        "--device",
        args.device,
    ]

    for seed in SEEDS:
        seed_root = output_base / f"seed_{seed}"
        log_path = log_dir / f"seed_{seed}.log"
        training_complete = _all_markers_exist(seed_root, TRAINING_MARKERS)
        evaluation_complete = _all_markers_exist(seed_root, EVALUATION_MARKERS)
        if training_complete and evaluation_complete:
            _cleanup_completed_trainer_work(seed_root)
            print(f"[SKIP] Seed {seed} is already complete: {seed_root}", flush=True)
            continue
        if not training_complete and (
            seed_root.exists() or _any_markers_exist(seed_root, TRAINING_MARKERS)
        ):
            raise RuntimeError(
                f"Seed {seed} has partial training output; refusing to rerun or "
                f"overwrite it: {seed_root}"
            )
        if training_complete and _any_markers_exist(seed_root, EVALUATION_MARKERS):
            raise RuntimeError(
                f"Seed {seed} has partial evaluation output; refusing to overwrite "
                f"it: {seed_root / 'evaluation'}"
            )

        if not training_complete:
            print(f"[RUN] Seed {seed} training; log: {log_path}", flush=True)
            _run(
                [
                    args.python,
                    str(SCRIPT_DIR / "run_experiment_2b.py"),
                    "--seed",
                    str(seed),
                    "--output-root",
                    str(seed_root),
                    *common,
                ],
                log_path,
            )
            if not _all_markers_exist(seed_root, TRAINING_MARKERS):
                raise RuntimeError(
                    f"Seed {seed} training exited successfully but completion "
                    f"markers are missing: {seed_root}"
                )

        _cleanup_completed_trainer_work(seed_root)

        print(f"[RUN] Seed {seed} evaluation; log: {log_path}", flush=True)
        _run(
            [
                args.python,
                str(SCRIPT_DIR / "evaluate_experiment_2b.py"),
                "--seed",
                str(seed),
                "--experiment-output",
                str(seed_root),
                *common,
            ],
            log_path,
        )
        print(f"[DONE] Seed {seed}", flush=True)


if __name__ == "__main__":
    main()
