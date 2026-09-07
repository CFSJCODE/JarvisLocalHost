"""Emit a non-sensitive JSON hardware and training-resource probe."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jarvis_localhost.hardware import (  # noqa: E402 - direct script support
    build_training_profile,
    configure_cpu_threads,
    detect_corpus_stats,
    detect_hardware,
    select_compute_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect local compute backends and print a JSON resource profile. "
            "No document text, paths, credentials or environment values are emitted."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        type=Path,
        default=[],
        help="Local corpus file/directory to measure; may be repeated.",
    )
    parser.add_argument(
        "--dxdiag",
        type=Path,
        default=None,
        help="Optional local dxdiag text report used only for hardware facts.",
    )
    parser.add_argument(
        "--configure-cpu-threads",
        action="store_true",
        help="Also apply the recommended bounded CPU thread settings.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of indented JSON.",
    )
    return parser


def create_report(args: argparse.Namespace) -> dict[str, object]:
    hardware = detect_hardware(args.dxdiag)
    device = select_compute_device(hardware)
    corpus_input = args.corpus if args.corpus else None
    corpus = detect_corpus_stats(corpus_input)
    profile = build_training_profile(
        hardware=hardware,
        corpus_stats=corpus,
        device=device,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "hardware": hardware.to_dict(),
        "device": device.to_dict(),
        "corpus": corpus.to_dict(),
        "training_profile": profile.to_dict(),
        "runtime_policy": {
            "pretrained_weights": False,
            "runtime_downloads_allowed": False,
        },
    }
    if args.configure_cpu_threads:
        report["cpu_thread_settings"] = configure_cpu_threads(hardware).to_dict()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = create_report(args)
    except Exception as exc:
        # Keep CLI failures machine-readable without echoing paths, environment
        # values or third-party exception payloads that could contain secrets.
        report = {
            "schema_version": 1,
            "status": "error",
            "error_type": type(exc).__name__,
        }
        print(json.dumps(report, sort_keys=True))
        return 1
    indent = None if args.compact else 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
