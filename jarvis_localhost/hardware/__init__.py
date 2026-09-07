"""Optional, side-effect-free hardware abstraction for local J.A.R.V.I.S."""

from .device import (
    BackendAttempt,
    CPUThreadSettings,
    DeviceDescriptor,
    HardwareInfo,
    configure_cpu_threads,
    detect_hardware,
    select_compute_device,
)
from .profiles import (
    CorpusStats,
    TrainingResourceProfile,
    build_training_profile,
    detect_corpus_stats,
)


__all__ = [
    "BackendAttempt",
    "CPUThreadSettings",
    "CorpusStats",
    "DeviceDescriptor",
    "HardwareInfo",
    "TrainingResourceProfile",
    "build_training_profile",
    "configure_cpu_threads",
    "detect_corpus_stats",
    "detect_hardware",
    "select_compute_device",
]
