"""Hardware discovery and safe compute-backend selection.

The module deliberately imports accelerator libraries only inside public
functions. Importing :mod:`jarvis_localhost.hardware` therefore works on a
minimal Python 3.10 installation without PyTorch.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


MIB = 1024**2
GIB = 1024**3
COMPUTE_BACKEND_ENV = "JARVIS_COMPUTE_BACKEND"
CPU_THREADS_ENV = "JARVIS_CPU_THREADS"
INTEROP_THREADS_ENV = "JARVIS_INTEROP_THREADS"


@dataclass(frozen=True)
class HardwareInfo:
    """Non-sensitive hardware facts used to choose local resource limits."""

    system: str
    release: str
    machine: str
    cpu_name: str
    physical_cpu_cores: int
    logical_cpu_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    installed_ram_bytes: int = 0
    gpu_name: str = ""
    gpu_dedicated_memory_bytes: int = 0
    gpu_shared_memory_bytes: int = 0
    directx_version: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_windows(self) -> bool:
        return self.system.casefold() == "windows"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without machine/user identity."""

        result = asdict(self)
        result["sources"] = list(self.sources)
        result["total_ram_mib"] = _bytes_to_mib(self.total_ram_bytes)
        result["available_ram_mib"] = _bytes_to_mib(self.available_ram_bytes)
        result["installed_ram_mib"] = _bytes_to_mib(self.installed_ram_bytes)
        result["gpu_dedicated_memory_mib"] = _bytes_to_mib(
            self.gpu_dedicated_memory_bytes
        )
        result["gpu_shared_memory_mib"] = _bytes_to_mib(
            self.gpu_shared_memory_bytes
        )
        return result


@dataclass(frozen=True)
class BackendAttempt:
    """One backend-selection decision suitable for diagnostics/manifests."""

    backend: str
    status: str
    reason: str


@dataclass(frozen=True)
class DeviceDescriptor:
    """Selected compute device plus an auditable selection explanation.

    ``torch_device`` is the actual value accepted by ``model.to(...)`` and
    ``tensor.to(...)``. It is the literal string ``"cpu"`` when PyTorch is not
    installed, so discovery remains usable before optional ML dependencies are
    installed.
    """

    backend: str
    torch_device: Any = field(repr=False, compare=False)
    display_name: str = "CPU"
    accelerated: bool = False
    reason: str = ""
    smoke_tested: bool = False
    torch_available: bool = False
    torch_version: str = ""
    attempts: tuple[BackendAttempt, ...] = field(default_factory=tuple)

    @property
    def device(self) -> Any:
        """Alias convenient for call sites that expect ``descriptor.device``."""

        return self.torch_device

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe descriptor; never serialize the backend object."""

        return {
            "backend": self.backend,
            "device": _safe_device_string(self.torch_device),
            "display_name": self.display_name,
            "accelerated": self.accelerated,
            "reason": self.reason,
            "smoke_tested": self.smoke_tested,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }


@dataclass(frozen=True)
class CPUThreadSettings:
    """Result of applying bounded CPU thread settings to the current process."""

    intra_op_threads: int
    inter_op_threads: int
    torch_configured: bool
    environment_configured: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


class _BackendUnavailable(RuntimeError):
    pass


class _SmokeTestFailed(RuntimeError):
    pass


class _OptionalModules:
    def __init__(self, loader: Callable[[str], Any]) -> None:
        self._loader = loader
        self._modules: dict[str, Any] = {}
        self._errors: dict[str, Exception] = {}

    def get(self, name: str) -> Any:
        if name in self._modules:
            return self._modules[name]
        if name in self._errors:
            raise self._errors[name]
        try:
            module = self._loader(name)
        except Exception as exc:
            self._errors[name] = exc
            raise
        self._modules[name] = module
        return module

    def loaded(self, name: str) -> Any | None:
        return self._modules.get(name)


SmokeTest = Callable[[Any, Any, str], None]


def detect_hardware(dxdiag_path: str | Path | None = None) -> HardwareInfo:
    """Detect CPU, memory and display facts without requiring ML libraries.

    On Windows, a previously generated ``dxdiag.txt`` augments live detection
    with DirectX and UMA memory information. The packaged hardware snapshot is
    used automatically when present; discovery never sends these facts over the
    network.
    """

    system = platform.system() or "Unknown"
    release = platform.release() or "Unknown"
    machine = platform.machine() or "Unknown"
    logical_cores = max(1, os.cpu_count() or 1)
    physical_cores = max(1, logical_cores // 2)
    total_ram = 0
    available_ram = 0
    sources = ["python-platform"]

    cpu_name = _detect_cpu_name(system)
    psutil_facts = _detect_psutil_facts()
    if psutil_facts:
        physical_cores = psutil_facts["physical_cpu_cores"] or physical_cores
        logical_cores = psutil_facts["logical_cpu_cores"] or logical_cores
        total_ram = psutil_facts["total_ram_bytes"]
        available_ram = psutil_facts["available_ram_bytes"]
        sources.append("psutil")
    elif system.casefold() == "windows":
        total_ram, available_ram = _detect_windows_memory()
        if total_ram:
            sources.append("windows-memory-api")

    dxdiag = _load_dxdiag(dxdiag_path)
    if dxdiag:
        sources.append("dxdiag")
        cpu_name = cpu_name or dxdiag.get("cpu_name", "")
        installed_ram = dxdiag.get("installed_ram_bytes", 0)
        os_ram = dxdiag.get("os_ram_bytes", 0)
        total_ram = total_ram or os_ram or installed_ram
        available_ram = available_ram or total_ram
    else:
        installed_ram = total_ram

    return HardwareInfo(
        system=system,
        release=release,
        machine=machine,
        cpu_name=cpu_name or "Unknown CPU",
        physical_cpu_cores=max(1, physical_cores),
        logical_cpu_cores=max(1, logical_cores),
        total_ram_bytes=max(0, total_ram),
        available_ram_bytes=max(0, available_ram or total_ram),
        installed_ram_bytes=max(0, installed_ram or total_ram),
        gpu_name=dxdiag.get("gpu_name", "") if dxdiag else "",
        gpu_dedicated_memory_bytes=(
            dxdiag.get("gpu_dedicated_memory_bytes", 0) if dxdiag else 0
        ),
        gpu_shared_memory_bytes=(
            dxdiag.get("gpu_shared_memory_bytes", 0) if dxdiag else 0
        ),
        directx_version=dxdiag.get("directx_version", "") if dxdiag else "",
        sources=tuple(sources),
    )


def select_compute_device(
    hardware: HardwareInfo | None = None,
    env: Mapping[str, str] | None = None,
    module_loader: Callable[[str], Any] = importlib.import_module,
    smoke_test: SmokeTest | None = None,
) -> DeviceDescriptor:
    """Select a verified compute backend and fall back safely to CPU.

    Automatic order is native Windows DirectML, CUDA/ROCm, Apple MPS, then
    CPU. Every accelerated candidate must pass an allocation, arithmetic and
    host round-trip smoke test. ``JARVIS_COMPUTE_BACKEND`` may force
    ``directml``, ``cuda``, ``rocm``, ``mps`` or ``cpu``; failed forced
    backends still fall back to CPU with the cause recorded.
    """

    hardware = hardware or detect_hardware()
    environment = os.environ if env is None else env
    preference = environment.get(COMPUTE_BACKEND_ENV, "auto").strip().casefold()
    preference = {"dml": "directml", "gpu": "cuda/rocm"}.get(
        preference, preference
    )
    modules = _OptionalModules(module_loader)
    tester = smoke_test or _default_smoke_test
    attempts: list[BackendAttempt] = []

    candidates = _candidate_order(preference, hardware)
    if candidates is None:
        attempts.append(
            BackendAttempt(
                preference or "<empty>",
                "invalid",
                f"Unsupported {COMPUTE_BACKEND_ENV} value; CPU fallback used.",
            )
        )
        return _cpu_descriptor(modules, attempts, explicitly_requested=False)

    for candidate in candidates:
        if candidate == "cpu":
            return _cpu_descriptor(
                modules,
                attempts,
                explicitly_requested=preference == "cpu",
            )
        selected = _try_accelerated_backend(
            candidate, hardware, modules, tester, attempts
        )
        if selected is not None:
            return selected

    return _cpu_descriptor(modules, attempts, explicitly_requested=False)


def configure_cpu_threads(
    hardware: HardwareInfo | None = None,
    env: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    apply_environment: bool = True,
) -> CPUThreadSettings:
    """Apply conservative CPU threading tuned for physical cores.

    The Ryzen 5 4600G default is six intra-op threads and two inter-op threads,
    leaving SMT capacity for Windows, document ingestion and the web server.
    Environment overrides are clamped to the detected logical CPU count.
    """

    hardware = hardware or detect_hardware()
    environment = os.environ if env is None else env
    logical = max(1, hardware.logical_cpu_cores)
    physical = min(logical, max(1, hardware.physical_cpu_cores))
    warnings: list[str] = []
    intra = _bounded_env_int(
        environment, CPU_THREADS_ENV, physical, 1, logical, warnings
    )
    inter_default = min(2, max(1, physical // 3))
    inter_cap = min(4, physical)
    inter = _bounded_env_int(
        environment, INTEROP_THREADS_ENV, inter_default, 1, inter_cap, warnings
    )

    environment_configured = False
    if apply_environment:
        os.environ["OMP_NUM_THREADS"] = str(intra)
        os.environ["MKL_NUM_THREADS"] = str(intra)
        os.environ["OPENBLAS_NUM_THREADS"] = str(intra)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(intra)
        os.environ["NUMEXPR_NUM_THREADS"] = str(intra)
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        environment_configured = True

    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            torch_module = None

    torch_configured = False
    if torch_module is not None:
        torch_configured = _apply_torch_threads(
            torch_module, intra, inter, warnings
        )
    else:
        warnings.append("PyTorch is unavailable; only process thread variables were set.")

    return CPUThreadSettings(
        intra_op_threads=intra,
        inter_op_threads=inter,
        torch_configured=torch_configured,
        environment_configured=environment_configured,
        warnings=tuple(warnings),
    )


def _candidate_order(
    preference: str, hardware: HardwareInfo
) -> tuple[str, ...] | None:
    valid = {"auto", "directml", "cuda", "rocm", "cuda/rocm", "mps", "cpu"}
    if preference not in valid:
        return None
    if preference != "auto":
        return (preference, "cpu") if preference != "cpu" else ("cpu",)
    candidates: list[str] = []
    if hardware.is_windows:
        candidates.append("directml")
    candidates.extend(("cuda/rocm", "mps", "cpu"))
    return tuple(candidates)


def _try_accelerated_backend(
    candidate: str,
    hardware: HardwareInfo,
    modules: _OptionalModules,
    tester: SmokeTest,
    attempts: list[BackendAttempt],
) -> DeviceDescriptor | None:
    try:
        if candidate == "directml":
            backend, device, name = _directml_device(hardware, modules, tester)
        elif candidate in {"cuda", "rocm", "cuda/rocm"}:
            backend, device, name = _cuda_rocm_device(candidate, modules, tester)
        elif candidate == "mps":
            backend, device, name = _mps_device(modules, tester)
        else:
            raise _BackendUnavailable("Unknown backend candidate.")
    except (ImportError, ModuleNotFoundError, _BackendUnavailable) as exc:
        attempts.append(
            BackendAttempt(candidate, "unavailable", _safe_exception(exc))
        )
        return None
    except (_SmokeTestFailed, OSError, RuntimeError) as exc:
        attempts.append(BackendAttempt(candidate, "failed", _safe_exception(exc)))
        return None
    except Exception as exc:  # third-party backends may raise custom errors
        attempts.append(BackendAttempt(candidate, "failed", _safe_exception(exc)))
        return None

    attempts.append(
        BackendAttempt(backend, "selected", "Accelerator smoke test passed.")
    )
    torch_module = modules.loaded("torch")
    prior_failures = [attempt for attempt in attempts[:-1] if attempt.status != "selected"]
    fallback_detail = ""
    if prior_failures:
        summary = "; ".join(
            f"{attempt.backend}: {attempt.reason}" for attempt in prior_failures
        )
        fallback_detail = f" Earlier candidates were rejected: {summary}"
    return DeviceDescriptor(
        backend=backend,
        torch_device=device,
        display_name=name,
        accelerated=True,
        reason=(
            f"{name} selected through {backend}; allocation, arithmetic and "
            f"CPU round-trip smoke test passed.{fallback_detail}"
        ),
        smoke_tested=True,
        torch_available=True,
        torch_version=_module_version(torch_module),
        attempts=tuple(attempts),
    )


def _directml_device(
    hardware: HardwareInfo,
    modules: _OptionalModules,
    tester: SmokeTest,
) -> tuple[str, Any, str]:
    if not hardware.is_windows:
        raise _BackendUnavailable("DirectML native mode requires Windows.")
    torch_module = modules.get("torch")
    directml = modules.get("torch_directml")
    device_count = getattr(directml, "device_count", None)
    if callable(device_count) and int(device_count()) < 1:
        raise _BackendUnavailable("torch-directml reports no DirectML devices.")
    device = directml.device()
    _run_smoke_test(tester, torch_module, device, "directml")
    name = _directml_device_name(directml) or hardware.gpu_name or "DirectML GPU"
    return "directml", device, name


def _cuda_rocm_device(
    requested: str,
    modules: _OptionalModules,
    tester: SmokeTest,
) -> tuple[str, Any, str]:
    torch_module = modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise _BackendUnavailable("This PyTorch build has no CUDA/ROCm runtime.")
    if not bool(cuda.is_available()):
        raise _BackendUnavailable("PyTorch reports CUDA/ROCm unavailable.")
    device_count = getattr(cuda, "device_count", None)
    if callable(device_count) and int(device_count()) < 1:
        raise _BackendUnavailable("PyTorch reports zero CUDA/ROCm devices.")

    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    backend = "rocm" if hip_version else "cuda"
    if requested in {"cuda", "rocm"} and requested != backend:
        raise _BackendUnavailable(
            f"The installed PyTorch runtime is {backend}, not {requested}."
        )
    device_factory = getattr(torch_module, "device", None)
    device = device_factory("cuda:0") if callable(device_factory) else "cuda:0"
    _run_smoke_test(tester, torch_module, device, backend)
    get_name = getattr(cuda, "get_device_name", None)
    name = str(get_name(0)) if callable(get_name) else f"{backend.upper()} GPU"
    return backend, device, name


def _mps_device(
    modules: _OptionalModules,
    tester: SmokeTest,
) -> tuple[str, Any, str]:
    torch_module = modules.get("torch")
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    is_available = getattr(mps, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise _BackendUnavailable("PyTorch reports MPS unavailable.")
    is_built = getattr(mps, "is_built", None)
    if callable(is_built) and not bool(is_built()):
        raise _BackendUnavailable("This PyTorch build has no MPS support.")
    device_factory = getattr(torch_module, "device", None)
    device = device_factory("mps") if callable(device_factory) else "mps"
    _run_smoke_test(tester, torch_module, device, "mps")
    return "mps", device, "Apple Metal (MPS)"


def _cpu_descriptor(
    modules: _OptionalModules,
    attempts: list[BackendAttempt],
    explicitly_requested: bool,
) -> DeviceDescriptor:
    try:
        torch_module = modules.get("torch")
    except Exception:
        torch_module = None

    device_factory = getattr(torch_module, "device", None)
    try:
        device = device_factory("cpu") if callable(device_factory) else "cpu"
    except Exception:
        device = "cpu"

    if explicitly_requested:
        reason = f"CPU selected explicitly through {COMPUTE_BACKEND_ENV}."
    elif attempts:
        failures = "; ".join(
            f"{attempt.backend}: {attempt.reason}" for attempt in attempts
        )
        reason = f"CPU fallback selected after accelerator checks. {failures}"
    else:
        reason = "CPU selected because no accelerator candidate was requested."
    attempts.append(BackendAttempt("cpu", "selected", reason))
    return DeviceDescriptor(
        backend="cpu",
        torch_device=device,
        display_name="CPU",
        accelerated=False,
        reason=reason,
        smoke_tested=False,
        torch_available=torch_module is not None,
        torch_version=_module_version(torch_module),
        attempts=tuple(attempts),
    )


def _default_smoke_test(torch_module: Any, device: Any, backend: str) -> None:
    del backend
    first = torch_module.tensor(
        [[1.0, 2.0]], device=device, requires_grad=True
    )
    second = torch_module.tensor([[2.0], [1.0]], device=device)
    result = (first @ second).sum()
    result.backward()
    if first.grad is None:
        raise RuntimeError("Accelerator did not produce gradients")
    to_cpu = getattr(result, "to", None)
    result = to_cpu("cpu") if callable(to_cpu) else result
    item = getattr(result, "item", None)
    value = float(item()) if callable(item) else float(result)
    if abs(value - 4.0) > 1e-6:
        raise RuntimeError(f"Unexpected arithmetic result: {value!r}")


def _run_smoke_test(
    tester: SmokeTest, torch_module: Any, device: Any, backend: str
) -> None:
    try:
        tester(torch_module, device, backend)
    except Exception as exc:
        raise _SmokeTestFailed(
            f"{backend} smoke test failed ({_safe_exception(exc)})."
        ) from exc


def _apply_torch_threads(
    torch_module: Any,
    intra: int,
    inter: int,
    warnings: list[str],
) -> bool:
    configured = False
    set_threads = getattr(torch_module, "set_num_threads", None)
    if callable(set_threads):
        try:
            set_threads(intra)
            configured = True
        except RuntimeError as exc:
            warnings.append(f"Could not set PyTorch intra-op threads: {_safe_exception(exc)}")
    set_interop = getattr(torch_module, "set_num_interop_threads", None)
    if callable(set_interop):
        try:
            set_interop(inter)
            configured = True
        except RuntimeError as exc:
            warnings.append(
                f"Could not set PyTorch inter-op threads: {_safe_exception(exc)}"
            )
    return configured


def _detect_cpu_name(system: str) -> str:
    if system.casefold() == "windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value:
                    return " ".join(str(value).split())
        except (ImportError, OSError):
            pass
    return " ".join((platform.processor() or "").split())


def _detect_psutil_facts() -> dict[str, int]:
    try:
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return {
            "physical_cpu_cores": int(psutil.cpu_count(logical=False) or 0),
            "logical_cpu_cores": int(psutil.cpu_count(logical=True) or 0),
            "total_ram_bytes": int(memory.total),
            "available_ram_bytes": int(memory.available),
        }
    except (ImportError, OSError, AttributeError, ValueError):
        return {}


def _detect_windows_memory() -> tuple[int, int]:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0, 0
        return int(status.total_physical), int(status.available_physical)
    except (AttributeError, OSError, ValueError):
        return 0, 0


def _load_dxdiag(dxdiag_path: str | Path | None) -> dict[str, Any]:
    path = Path(dxdiag_path) if dxdiag_path is not None else _default_dxdiag_path()
    if path is None or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return _parse_dxdiag(text)


def _default_dxdiag_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "data" / "hardware" / "dxdiag.txt"
    return candidate if candidate.is_file() else None


def _parse_dxdiag(text: str) -> dict[str, Any]:
    patterns = {
        "cpu_name": r"^\s*Processor:\s*(.+?)\s*$",
        "installed_ram_mib": r"^\s*Memory:\s*(\d+)MB RAM\s*$",
        "os_ram_mib": r"^\s*Available OS Memory:\s*(\d+)MB RAM\s*$",
        "directx_version": r"^\s*DirectX Version:\s*(.+?)\s*$",
        "gpu_name": r"^\s*Card name:\s*(.+?)\s*$",
        "gpu_dedicated_memory_mib": r"^\s*Dedicated Memory:\s*(\d+) MB\s*$",
        "gpu_shared_memory_mib": r"^\s*Shared Memory:\s*(\d+) MB\s*$",
    }
    values: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        values[name] = int(match.group(1)) if name.endswith("_mib") else match.group(1)
    return {
        "cpu_name": values.get("cpu_name", ""),
        "installed_ram_bytes": values.get("installed_ram_mib", 0) * MIB,
        "os_ram_bytes": values.get("os_ram_mib", 0) * MIB,
        "directx_version": values.get("directx_version", ""),
        "gpu_name": values.get("gpu_name", ""),
        "gpu_dedicated_memory_bytes": (
            values.get("gpu_dedicated_memory_mib", 0) * MIB
        ),
        "gpu_shared_memory_bytes": values.get("gpu_shared_memory_mib", 0) * MIB,
    }


def _directml_device_name(directml: Any) -> str:
    get_name = getattr(directml, "device_name", None)
    if not callable(get_name):
        return ""
    try:
        return str(get_name(0)).rstrip("\x00").strip()
    except (RuntimeError, TypeError, ValueError):
        return ""


def _bounded_env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    warnings: list[str],
) -> int:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        requested = int(str(raw).strip())
    except ValueError:
        warnings.append(f"Ignored invalid integer in {name}.")
        return default
    bounded = min(maximum, max(minimum, requested))
    if bounded != requested:
        warnings.append(f"Clamped {name} to the safe range {minimum}..{maximum}.")
    return bounded


def _module_version(module: Any | None) -> str:
    return str(getattr(module, "__version__", "")) if module is not None else ""


def _safe_exception(exc: BaseException) -> str:
    if isinstance(exc, ModuleNotFoundError):
        missing = getattr(exc, "name", None) or "optional module"
        return f"ModuleNotFoundError: {missing} is not installed."
    message = " ".join(str(exc).split())
    message = re.sub(r"[A-Za-z]:\\[^\s;]+", "<local-path>", message)
    message = re.sub(r"/(?:[^\s/]+/)+[^\s;]+", "<local-path>", message)
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _safe_device_string(device: Any) -> str:
    try:
        return str(device)
    except Exception:
        return "<backend-device>"


def _bytes_to_mib(value: int) -> int:
    return int(round(max(0, value) / MIB))


__all__ = [
    "BackendAttempt",
    "COMPUTE_BACKEND_ENV",
    "CPUThreadSettings",
    "CPU_THREADS_ENV",
    "DeviceDescriptor",
    "GIB",
    "HardwareInfo",
    "INTEROP_THREADS_ENV",
    "MIB",
    "configure_cpu_threads",
    "detect_hardware",
    "select_compute_device",
]
