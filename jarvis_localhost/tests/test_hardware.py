"""Tests for the optional hardware layer; no real PyTorch import is required."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_localhost.hardware import (
    CorpusStats,
    DeviceDescriptor,
    HardwareInfo,
    build_training_profile,
    configure_cpu_threads,
    detect_corpus_stats,
    detect_hardware,
    select_compute_device,
)
from jarvis_localhost.hardware.device import GIB, MIB
from jarvis_localhost.hardware.profiles import (
    MAX_ACCELERATOR_MEMORY_BUDGET_BYTES,
    MAX_CONTEXT_LENGTH,
    MAX_HOST_MEMORY_BUDGET_BYTES,
    MAX_TRAINING_STEPS,
)


class FakeCuda:
    def __init__(self, available: bool = False, count: int = 0) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"


class FakeMps:
    def __init__(self, available: bool = False) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def is_built(self) -> bool:
        return self._available


class FakeTorch:
    __version__ = "2.4.1-fake"

    def __init__(
        self,
        cuda_available: bool = False,
        hip_version: str | None = None,
        mps_available: bool = False,
    ) -> None:
        self.cuda = FakeCuda(cuda_available, 1 if cuda_available else 0)
        self.version = SimpleNamespace(hip=hip_version)
        self.backends = SimpleNamespace(mps=FakeMps(mps_available))
        self.intra_threads: list[int] = []
        self.interop_threads: list[int] = []

    @staticmethod
    def device(name: str) -> str:
        return name

    def set_num_threads(self, count: int) -> None:
        self.intra_threads.append(count)

    def set_num_interop_threads(self, count: int) -> None:
        self.interop_threads.append(count)


class FakeDirectML:
    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def device() -> str:
        return "privateuseone:0"

    @staticmethod
    def device_name(index: int) -> str:
        return f"Fake DirectML Radeon {index}\x00"


def target_hardware(system: str = "Windows") -> HardwareInfo:
    return HardwareInfo(
        system=system,
        release="11",
        machine="AMD64",
        cpu_name="AMD Ryzen 5 4600G with Radeon Graphics",
        physical_cpu_cores=6,
        logical_cpu_cores=12,
        total_ram_bytes=32 * GIB,
        available_ram_bytes=24 * GIB,
        installed_ram_bytes=40 * GIB,
        gpu_name="AMD Radeon(TM) Graphics",
        gpu_dedicated_memory_bytes=8_175 * MIB,
        gpu_shared_memory_bytes=16_277 * MIB,
        directx_version="DirectX 12",
        sources=("test",),
    )


def accelerated_device(backend: str = "directml") -> DeviceDescriptor:
    return DeviceDescriptor(
        backend=backend,
        torch_device="privateuseone:0",
        display_name="AMD Radeon(TM) Graphics",
        accelerated=True,
        reason="test accelerator",
        smoke_tested=True,
        torch_available=True,
        torch_version="2.4.1",
    )


def missing_loader(name: str) -> object:
    raise ModuleNotFoundError(f"No module named {name!r}", name=name)


class DeviceSelectionTests(unittest.TestCase):
    def test_import_and_cpu_fallback_work_without_torch(self) -> None:
        descriptor = select_compute_device(
            target_hardware(), module_loader=missing_loader
        )

        self.assertEqual("cpu", descriptor.backend)
        self.assertEqual("cpu", descriptor.torch_device)
        self.assertFalse(descriptor.torch_available)
        self.assertIn("CPU fallback", descriptor.reason)
        self.assertEqual(
            ["directml", "cuda/rocm", "mps", "cpu"],
            [attempt.backend for attempt in descriptor.attempts],
        )

    def test_directml_is_first_on_native_windows(self) -> None:
        fake_torch = FakeTorch(cuda_available=True)
        loaded: list[str] = []
        smoke_calls: list[tuple[str, str]] = []

        def loader(name: str) -> object:
            loaded.append(name)
            return {"torch": fake_torch, "torch_directml": FakeDirectML()}[name]

        def smoke(_torch: object, device: object, backend: str) -> None:
            smoke_calls.append((str(device), backend))

        descriptor = select_compute_device(
            target_hardware(), module_loader=loader, smoke_test=smoke
        )

        self.assertEqual("directml", descriptor.backend)
        self.assertEqual("privateuseone:0", descriptor.torch_device)
        self.assertEqual("Fake DirectML Radeon 0", descriptor.display_name)
        self.assertEqual(["torch", "torch_directml"], loaded)
        self.assertEqual([("privateuseone:0", "directml")], smoke_calls)
        self.assertTrue(descriptor.smoke_tested)

    def test_failed_directml_smoke_test_falls_back_to_cuda(self) -> None:
        fake_torch = FakeTorch(cuda_available=True)

        def loader(name: str) -> object:
            return {"torch": fake_torch, "torch_directml": FakeDirectML()}[name]

        def smoke(_torch: object, _device: object, backend: str) -> None:
            if backend == "directml":
                raise RuntimeError("simulated DirectML kernel failure")

        descriptor = select_compute_device(
            target_hardware(), module_loader=loader, smoke_test=smoke
        )

        self.assertEqual("cuda", descriptor.backend)
        self.assertEqual("cuda:0", descriptor.torch_device)
        self.assertEqual("failed", descriptor.attempts[0].status)
        self.assertIn("smoke test failed", descriptor.attempts[0].reason)
        self.assertEqual("selected", descriptor.attempts[1].status)
        self.assertIn("Earlier candidates were rejected", descriptor.reason)

    def test_rocm_runtime_is_reported_separately(self) -> None:
        fake_torch = FakeTorch(cuda_available=True, hip_version="6.2")

        descriptor = select_compute_device(
            target_hardware(system="Linux"),
            module_loader=lambda _name: fake_torch,
            smoke_test=lambda _torch, _device, _backend: None,
        )

        self.assertEqual("rocm", descriptor.backend)
        self.assertEqual("cuda:0", descriptor.torch_device)

    def test_mps_is_used_after_unavailable_cuda(self) -> None:
        fake_torch = FakeTorch(mps_available=True)

        descriptor = select_compute_device(
            target_hardware(system="Darwin"),
            module_loader=lambda _name: fake_torch,
            smoke_test=lambda _torch, _device, _backend: None,
        )

        self.assertEqual("mps", descriptor.backend)
        self.assertEqual("mps", descriptor.torch_device)
        self.assertEqual("unavailable", descriptor.attempts[0].status)

    def test_cpu_environment_override_skips_accelerator_modules(self) -> None:
        fake_torch = FakeTorch(cuda_available=True)
        loaded: list[str] = []

        def loader(name: str) -> object:
            loaded.append(name)
            if name != "torch":
                self.fail(f"unexpected accelerator import: {name}")
            return fake_torch

        descriptor = select_compute_device(
            target_hardware(),
            env={"JARVIS_COMPUTE_BACKEND": "cpu"},
            module_loader=loader,
        )

        self.assertEqual("cpu", descriptor.backend)
        self.assertEqual(["torch"], loaded)
        self.assertIn("explicitly", descriptor.reason)


class ResourceProfileTests(unittest.TestCase):
    def test_4600g_defaults_are_bounded_and_desktop_safe(self) -> None:
        corpus = CorpusStats(
            document_count=20,
            byte_count=4_000_000,
            character_count=1_000_000,
            estimated_tokens=250_000,
            source_kind="test",
        )

        profile = build_training_profile(
            hardware=target_hardware(),
            corpus_stats=corpus,
            device=accelerated_device(),
            env={},
        )

        self.assertEqual("small", profile.model_tier)
        self.assertEqual(256, profile.context_length)
        self.assertEqual(8, profile.batch_size)
        self.assertEqual(6, profile.cpu_threads)
        self.assertEqual(2, profile.interop_threads)
        self.assertEqual(0, profile.dataloader_workers)
        expected_parameters = (
            profile.vocabulary_size * profile.embedding_dimension
            + profile.transformer_layers
            * (
                4 * profile.embedding_dimension * profile.embedding_dimension
                + 2 * profile.embedding_dimension * profile.feed_forward_dimension
                + 4 * profile.embedding_dimension
            )
            + 2 * profile.embedding_dimension
        )
        self.assertEqual(expected_parameters, profile.estimated_parameter_count)
        self.assertLessEqual(
            profile.host_memory_budget_bytes, MAX_HOST_MEMORY_BUDGET_BYTES
        )
        self.assertLessEqual(
            profile.accelerator_memory_budget_bytes,
            MAX_ACCELERATOR_MEMORY_BUDGET_BYTES,
        )
        self.assertFalse(profile.pretrained_weights)
        self.assertFalse(profile.runtime_downloads_allowed)

    def test_profile_scales_from_corpus_instead_of_fixed_model_size(self) -> None:
        compact = build_training_profile(
            hardware=target_hardware(),
            corpus_stats=CorpusStats(1, 4_000, 4_000, 1_000, "test"),
            device=accelerated_device(),
            env={},
        )
        medium = build_training_profile(
            hardware=target_hardware(),
            corpus_stats=CorpusStats(200, 8_000_000, 8_000_000, 2_000_000, "test"),
            device=accelerated_device(),
            env={},
        )

        self.assertEqual("compact", compact.model_tier)
        self.assertEqual("medium", medium.model_tier)
        self.assertGreater(medium.context_length, compact.context_length)
        self.assertGreater(
            medium.estimated_parameter_count, compact.estimated_parameter_count
        )
        self.assertNotEqual(medium.max_steps, compact.max_steps)

    def test_environment_overrides_are_clamped_to_safe_caps(self) -> None:
        environment = {
            "JARVIS_MAX_RAM_GB": "999",
            "JARVIS_MAX_GPU_MEMORY_GB": "999",
            "JARVIS_CPU_THREADS": "99",
            "JARVIS_INTEROP_THREADS": "0",
            "JARVIS_DATALOADER_WORKERS": "99",
            "JARVIS_BATCH_SIZE": "999",
            "JARVIS_CONTEXT_LENGTH": "999",
            "JARVIS_GRAD_ACCUMULATION_STEPS": "999",
            "JARVIS_MAX_STEPS": "999999",
        }

        profile = build_training_profile(
            hardware=target_hardware(),
            corpus_stats=CorpusStats(5, 1_000_000, 1_000_000, 250_000, "test"),
            device=accelerated_device(),
            env=environment,
        )

        self.assertLessEqual(
            profile.host_memory_budget_bytes, MAX_HOST_MEMORY_BUDGET_BYTES
        )
        self.assertLessEqual(
            profile.accelerator_memory_budget_bytes,
            MAX_ACCELERATOR_MEMORY_BUDGET_BYTES,
        )
        self.assertEqual(MAX_CONTEXT_LENGTH, profile.context_length)
        self.assertLessEqual(profile.batch_size, 8)
        self.assertEqual(16, profile.gradient_accumulation_steps)
        self.assertEqual(MAX_TRAINING_STEPS, profile.max_steps)
        self.assertEqual(12, profile.cpu_threads)
        self.assertEqual(1, profile.interop_threads)
        self.assertEqual(4, profile.dataloader_workers)
        self.assertGreaterEqual(len(profile.warnings), 8)

    def test_invalid_environment_values_are_ignored_with_warnings(self) -> None:
        profile = build_training_profile(
            hardware=target_hardware(),
            corpus_stats=CorpusStats(1, 10_000, 10_000, 2_500, "test"),
            device=accelerated_device(),
            env={
                "JARVIS_MAX_RAM_GB": "not-a-number",
                "JARVIS_CPU_THREADS": "many",
                "JARVIS_CONTEXT_LENGTH": "wide",
            },
        )

        self.assertEqual(6, profile.cpu_threads)
        self.assertEqual(128, profile.context_length)
        self.assertTrue(any("invalid" in warning.casefold() for warning in profile.warnings))

    def test_corpus_probe_returns_aggregates_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "private-name.txt"
            pdf_path = root / "another-private-name.pdf"
            text_path.write_text("alpha beta gamma " * 100, encoding="utf-8")
            pdf_path.write_bytes(b"%PDF-test" * 100)

            stats = detect_corpus_stats(root)

        self.assertEqual(2, stats.document_count)
        self.assertGreater(stats.byte_count, 0)
        self.assertGreater(stats.estimated_tokens, 0)
        serialized = str(stats.to_dict())
        self.assertNotIn("private-name", serialized)
        self.assertNotIn(directory, serialized)

    def test_raw_pdf_size_does_not_invent_a_token_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "compressed-or-image-heavy.pdf"
            pdf_path.write_bytes(b"%PDF-1.7" + b"x" * 100_000)
            stats = detect_corpus_stats(pdf_path)

        self.assertEqual(1, stats.document_count)
        self.assertGreater(stats.byte_count, 0)
        self.assertEqual(0, stats.character_count)
        self.assertEqual(0, stats.estimated_tokens)


class CPUThreadTests(unittest.TestCase):
    def test_cpu_thread_tuning_uses_physical_cores_and_caps_overrides(self) -> None:
        fake_torch = FakeTorch()

        settings = configure_cpu_threads(
            target_hardware(),
            env={"JARVIS_CPU_THREADS": "99", "JARVIS_INTEROP_THREADS": "99"},
            torch_module=fake_torch,
            apply_environment=False,
        )

        self.assertEqual(12, settings.intra_op_threads)
        self.assertEqual(4, settings.inter_op_threads)
        self.assertEqual([12], fake_torch.intra_threads)
        self.assertEqual([4], fake_torch.interop_threads)
        self.assertTrue(settings.torch_configured)
        self.assertFalse(settings.environment_configured)

    def test_cpu_thread_tuning_sets_process_variables(self) -> None:
        fake_torch = FakeTorch()
        with patch.dict(os.environ, {}, clear=False):
            settings = configure_cpu_threads(
                target_hardware(), env={}, torch_module=fake_torch
            )
            self.assertEqual("6", os.environ["OMP_NUM_THREADS"])
            self.assertEqual("6", os.environ["MKL_NUM_THREADS"])
        self.assertTrue(settings.environment_configured)


class HardwareDetectionTests(unittest.TestCase):
    def test_dxdiag_adds_4600g_and_uma_display_facts(self) -> None:
        report = """
          Processor: AMD Ryzen 5 4600G with Radeon Graphics (12 CPUs), ~4.0GHz
             Memory: 40960MB RAM
Available OS Memory: 32554MB RAM
    DirectX Version: DirectX 12
          Card name: AMD Radeon(TM) Graphics
    Dedicated Memory: 8175 MB
       Shared Memory: 16277 MB
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dxdiag.txt"
            path.write_text(report, encoding="utf-8")
            hardware = detect_hardware(path)

        self.assertEqual("AMD Radeon(TM) Graphics", hardware.gpu_name)
        self.assertEqual(8_175 * MIB, hardware.gpu_dedicated_memory_bytes)
        self.assertEqual(16_277 * MIB, hardware.gpu_shared_memory_bytes)
        self.assertEqual(40_960 * MIB, hardware.installed_ram_bytes)
        self.assertEqual("DirectX 12", hardware.directx_version)
        self.assertIn("dxdiag", hardware.sources)


if __name__ == "__main__":
    unittest.main()
