"""Synthetic RAM planning tests; no accelerator or training runtime is loaded."""

from __future__ import annotations

import unittest
from dataclasses import replace

from jarvis_localhost.hardware import CorpusStats, DeviceDescriptor, HardwareInfo
from jarvis_localhost.hardware.device import GIB, MIB
from jarvis_localhost.hardware.profiles import build_training_profile


def desktop_hardware(*, total: float = 31.79, available: float = 21) -> HardwareInfo:
    return HardwareInfo(
        system="Windows",
        release="11",
        machine="AMD64",
        cpu_name="Synthetic Ryzen 4600G UMA",
        physical_cpu_cores=6,
        logical_cpu_cores=12,
        total_ram_bytes=int(total * GIB),
        available_ram_bytes=int(available * GIB),
        installed_ram_bytes=40 * GIB,
        gpu_name="Synthetic Radeon",
        gpu_dedicated_memory_bytes=8175 * MIB,
        gpu_shared_memory_bytes=16277 * MIB,
        sources=("test",),
    )


def profile_for(hardware: HardwareInfo, env: dict[str, str]):
    return build_training_profile(
        hardware=hardware,
        corpus_stats=CorpusStats(269, 0, 0, 50_000_000, "test"),
        device=DeviceDescriptor(
            backend="directml",
            torch_device="synthetic-device-not-created",
            display_name="Synthetic Radeon",
            accelerated=True,
        ),
        env=env,
    )


class ResourceBudgetOverrideTests(unittest.TestCase):
    def test_explicit_24_gib_preserves_model_and_training_plan(self) -> None:
        hardware = desktop_hardware()
        environment = {"JARVIS_MAX_GPU_MEMORY_GB": "6", "JARVIS_MAX_STEPS": "1890"}
        default = profile_for(hardware, environment)
        requested = profile_for(hardware, {**environment, "JARVIS_MAX_RAM_GB": "24"})

        self.assertEqual(24 * GIB, requested.host_memory_budget_bytes)
        self.assertEqual("medium", requested.model_tier)
        self.assertEqual(
            default,
            replace(
                requested,
                host_memory_budget_bytes=default.host_memory_budget_bytes,
                reason=default.reason,
                warnings=default.warnings,
            ),
        )
        self.assertTrue(any("currently available RAM" in w for w in requested.warnings))
        self.assertTrue(any("not independent allocations" in w for w in requested.warnings))

    def test_defaults_keep_conservative_total_available_and_absolute_caps(self) -> None:
        for total, available, expected in (
            (32, 24, int(32 * GIB * 0.40)),
            (32, 10, 7 * GIB),
            (128, 120, 16 * GIB),
        ):
            with self.subTest(total=total, available=available):
                profile = profile_for(desktop_hardware(total=total, available=available), {})
                self.assertEqual(expected, profile.host_memory_budget_bytes)

    def test_explicit_budget_reserves_four_gib_on_smaller_desktop(self) -> None:
        hardware = desktop_hardware(total=12, available=10)
        profile = profile_for(hardware, {"JARVIS_MAX_RAM_GB": "24"})
        self.assertEqual(8 * GIB, profile.host_memory_budget_bytes)
        self.assertTrue(any("Clamped JARVIS_MAX_RAM_GB" in w for w in profile.warnings))

    def test_explicit_budget_reserves_twenty_percent_on_larger_desktop(self) -> None:
        hardware = desktop_hardware(total=24, available=24)
        profile = profile_for(hardware, {"JARVIS_MAX_RAM_GB": "24"})
        self.assertAlmostEqual(19.2, profile.host_memory_budget_bytes / GIB, places=8)
        self.assertGreaterEqual(hardware.total_ram_bytes - profile.host_memory_budget_bytes,
                                hardware.total_ram_bytes * 0.20)

    def test_available_memory_warns_without_changing_explicit_ceiling(self) -> None:
        for available in (4, 21, 29):
            with self.subTest(available=available):
                profile = profile_for(desktop_hardware(available=available),
                                      {"JARVIS_MAX_RAM_GB": "24"})
                self.assertEqual(24 * GIB, profile.host_memory_budget_bytes)
                pressure = [w for w in profile.warnings if "currently available RAM" in w]
                self.assertEqual(available < 24, bool(pressure))
                if pressure:
                    self.assertIn("not a reservation or enforced process RSS limit", pressure[0])

    def test_explicit_lower_budget_is_respected(self) -> None:
        profile = profile_for(desktop_hardware(), {"JARVIS_MAX_RAM_GB": "2.5"})
        self.assertEqual(int(2.5 * GIB), profile.host_memory_budget_bytes)
        self.assertFalse(any("Clamped JARVIS_MAX_RAM_GB" in w for w in profile.warnings))

    def test_extreme_finite_request_clamps_without_overflow(self) -> None:
        hardware = desktop_hardware(total=128, available=120)
        for value in ("999", "1e300"):
            with self.subTest(value=value):
                profile = profile_for(hardware, {"JARVIS_MAX_RAM_GB": value})
                self.assertEqual(24 * GIB, profile.host_memory_budget_bytes)
                self.assertTrue(any("Clamped JARVIS_MAX_RAM_GB" in w for w in profile.warnings))

    def test_invalid_request_keeps_default_with_warning(self) -> None:
        hardware = desktop_hardware()
        default = profile_for(hardware, {})
        for value in ("not-a-number", "nan", "inf", "-1", "0"):
            with self.subTest(value=value):
                profile = profile_for(hardware, {"JARVIS_MAX_RAM_GB": value})
                self.assertEqual(default.host_memory_budget_bytes, profile.host_memory_budget_bytes)
                self.assertTrue(any("Ignored" in w and "JARVIS_MAX_RAM_GB" in w
                                    for w in profile.warnings))

    def test_unknown_capacity_retains_safe_fallback(self) -> None:
        hardware = replace(desktop_hardware(), total_ram_bytes=0, available_ram_bytes=0)
        profile = profile_for(hardware, {"JARVIS_MAX_RAM_GB": "24"})
        self.assertEqual(4 * GIB, profile.host_memory_budget_bytes)

    def test_shared_accelerator_warning_requires_selected_accelerator(self) -> None:
        profile = build_training_profile(
            hardware=desktop_hardware(),
            corpus_stats=CorpusStats(1, 0, 0, 1000, "test"),
            device=DeviceDescriptor(backend="cpu", torch_device="cpu"),
            env={"JARVIS_MAX_RAM_GB": "24"},
        )
        self.assertEqual(0, profile.accelerator_memory_budget_bytes)
        self.assertFalse(any("not independent allocations" in w for w in profile.warnings))


if __name__ == "__main__":
    unittest.main()
