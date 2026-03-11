"""
system_monitor.py — J.A.R.V.I.S System Intelligence Module
Collects real-time hardware and OS metrics using psutil.
"""

import psutil
import platform
import time
import socket
import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class CPUInfo:
    percent: float
    freq_mhz: float
    cores_physical: int
    cores_logical: int
    per_core: List[float]
    temperature: float | None


@dataclass
class MemoryInfo:
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float


@dataclass
class DiskInfo:
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float
    read_mb: float
    write_mb: float


@dataclass
class NetworkInfo:
    bytes_sent_mb: float
    bytes_recv_mb: float
    packets_sent: int
    packets_recv: int
    hostname: str
    ip_address: str


@dataclass
class SystemSnapshot:
    timestamp: str
    platform: str
    uptime_hours: float
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo
    network: NetworkInfo
    top_processes: List[Dict]
    boot_time: str


class SystemMonitor:
    """Real-time system metrics collector for J.A.R.V.I.S"""

    def __init__(self):
        self._net_start = psutil.net_io_counters()
        self._disk_start = psutil.disk_io_counters()
        self._start_time = time.time()

    # ─── CPU ──────────────────────────────────────────────────────────────────
    def get_cpu(self) -> CPUInfo:
        freq = psutil.cpu_freq()
        temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        temp = round(temps[key][0].current, 1)
                        break
        except Exception:
            pass

        return CPUInfo(
            percent=psutil.cpu_percent(interval=0.1),
            freq_mhz=round(freq.current, 1) if freq else 0.0,
            cores_physical=psutil.cpu_count(logical=False) or 1,
            cores_logical=psutil.cpu_count(logical=True) or 1,
            per_core=psutil.cpu_percent(interval=0.1, percpu=True),
            temperature=temp,
        )

    # ─── Memory ───────────────────────────────────────────────────────────────
    def get_memory(self) -> MemoryInfo:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        gb = 1024 ** 3
        return MemoryInfo(
            total_gb=round(vm.total / gb, 2),
            used_gb=round(vm.used / gb, 2),
            available_gb=round(vm.available / gb, 2),
            percent=vm.percent,
            swap_total_gb=round(sw.total / gb, 2),
            swap_used_gb=round(sw.used / gb, 2),
            swap_percent=sw.percent,
        )

    # ─── Disk ─────────────────────────────────────────────────────────────────
    def get_disk(self) -> DiskInfo:
        disk = psutil.disk_usage("/")
        gb = 1024 ** 3
        mb = 1024 ** 2
        try:
            io = psutil.disk_io_counters()
            read_mb = round(io.read_bytes / mb, 2) if io else 0.0
            write_mb = round(io.write_bytes / mb, 2) if io else 0.0
        except Exception:
            read_mb = write_mb = 0.0

        return DiskInfo(
            total_gb=round(disk.total / gb, 2),
            used_gb=round(disk.used / gb, 2),
            free_gb=round(disk.free / gb, 2),
            percent=disk.percent,
            read_mb=read_mb,
            write_mb=write_mb,
        )

    # ─── Network ──────────────────────────────────────────────────────────────
    def get_network(self) -> NetworkInfo:
        net = psutil.net_io_counters()
        mb = 1024 ** 2
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except Exception:
            hostname, ip = "UNKNOWN", "0.0.0.0"

        return NetworkInfo(
            bytes_sent_mb=round(net.bytes_sent / mb, 2),
            bytes_recv_mb=round(net.bytes_recv / mb, 2),
            packets_sent=net.packets_sent,
            packets_recv=net.packets_recv,
            hostname=hostname,
            ip_address=ip,
        )

    # ─── Processes ────────────────────────────────────────────────────────────
    def get_top_processes(self, n: int = 5) -> List[Dict]:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                if info["cpu_percent"] is not None:
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"][:20],
                        "cpu": round(info["cpu_percent"], 1),
                        "mem": round(info["memory_percent"], 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(procs, key=lambda x: x["cpu"], reverse=True)[:n]

    # ─── Full Snapshot ────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        boot = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime = (time.time() - psutil.boot_time()) / 3600

        snap = SystemSnapshot(
            timestamp=datetime.datetime.now().isoformat(),
            platform=f"{platform.system()} {platform.release()}",
            uptime_hours=round(uptime, 2),
            cpu=self.get_cpu(),
            memory=self.get_memory(),
            disk=self.get_disk(),
            network=self.get_network(),
            top_processes=self.get_top_processes(),
            boot_time=boot.strftime("%d/%m/%Y %H:%M"),
        )
        return asdict(snap)

    # ─── Alerts ───────────────────────────────────────────────────────────────
    def check_alerts(self) -> List[str]:
        alerts = []
        snap = self.snapshot()
        if snap["cpu"]["percent"] > 90:
            alerts.append(f"⚠ CPU crítica: {snap['cpu']['percent']}%")
        if snap["memory"]["percent"] > 85:
            alerts.append(f"⚠ Memória crítica: {snap['memory']['percent']}%")
        if snap["disk"]["percent"] > 90:
            alerts.append(f"⚠ Disco crítico: {snap['disk']['percent']}%")
        if snap["cpu"]["temperature"] and snap["cpu"]["temperature"] > 85:
            alerts.append(f"⚠ Temperatura CPU: {snap['cpu']['temperature']}°C")
        return alerts
