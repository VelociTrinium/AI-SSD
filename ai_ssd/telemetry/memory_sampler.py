"""High-resolution process memory sampling for exact RSS telemetry."""

from __future__ import annotations
import os
import sys
import time
import threading
from typing import Dict, List, Optional
import numpy as np
import psutil


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in MB via psutil."""
    try:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def get_proc_status_memory() -> Dict[str, float]:
    """Reads Linux /proc/self/status metrics in MB. Returns zeros on Windows."""
    res = {"vm_rss_mb": 0.0, "rss_anon_mb": 0.0, "rss_file_mb": 0.0, "vm_peak_mb": 0.0}
    if sys.platform != "win32" and os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        res["vm_rss_mb"] = float(line.split()[1]) / 1024.0
                    elif line.startswith("RssAnon:"):
                        res["rss_anon_mb"] = float(line.split()[1]) / 1024.0
                    elif line.startswith("RssFile:"):
                        res["rss_file_mb"] = float(line.split()[1]) / 1024.0
                    elif line.startswith("VmPeak:"):
                        res["vm_peak_mb"] = float(line.split()[1]) / 1024.0
        except Exception:
            pass
    else:
        current_rss = get_current_rss_mb()
        res["vm_rss_mb"] = current_rss
        res["vm_peak_mb"] = current_rss
    return res


class ProcessMemorySampler:
    """Threaded high-resolution process RSS sampler.
    
    Continuously samples OS process Resident Set Size (RSS) during execution
    intervals to accurately calculate min, avg, peak, and std memory.
    """

    def __init__(self, sample_interval_s: float = 0.002):
        self.sample_interval_s = sample_interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()

    def start(self) -> None:
        """Starts background memory sampling thread."""
        self.samples = [self._get_rss()]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """Stops sampling and returns summary statistics in MB."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.samples.append(self._get_rss())

        if not self.samples:
            return {
                "min_rss_mb": 0.0,
                "avg_rss_mb": 0.0,
                "peak_rss_mb": 0.0,
                "std_rss_mb": 0.0,
                "sample_count": 0,
            }

        return {
            "min_rss_mb": float(np.min(self.samples)),
            "avg_rss_mb": float(np.mean(self.samples)),
            "peak_rss_mb": float(np.max(self.samples)),
            "std_rss_mb": float(np.std(self.samples)),
            "sample_count": len(self.samples),
        }

    def _get_rss(self) -> float:
        try:
            return self._proc.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._get_rss())
            except Exception:
                pass
            time.sleep(self.sample_interval_s)
