"""Measurements; distinguish actual RSS sampling from Python allocations."""
import os
import threading
import time
import tracemalloc


class Measurements:
    def __init__(self):
        self._halt = threading.Event()
        self._thread = None
        self.rss_sampled_max = None
        self.has_psutil = False
        self.process_start = 0.0
        self.wall_start = 0.0

    def __enter__(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.has_psutil = True

            def sample():
                while not self._halt.is_set():
                    try:
                        usage = process.memory_info().rss
                        self.rss_sampled_max = max(
                            self.rss_sampled_max or 0, usage
                        )
                    except psutil.Error:
                        pass
                    self._halt.wait(0.01)
            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        except ImportError:
            self.has_psutil = False
        tracemalloc.start()
        self.wall_start = time.perf_counter()
        self.process_start = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall_seconds = time.perf_counter() - self.wall_start
        self.cpu_seconds = time.process_time() - self.process_start
        self.python_current, self.python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def rss_sampled_mib(self):
        return (self.rss_sampled_max / 2**20
                if self.rss_sampled_max is not None else None)

    @property
    def python_peak_mib(self):
        return self.python_peak / 2**20
