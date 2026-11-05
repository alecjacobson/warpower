"""Sample total GPU memory usage through NVML without launching GPU work."""

import ctypes
import threading


class MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class MemorySampler:
    def __init__(self, index=0, interval=0.001):
        self.interval = interval
        self.samples = 0
        self.error = None
        self.peak = 0
        self.initial = 0
        self.stop_event = threading.Event()
        self.thread = None
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            self._check(self.lib.nvmlInit_v2())
            self.handle = ctypes.c_void_p()
            self._check(
                self.lib.nvmlDeviceGetHandleByIndex_v2(
                    ctypes.c_uint(index), ctypes.byref(self.handle)
                )
            )
            self.initial = self.read()
            self.peak = self.initial
        except (OSError, RuntimeError) as error:
            self.error = str(error)

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"NVML returned error {code}")

    def read(self):
        info = MemoryInfo()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)))
        return info.used

    def sample(self):
        try:
            self.peak = max(self.peak, self.read())
            self.samples += 1
        except RuntimeError as error:
            self.error = str(error)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        if self.error is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.sample()
            self.lib.nvmlShutdown()
        return {
            "method": "device-wide NVML sampling; includes inputs, CUDA contexts, and allocator reserves; approximate peak",
            "interval_seconds": self.interval,
            "samples": self.samples,
            "before_construction_bytes": self.initial,
            "peak_bytes": self.peak,
            "error": self.error,
        }
