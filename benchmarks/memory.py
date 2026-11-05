"""Sample GPU memory in a separate process, including GIL-holding CUDA calls."""

import ctypes
import multiprocessing as mp


class MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def check(code):
    if code:
        raise RuntimeError(f"NVML returned error {code}")


def sample_loop(index, interval, stop, connection):
    record = {"before_construction_bytes": 0, "peak_bytes": 0, "samples": 0, "error": None}
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        check(lib.nvmlInit_v2())
        handle = ctypes.c_void_p()
        check(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)))
        info = MemoryInfo()
        check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)))
        record["before_construction_bytes"] = info.used
        record["peak_bytes"] = info.used
    except (OSError, RuntimeError) as error:
        record["error"] = str(error)
        connection.send(record)
        return
    connection.send({"ready": True})
    try:
        while True:
            check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)))
            record["peak_bytes"] = max(record["peak_bytes"], info.used)
            record["samples"] += 1
            if stop.wait(interval):
                break
    except RuntimeError as error:
        record["error"] = str(error)
    finally:
        lib.nvmlShutdown()
        connection.send(record)
        connection.close()


class MemorySampler:
    def __init__(self, index=0, interval=0.001):
        self.index, self.interval = index, interval
        self.context = mp.get_context("spawn")
        self.stop = self.context.Event()
        self.connection, child = self.context.Pipe(duplex=False)
        self.process = self.context.Process(
            target=sample_loop, args=(index, interval, self.stop, child)
        )
        self.initial_error = None

    def start(self):
        self.process.start()
        initial = self.connection.recv()
        if not initial.get("ready"):
            self.initial_error = initial

    def finish(self):
        self.stop.set()
        record = self.initial_error or self.connection.recv()
        self.process.join()
        self.connection.close()
        return {
            "method": "device-wide NVML sampling in a separate process; includes inputs, contexts, and allocator reserves; approximate peak",
            "interval_seconds": self.interval,
            **record,
        }
