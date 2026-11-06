"""Optional device iteration/mode controls for scheduler-owned VTM cores."""

import weakref

import warp as wp

wp.set_module_options({"enable_backward": False})


def scalar(value, device, name):
    if (
        not isinstance(value, wp.array)
        or value.dtype != wp.int32
        or value.shape != (1,)
        or value.device != device
        or not value.is_contiguous
    ):
        raise ValueError(name + " must be one contiguous device int32")
    return value


@wp.kernel
def gate(
    budget: wp.array[int],
    mode: wp.array[int],
    use_budget: bool,
    use_mode: bool,
    maximum: int,
    iterations: wp.array[int],
    error: wp.array[int],
    running: wp.array[int],
):
    invalid = False
    if use_budget:
        invalid = budget[0] < 0 or budget[0] > maximum
    if use_mode:
        invalid = invalid or mode[0] < 0 or mode[0] > 1
    if invalid:
        error[0] |= 32
        running[0] = 0
    elif use_budget and iterations[0] >= budget[0]:
        running[0] = 0


class DeviceControls:
    def __init__(self, core, budget, mode):
        self.core = weakref.proxy(core)
        self.use_budget, self.use_mode = budget is not None, mode is not None
        self.budget = (
            core.iterations if budget is None else scalar(budget, core.device, "iteration_limit")
        )
        self.mode = (
            core.iterations if mode is None else scalar(mode, core.device, "selective_enabled")
        )
        wp.load_module(module=__name__, device=core.device)

    def gate(self):
        c = self.core
        wp.launch(
            gate,
            1,
            inputs=[
                self.budget,
                self.mode,
                self.use_budget,
                self.use_mode,
                c.max_iterations,
                c.iterations,
                c.error,
                c._running,
            ],
            device=c.device,
        )
