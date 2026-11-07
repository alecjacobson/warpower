"""Private #190 experiment: reuse only the preceding accepted current evaluation."""
import warp as wp
from warpower import _selective, _vtm

wp.set_module_options({'enable_backward': False})
Base = _selective.SelectiveUpdate


@wp.kernel
def accepted_evaluation(error: wp.array[int], accepted: wp.array[int], needed: wp.array[int]):
    needed[0] = int(error[0] != 0 or accepted[0] == 0)


class CurrentSelection(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_current = False  # Recording context only; never a device decision.
        with wp.ScopedDevice(self.device):
            self._current_needed = wp.ones(1, dtype=int)
            wp.load_module(module=__name__, device=self.device)

    def invalidate(self):
        self._current_needed.fill_(1)

    def reset(self):
        self.invalidate()
        super().reset()

    def _select_all(self):
        self.invalidate()
        super()._select_all()

    def evaluate(self):
        if self._allow_current:
            wp.capture_if(self._current_needed, lambda: Base.evaluate(self))
        else:
            # Explicit/direct evaluations and every candidate check stay fresh.
            self.invalidate()
            Base.evaluate(self)

    def _check(self):
        super()._check()
        wp.launch(accepted_evaluation, 1,
                  inputs=[self.error, self.core._accepted, self._current_needed],
                  device=self.device)

    def select_current(self):
        # Only the optimizer's uninterrupted step schedule calls this entry.
        # Base.select records its branches immediately; check() is recorded
        # afterward, outside this Python context, and always evaluates freshly.
        self._allow_current = True
        try:
            super().select()
        finally:
            self._allow_current = False


def install():
    assert _selective.SelectiveUpdate is Base
    previous_step = _vtm.VTMWorkspace._step
    previous_try = _vtm.VTMWorkspace._try_step
    previous_restore = _vtm.VTMWorkspace._restore

    def step(self):
        if not isinstance(self.selection, CurrentSelection):
            return previous_step(self)
        self.selection.select_current()
        wp.capture_if(self.selection.ready, self._step_valid)
        if self.controls is not None:
            self.controls.gate()

    def try_step(self):
        if isinstance(self.selection, CurrentSelection):
            self.selection.invalidate()
        previous_try(self)

    def restore(self):
        if isinstance(self.selection, CurrentSelection):
            self.selection.invalidate()
        previous_restore(self)

    _selective.SelectiveUpdate = CurrentSelection
    _vtm.VTMWorkspace._step = step
    _vtm.VTMWorkspace._try_step = try_step
    _vtm.VTMWorkspace._restore = restore
