"""Setup ownership for transient clipping scratch in one serial constructor family.

Only kernel-local cell geometry is shared. Hierarchies, certificates, outputs,
retained fragments and strategy state remain separately owned by each builder.
The largest constructor must reserve storage first; storage never grows after
its first allocation because captured graphs retain its original addresses.
"""

import warp as wp

from ._cell import allocate_scratch


class ClippingScratch:
    def __init__(self):
        self.scratch = None

    def allocate(self, cells, capacity, device):
        slots = ((cells + 31) // 32) * 32 * capacity
        if self.scratch is None:
            self.scratch = allocate_scratch(cells, capacity, device, robust=True)
        elif (
            self.scratch.capacity != capacity
            or len(self.scratch.planes) < slots
            or self.scratch.planes.device != wp.get_device(device)
        ):
            raise ValueError(
                "Shared clipping storage must be reserved by the largest compatible builder"
            )
        return self.scratch
