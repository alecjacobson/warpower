"""Private experiment: midpoint proposals only inside existing recovery fitting."""
from representable_boundary_v1 import install as install_midpoint


def install(mesh):
    core = mesh.core
    assert core.local_backtracking is None
    names = ['_launch', '_accept', '_propose', '_propose_args']
    original = {name: getattr(core, name) for name in names}
    install_midpoint(core)
    corrected = {name: getattr(core, name) for name in names}
    for name, value in original.items():
        setattr(core, name, value)
    recovery = mesh._recovery

    def recovery_only():
        # This Python switch selects nodes while the full graph is recorded.
        # Its execution predicate remains the original device recovery flag.
        for name, value in corrected.items():
            setattr(core, name, value)
        try:
            recovery()
        finally:
            for name, value in original.items():
                setattr(core, name, value)

    mesh._recovery = recovery_only
    return mesh
