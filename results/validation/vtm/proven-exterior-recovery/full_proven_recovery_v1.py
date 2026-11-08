"""Private fresh-solve fallback after the original fitting/band policy fails."""
import warp as wp

from warpower import _extraction, _mesher
from proven_exterior_transaction_v1 import ProofRemoval
from representable_boundary_v1 import install as install_midpoint

wp.set_module_options({'enable_backward': False, 'fast_math': False})


@wp.kernel
def last_chance(error: wp.array[int], reason: wp.array[int], enabled: wp.array[int]):
    enabled[0] = int(error[0] == 0 and reason[0] != 0)


def install():
    context = {'enabled': None}
    init_extraction = _extraction.ExtractionWorkspace.__init__

    def extraction_init(self, *args, **kwargs):
        init_extraction(self, *args, **kwargs)
        if self._exterior is not None and context['enabled'] is not None:
            self._proof = ProofRemoval(self)
            self._proof.enabled = context['enabled']

    _extraction.ExtractionWorkspace.__init__ = extraction_init
    init_mesh = _mesher.VariationalMeshingWorkspace.__init__
    instances = []

    def mesher_init(self, domain, *args, **kwargs):
        enabled = wp.zeros(1, dtype=int, device=domain.device)
        context['enabled'] = enabled
        try:
            init_mesh(self, domain, *args, **kwargs)
        finally:
            context['enabled'] = None
        assert self.exterior_recovery and self.core.local_backtracking is None
        self._proof_enabled = enabled
        c = self.core
        names = ['_launch', '_accept', '_propose', '_propose_args']
        ordinary = {name: getattr(c, name) for name in names}
        install_midpoint(c)
        corrected = {name: getattr(c, name) for name in names}
        for name, value in ordinary.items():
            setattr(c, name, value)
        launch, extract = self._launch, self._extract

        def extract_with_proof_flag():
            extract()
            e = self.extraction
            launch(_mesher.remember_material_recovery, 1, [e.valid, e._proof.removed, self._material_recovery])

        self._extract = extract_with_proof_flag

        def final_recovery():
            # One additional, explicitly designated recovery step. Ordinary
            # growth/polish/quality kernels and their budgets stay unchanged.
            for name, value in corrected.items():
                setattr(c, name, value)
            try:
                self._global(1)
            finally:
                for name, value in ordinary.items():
                    setattr(c, name, value)
            wp.capture_if(self.ready, self._extract)

        def before_require(kernel, dim, args):
            if kernel is _mesher.require_extraction:
                launch(last_chance, 1, [self.error, self._state.extraction_reason, enabled])
                wp.capture_if(enabled, final_recovery)
            launch(kernel, dim, args)

        self._launch = before_require
        instances.append(self)

    _mesher.VariationalMeshingWorkspace.__init__ = mesher_init
    run = _mesher.VariationalMeshingWorkspace.run

    def recorded_run(self, *args, **kwargs):
        assert wp.get_stream(self.device).is_capturing
        self._proof_enabled.zero_()
        self.core._tie_stats.zero_()
        return run(self, *args, **kwargs)

    _mesher.VariationalMeshingWorkspace.run = recorded_run
    wp.load_module(module=__name__, device='cuda:0')
    return instances
