"""Internal VTM reconstruction from recertified previous cells, with full fallback."""

import warp as wp

from . import _validation
from ._delaunay_recovery import DelaunayWorkspace, reset
from ._facet_seeds import CertifiedSeeds
from ._proved_completion import SeededFacetCompletion

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def begin(
    count: wp.array[int],
    ready: wp.array[int],
    attempted: wp.array[int],
    accepted: wp.array[int],
    error: wp.array[int],
):
    ready[0] = int(count[0] > 0)
    attempted[0] = ready[0]
    accepted[0] = 0
    error[0] = 0


@wp.kernel
def input_ready(validation: wp.array[int], ready: wp.array[int], error: wp.array[int]):
    ready[0] = int(validation[0] == 0)
    error[0] = int(validation[0] != 0)


@wp.kernel
def completed(
    seed_error: wp.array[int],
    completion_error: wp.array[int],
    count: wp.array[int],
    capacity: int,
    error: wp.array[int],
    accepted: wp.array[int],
):
    error[0] = (seed_error[0] << 8) | completion_error[0]
    if count[0] < 1 or count[0] > capacity:
        error[0] |= 16
    accepted[0] = int(error[0] == 0)


@wp.kernel
def fallback(error: wp.array[int], ready: wp.array[int]):
    ready[0] = int(error[0] != 0)


@wp.kernel
def remember(error: wp.array[int], count: wp.array[int], saved: wp.array[int]):
    saved[0] = 0
    if error[0] == 0:
        saved[0] = count[0]


class ReuseDelaunayWorkspace(DelaunayWorkspace):
    """Bounded extra strategy for small/medium VTM trajectories.

    Rebuild current canonical owners, discard previous cells that are no longer
    strict symbolic Delaunay facets, and complete the remaining fragment. Every
    accepted result passes the full original-coordinate certificate. An invalid
    input or an exhausted fragment uses the ordinary constructor and its error
    contract. The standalone DelaunayWorkspace defaults remain unchanged.
    """

    _facet_query_block_dim = 32

    def __init__(self, *args, reuse_max_rounds=4, **kwargs):
        super().__init__(*args, **kwargs)
        with wp.ScopedDevice(self.device):
            self._previous = wp.empty_like(self.tetrahedra)
            self._previous_count = wp.zeros(1, dtype=int)
            self._reuse_ready = wp.zeros(1, dtype=int)
            self.reuse_attempted = wp.zeros(1, dtype=int)
            self.reuse_accepted = wp.zeros(1, dtype=int)
            self.reuse_error = wp.zeros(1, dtype=int)
            self._reuse_filter = CertifiedSeeds(
                self.points,
                self._previous,
                self._previous_count,
                self._original.tree,
                block_dim=self._facet_query_block_dim,
                prepare_geometry=True,
            )
            self._reuse_completion = SeededFacetCompletion(
                self._reuse_filter,
                max_rounds=reuse_max_rounds,
                query_block_dim=self._facet_query_block_dim,
            )
            # The filter just certified these cells against the current input.
            # Keep that proof through this completion even beyond the ordinary
            # small-cloud query threshold; only newly introduced cells need
            # another sphere check. No proof survives a call to run().
            self._reuse_completion._certificate.point_parallel = True
            self.reuse_seeds = self._reuse_filter.tetrahedra_count
            self.reuse_rounds = self._reuse_completion.rounds
            wp.load_module(module=__name__, device=self.device)
            with wp.ScopedCapture() as capture:
                self._record_reuse()
            self._reuse_graph = capture.graph
        wp.synchronize_device(self.device)

    def clear_reuse(self):
        """Device operation: start a fresh trajectory without previous fragments."""
        self._previous_count.zero_()

    def _record_reuse(self):
        self._launch(reset, 1, [self._state, self.cache_strategy])
        self.reuse_seeds.zero_()
        self.reuse_rounds.zero_()
        self._launch(
            begin,
            1,
            [
                self._previous_count,
                self._reuse_ready,
                self.reuse_attempted,
                self.reuse_accepted,
                self.reuse_error,
            ],
        )
        wp.capture_if(self._reuse_ready, self._validate_reuse)
        self._launch(fallback, 1, [self.error, self._reuse_ready])
        wp.capture_if(self._reuse_ready, super().run)
        wp.copy(self._previous, self.tetrahedra)
        self._launch(remember, 1, [self.error, self.tetrahedra_count, self._previous_count])

    def _validate_reuse(self):
        # Do not send NaN/Inf producers into the hierarchy or exact predicates.
        self._base._validation.zero_()
        self._launch(
            _validation.validate,
            self.n + 1,
            [
                self.points,
                self._base.weights,
                self._base._empty,
                self._base._empty,
                False,
                self._base._validation,
            ],
        )
        self._launch(input_ready, 1, [self._base._validation, self._reuse_ready, self.reuse_error])
        wp.capture_if(self._reuse_ready, self._attempt_reuse)

    def _attempt_reuse(self):
        self._refresh_original()
        self._reuse_completion.run()
        self._launch(
            completed,
            1,
            [
                self._reuse_filter.error,
                self._reuse_completion.error,
                self._reuse_completion.tetrahedra_count,
                len(self.tetrahedra),
                self.reuse_error,
                self.reuse_accepted,
            ],
        )
        wp.capture_if(self.reuse_accepted, self._publish_reuse)

    def _publish_reuse(self):
        source = self._reuse_completion
        wp.copy(self.tetrahedra, source.tetrahedra)
        wp.copy(self.tetrahedra_count, source.tetrahedra_count)
        self.error.zero_()
        self.certificate_error.zero_()
        self.recovered.fill_(1)
        # Pole extraction and later recertification need the matching face index.
        wp.copy(self._candidate, self.tetrahedra)
        wp.copy(self._candidate_count, self.tetrahedra_count)
        for name in (
            "statistics",
            "_used",
            "_table",
            "_occurrences",
            "_balance",
            "_parents",
            "_boundary",
            "_boundary_count",
        ):
            wp.copy(getattr(self._certificate, name), getattr(source._certificate, name))
        if self._certificate._anchors is not None:
            self._certificate._anchors.zero_()
            self._certificate._local.zero_()
            self._certificate._global.fill_(1)

    def run(self, stream=None):
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Workspace stream must use the input device")
        with wp.ScopedStream(stream):
            if stream.is_capturing:
                if stream not in self.device.captures:
                    raise RuntimeError(
                        "Use Warp ScopedCapture on the shared stream for conditional graphs"
                    )
                self._record_reuse()
            else:
                wp.capture_launch(self._reuse_graph, stream=stream)
        return self
