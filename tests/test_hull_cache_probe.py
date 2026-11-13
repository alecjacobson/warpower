"""The experimental cache must retain the original geometric guarantees."""

import numpy as np
import pytest

from benchmarks.replay.hull_cache_probe import HullCache, install
from tests import test_reuse_delaunay as reuse_tests
from tests.test_facet_completion import canonical, execute
from tests.test_local_certificate import positive
from tests.test_proved_completion import completion, global_certificate


@pytest.mark.parametrize("subset", ["one", "hole", "full"])
def test_cached_hull_proofs_match_independent_uncached_certificate(subset):
    from scipy.spatial import Delaunay

    points = np.random.default_rng(941).normal(size=(32, 3)).astype(np.float32)
    full = positive(points, Delaunay(points).simplices)
    seeds = {"one": full[:1], "hole": full[1:], "full": full}[subset]
    workspace, retained = completion(points, seeds)
    cache = HullCache(workspace)
    for _ in range(2):
        error, actual = execute(workspace)
        assert error == 0
        np.testing.assert_array_equal(canonical(actual), canonical(full))
        global_certificate(workspace)
        misses, hits = cache.statistics.numpy()
        assert misses > 0
        if subset == "one":
            assert hits > 0
        if subset == "full":
            assert hits == 0  # No stale proofs survive a second unchanged run.


def _install_in_existing_regression(monkeypatch):
    original = reuse_tests.ReuseDelaunayWorkspace

    def create(*args, **kwargs):
        workspace = original(*args, **kwargs)
        install(workspace)
        return workspace

    monkeypatch.setattr(reuse_tests, "ReuseDelaunayWorkspace", create)


@pytest.mark.parametrize("rounds", [0, 4])
def test_device_mutation_invalid_input_and_recovery_clear_cached_proofs(monkeypatch, rounds):
    _install_in_existing_regression(monkeypatch)
    reuse_tests.test_changing_device_producer_invalid_phase_and_consumer_in_one_graph(rounds)


@pytest.mark.parametrize("corruption", ["count", "indices", "empty"])
def test_bad_retained_mesh_does_not_acquire_hull_proofs(monkeypatch, corruption):
    _install_in_existing_regression(monkeypatch)
    reuse_tests.test_bad_retained_fragment_cannot_bypass_fresh_construction(corruption)


def test_exhausted_capacity_does_not_publish_partial_mesh(monkeypatch):
    _install_in_existing_regression(monkeypatch)
    reuse_tests.test_exhausted_output_capacity_stays_empty_across_replays()


def test_exact_ties_duplicates_and_ulp_motion(monkeypatch):
    _install_in_existing_regression(monkeypatch)
    reuse_tests.test_exact_cospherical_duplicates_and_one_ulp_motion()
