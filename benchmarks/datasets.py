"""Synthetic distributions from Paragram, Section 5.1."""

import numpy as np
from scipy.spatial import cKDTree


def generate(n, distribution="uniform", seed=42, weighted=False):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        points = rng.uniform(-10, 10, (n, 3))
    elif distribution == "gradient":
        points = rng.uniform(-10, 10, (n, 3))
        points[:, 0] = -10 + 20 * np.sqrt(rng.random(n))
    elif distribution in ("cluster5", "cluster10"):
        k = int(distribution.removeprefix("cluster"))
        centers = rng.uniform(-10, 10, (k, 3))
        points = rng.normal(0, 0.1, (n, 3)) + centers[np.arange(n) % k]
        np.clip(points, -10, 10, out=points)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    points = points.astype(np.float32)
    weights = None
    if weighted:
        # Evaluate all sites, rather than an approximate sampled median.
        distances = cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]
        sigma = float(np.median(distances)) ** 2 / 3
        weights = rng.normal(0, sigma, n).astype(np.float32)
    return points, weights
