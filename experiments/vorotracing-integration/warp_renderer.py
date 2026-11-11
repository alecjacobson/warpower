"""Warp Voronoi texture tracing and explicit replay derivatives.

The reference is VoroTracing 75910484. Its training contract uses rounded FP16
neighbor differences for traversal, FP32 appearance, texture-coordinate stop
gradients, and a density-only distortion derivative. We retain those choices
explicitly instead of silently changing the optimizer with full autodiff.
All kernels in this module are written in Warp; Torch supplies tensor ownership
and the autograd boundary. No RadFoam/VoroTracing CUDA extension is loaded.
"""

import torch
import warp as wp
from numerics import sanitize

from warpower import _bvh
from warpower._boundary_precision import nearest_site32
from warpower._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": True})
vec4h = wp.types.vector(4, wp.float16)
HASH_X = wp.constant(wp.uint32(0x9E3779B1))
HASH_Y = wp.constant(wp.uint32(0x85EBCA77))
HASH_Z = wp.constant(wp.uint32(0xC2B2AE3D))


@wp.func
def reference_dot(a: wp.vec3, b: wp.vec3):
    # Eigen's three-element reduction is x + (y + z). Warp's generic dot
    # accumulates (x + y) + z. Matching the reference matters near a vertex:
    # a last-bit face-distance change can select a different traversal path.
    return a[0] * b[0] + (a[1] * b[1] + a[2] * b[2])


@wp.func
def reference_normalize(d: wp.vec3):
    # The caller supplies a nonzero ray or checks the 1e-8 surface cutoff.
    # wp.normalize uses a larger kEps cutoff and changes small-cell texture UVs.
    return d / wp.sqrt(reference_dot(d, d))


@wp.func
def oct_uv(d: wp.vec3):
    inv = 1.0 / wp.max(wp.abs(d[0]) + wp.abs(d[1]) + wp.abs(d[2]), 1.0e-10)
    u, v = d[0] * inv, d[1] * inv
    if d[2] < 0.0:
        su, sv = float(1.0), float(1.0)
        if u < 0.0:
            su = -1.0
        if v < 0.0:
            sv = -1.0
        u, v = (1.0 - wp.abs(v)) * su, (1.0 - wp.abs(u)) * sv
    return wp.vec2((u * 0.5 + 0.5) * 8.0 - 0.5, (v * 0.5 + 0.5) * 8.0 - 0.5)


@wp.func
def wrapped_texel(u: int, v: int):
    # Only the immediately adjacent border is sampled by oct_uv.
    wu, wv = (u + 8) % 8, (v + 8) % 8
    fu, fv = int(0), int(0)
    if u < 0 or u >= 8:
        fu = 1
    if v < 0 or v >= 8:
        fv = 1
    if fu != fv:
        wu, wv = 7 - wu, 7 - wv
    return wv * 8 + wu


@wp.struct
class Bilinear:
    ids: wp.vec4i
    weights: wp.vec4


@wp.func
def interpolation(uv: wp.vec2):
    u, v = int(wp.floor(uv[0])), int(wp.floor(uv[1]))
    a, b = uv[0] - float(u), uv[1] - float(v)
    s = Bilinear()
    s.ids = wp.vec4i(
        wrapped_texel(u, v),
        wrapped_texel(u + 1, v),
        wrapped_texel(u, v + 1),
        wrapped_texel(u + 1, v + 1),
    )
    s.weights = wp.vec4((1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b)
    return s


@wp.func
def texture(t: wp.array2d[wp.vec3], cell: int, s: Bilinear):
    return (
        s.weights[0] * t[cell, s.ids[0]]
        + s.weights[1] * t[cell, s.ids[1]]
        + s.weights[2] * t[cell, s.ids[2]]
        + s.weights[3] * t[cell, s.ids[3]]
    )


@wp.func
def scatter(t: wp.array2d[wp.vec3], cell: int, s: Bilinear, g: wp.vec3):
    for j in range(4):
        wp.atomic_add(t, cell, s.ids[j], s.weights[j] * g)


@wp.func
def surface_sample(origin: wp.vec3, direction: wp.vec3, t: float, site: wp.vec3):
    d = origin + t * direction - site
    if wp.sqrt(reference_dot(d, d)) > 1.0e-8:
        d = reference_normalize(d)
    else:
        d = direction
    return interpolation(oct_uv(d))


@wp.func
def sigmoid(x: wp.vec3):
    return wp.vec3(
        1.0 / (1.0 + wp.exp(-x[0])), 1.0 / (1.0 + wp.exp(-x[1])), 1.0 / (1.0 + wp.exp(-x[2]))
    )


@wp.func
def half_rgb(t: vec4h):
    return wp.vec3(float(t[0]), float(t[1]), float(t[2]))


@wp.func
def half_texture(t: wp.array[wp.uint64], cell: int, s: Bilinear):
    # Load an aligned 64-bit word, then reinterpret the four half channels.
    # A vec4h array has only two-byte type alignment and otherwise generates
    # separate channel loads, losing the paper's packed-texel optimization.
    # R=8 is fixed by this adapter, so flattening also removes a dynamic stride.
    return (
        s.weights[0] * half_rgb(wp.cast(t[cell * 64 + s.ids[0]], vec4h))
        + s.weights[1] * half_rgb(wp.cast(t[cell * 64 + s.ids[1]], vec4h))
        + s.weights[2] * half_rgb(wp.cast(t[cell * 64 + s.ids[2]], vec4h))
        + s.weights[3] * half_rgb(wp.cast(t[cell * 64 + s.ids[3]], vec4h))
    )


@wp.kernel
def prepare_edges(
    points: wp.array[wp.vec3],
    adj: wp.array[wp.uint32],
    offsets: wp.array[wp.uint32],
    edges: wp.array[vec4h],
):
    cell = wp.tid()
    p = points[cell]
    for e in range(int(offsets[cell]), int(offsets[cell + 1])):
        d = points[int(adj[e])] - p
        edges[e] = vec4h(wp.float16(d[0]), wp.float16(d[1]), wp.float16(d[2]), wp.float16(0.0))


@wp.struct
class Exit:
    t: float
    cell: int


@wp.func
def exit_face(
    p: wp.vec3,
    o: wp.vec3,
    d: wp.vec3,
    cell: int,
    adj: wp.array[wp.uint32],
    offsets: wp.array[wp.uint32],
    edges: wp.array[wp.uint64],
):
    result = Exit()
    result.t = wp.inf
    result.cell = -1
    for e in range(int(offsets[cell]), int(offsets[cell + 1])):
        h = wp.cast(edges[e], vec4h)
        n = wp.vec3(float(h[0]), float(h[1]), float(h[2]))
        dp = reference_dot(n, d)
        if dp > 0.0:
            t = reference_dot(p + n * 0.5 - o, n) / dp
            if t < result.t:
                result.t = t
                result.cell = e
    if result.cell >= 0:
        result.cell = int(adj[result.cell])
    return result


@wp.kernel
def trace_forward(
    points: wp.array[wp.vec3],
    diffuse: wp.array2d[wp.vec3],
    specular: wp.array2d[wp.vec3],
    density: wp.array[float],
    adj: wp.array[wp.uint32],
    offsets: wp.array[wp.uint32],
    edges: wp.array[wp.uint64],
    rays: wp.array2d[wp.vec3],
    starts: wp.array[wp.uint32],
    threshold: float,
    max_steps: int,
    rgba: wp.array[wp.vec4],
    distortion: wp.array[float],
    totals: wp.array[wp.vec2],
    counts: wp.array[int],
):
    ray = wp.tid()
    o, d = rays[ray, 0], reference_normalize(rays[ray, 1])
    view_sample = interpolation(oct_uv(-d))
    cell = int(starts[ray])
    t0, trans = float(0.0), float(1.0)
    color = wp.vec3(0.0)
    W, S, pair_sum, self_sum = float(0.0), float(0.0), float(0.0), float(0.0)
    steps = int(0)
    while True:
        steps += 1
        if steps > max_steps:
            break
        p = points[cell]
        hit = exit_face(p, o, d, cell, adj, offsets, edges)
        if hit.cell < 0:
            break
        if hit.t > t0:
            sigma = density[cell]
            rgb = wp.vec3(0.0)
            if sigma > 1.0e-6:
                surface = surface_sample(o, d, t0, p)
                rgb = sigmoid(
                    texture(diffuse, cell, surface) + texture(specular, cell, view_sample)
                )
            alpha = 1.0 - wp.exp(-sigma * (hit.t - t0))
            weight = trans * alpha
            s0, s1 = 1.0 - 1.0 / (1.0 + t0), 1.0 - 1.0 / (1.0 + hit.t)
            middle, width = (s0 + s1) * 0.5, s1 - s0
            pair_sum += weight * (middle * W - S)
            self_sum += weight * weight * width
            W += weight
            S += weight * middle
            color += weight * rgb
            trans *= 1.0 - alpha
            if trans <= threshold:
                break
        t0 = wp.max(t0, hit.t)
        cell = hit.cell
    rgba[ray] = wp.vec4(color[0], color[1], color[2], 1.0 - trans)
    distortion[ray] = 2.0 * pair_sum + self_sum / 3.0
    totals[ray] = wp.vec2(W, S)
    counts[ray] = steps


@wp.func
def intersection_grad(p: wp.vec3, q: wp.vec3, o: wp.vec3, d: wp.vec3):
    n = q - p
    numerator = reference_dot((p + q) * 0.5 - o, n)
    denominator = reference_dot(n, d)
    return (numerator * d + denominator * (o - p)) / (denominator * denominator)


# The full-checkpoint launch sweep favors this cap over 48/64/default on L40.
@wp.kernel(cuda_max_registers=80)
def trace_inference(
    points: wp.array[wp.vec3],
    diffuse: wp.array[wp.uint64],
    specular: wp.array[wp.uint64],
    density: wp.array[float],
    adj: wp.array[wp.uint32],
    offsets: wp.array[wp.uint32],
    edges: wp.array[wp.uint64],
    rays: wp.array2d[wp.vec3],
    starts: wp.array[wp.uint32],
    permutation: wp.array[int],
    threshold: float,
    max_steps: int,
    skip: float,
    rgba: wp.array[vec4h],
):
    ray = wp.tid()
    if permutation.shape[0] > 0:
        ray = permutation[ray]
    o, d = rays[ray, 0], reference_normalize(rays[ray, 1])
    view_sample = interpolation(oct_uv(-d))
    cell = int(starts[ray])
    t0, trans = float(0.0), float(1.0)
    color = wp.vec3(0.0)
    for step in range(max_steps):
        p = points[cell]
        hit = exit_face(p, o, d, cell, adj, offsets, edges)
        if hit.cell < 0:
            break
        if hit.t > t0:
            sigma = density[cell]
            if sigma > 1.0e-6:
                alpha = 1.0 - wp.exp(-sigma * (hit.t - t0))
                weight = trans * alpha
                if weight >= skip:
                    surface = surface_sample(o, d, t0, p)
                    rgb = sigmoid(
                        half_texture(diffuse, cell, surface)
                        + half_texture(specular, cell, view_sample)
                    )
                    color += weight * rgb
                trans *= 1.0 - alpha
            if trans <= threshold:
                break
        t0 = wp.max(t0, hit.t)
        cell = hit.cell
    rgba[ray] = vec4h(
        wp.float16(color[0]), wp.float16(color[1]), wp.float16(color[2]), wp.float16(1.0 - trans)
    )


@wp.kernel
def trace_backward(
    points: wp.array[wp.vec3],
    diffuse: wp.array2d[wp.vec3],
    specular: wp.array2d[wp.vec3],
    density: wp.array[float],
    adj: wp.array[wp.uint32],
    offsets: wp.array[wp.uint32],
    edges: wp.array[wp.uint64],
    rays: wp.array2d[wp.vec3],
    starts: wp.array[wp.uint32],
    threshold: float,
    max_steps: int,
    rgba: wp.array[wp.vec4],
    distortion: wp.array[float],
    totals: wp.array[wp.vec2],
    grad_rgba: wp.array[wp.vec4],
    grad_distortion: wp.array[float],
    grad_points: wp.array[wp.vec3],
    grad_diffuse: wp.array2d[wp.vec3],
    grad_specular: wp.array2d[wp.vec3],
    grad_density: wp.array[float],
):
    ray = wp.tid()
    o, d = rays[ray, 0], reference_normalize(rays[ray, 1])
    view_sample = interpolation(oct_uv(-d))
    cell = int(starts[ray])
    t0, trans = float(0.0), float(1.0)
    color = wp.vec3(0.0)
    W, S, running_deriv = float(0.0), float(0.0), float(0.0)
    total_deriv = 2.0 * distortion[ray] * grad_distortion[ray]
    final, gradient = rgba[ray], grad_rgba[ray]
    final_rgb = wp.vec3(final[0], final[1], final[2])
    rgb_gradient = wp.vec3(gradient[0], gradient[1], gradient[2])
    previous = int(-1)
    previous_p = wp.vec3(0.0)
    previous_g, current_g = wp.vec3(0.0), wp.vec3(0.0)
    steps = int(0)
    while True:
        steps += 1
        if steps > max_steps:
            break
        p = points[cell]
        hit = exit_face(p, o, d, cell, adj, offsets, edges)
        if hit.cell < 0:
            break
        if hit.t > t0:
            sigma, delta = density[cell], hit.t - t0
            rgb = wp.vec3(0.0)
            surface = surface_sample(o, d, t0, p)
            if sigma > 1.0e-6:
                rgb = sigmoid(
                    texture(diffuse, cell, surface) + texture(specular, cell, view_sample)
                )
            alpha = 1.0 - wp.exp(-sigma * delta)
            weight = trans * alpha
            color += weight * rgb
            rest = (final_rgb - color) / (trans * (1.0 - alpha + 1.0e-6))
            ga = trans * reference_dot(rgb - rest, rgb_gradient)
            ga += (1.0 - final[3]) * gradient[3] / (1.0 - alpha + 1.0e-6)
            ds = delta * (1.0 - alpha)
            gs = ga * ds
            gt1 = ga * sigma * (1.0 - alpha)
            gt0 = -gt1
            # Upstream distortion deliberately affects densities, not positions.
            gd = grad_distortion[ray]
            if gd != 0.0:
                s0, s1 = 1.0 - 1.0 / (1.0 + t0), 1.0 - 1.0 / (1.0 + hit.t)
                middle, width = (s0 + s1) * 0.5, s1 - s0
                remaining_W = totals[ray][0] - W - weight
                remaining_S = totals[ray][1] - S - weight * middle
                gw = (
                    2.0 * (middle * W - S)
                    + 2.0 * (remaining_S - middle * remaining_W)
                    + (2.0 / 3.0) * weight * width
                ) * gd
                weighted_deriv = gw * weight
                gs += gw * trans * ds - delta * (total_deriv - running_deriv - weighted_deriv)
                running_deriv += weighted_deriv
                W += weight
                S += weight * middle
            if previous >= 0:
                previous_g += gt0 * intersection_grad(previous_p, p, o, d)
            current_g += gt0 * intersection_grad(p, previous_p, o, d)
            current_g += gt1 * intersection_grad(p, points[hit.cell], o, d)
            next_g = gt1 * intersection_grad(points[hit.cell], p, o, d)
            if previous >= 0:
                wp.atomic_add(grad_points, previous, previous_g)
            previous, previous_p = cell, p
            previous_g, current_g = current_g, next_g
            if sigma > 1.0e-6:
                g_logit = wp.cw_mul(wp.cw_mul(weight * rgb_gradient, rgb), wp.vec3(1.0) - rgb)
                scatter(grad_diffuse, cell, surface, g_logit)
                scatter(grad_specular, cell, view_sample, g_logit)
            wp.atomic_add(grad_density, cell, gs)
            trans *= 1.0 - alpha
            if trans < threshold:
                wp.atomic_add(grad_points, previous, previous_g)
                break
        t0 = wp.max(t0, hit.t)
        cell = hit.cell


@wp.kernel
def query_nearest(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    result: wp.array[wp.uint32],
):
    i = wp.tid()
    hit = nearest_site32(tree, points, queries[i], active, occupied, False)
    result[i] = wp.uint32(hit.index)


@wp.func
def origin_bucket(p: wp.vec3):
    # Float-bit hashes are used only to select a candidate representative.
    # Equality is checked before reuse, so a collision never changes the answer.
    a, b, c = wp.cast(p[0], wp.uint32), wp.cast(p[1], wp.uint32), wp.cast(p[2], wp.uint32)
    h = (a * HASH_X) ^ (b * HASH_Y) ^ (c * HASH_Z)
    h = h ^ (h >> wp.uint32(16))
    return int(h & wp.uint32(8191))


@wp.kernel
def camera_representatives(queries: wp.array[wp.vec3], representatives: wp.array[int]):
    i = wp.tid()
    bucket = origin_bucket(queries[i])
    wp.atomic_min(representatives, bucket, i)


@wp.kernel
def query_representatives(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    representatives: wp.array[int],
    result: wp.array[wp.uint32],
):
    bucket = wp.tid()
    ray = representatives[bucket]
    if ray < queries.shape[0]:
        result[bucket] = wp.uint32(
            nearest_site32(tree, points, queries[ray], active, occupied, False).index
        )


@wp.kernel
def expand_representatives(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    representatives: wp.array[int],
    representative_starts: wp.array[wp.uint32],
    result: wp.array[wp.uint32],
):
    i = wp.tid()
    query = queries[i]
    bucket = origin_bucket(query)
    representative = queries[representatives[bucket]]
    if (
        query[0] == representative[0]
        and query[1] == representative[1]
        and query[2] == representative[2]
    ):
        result[i] = representative_starts[bucket]
    else:
        result[i] = wp.uint32(nearest_site32(tree, points, query, active, occupied, False).index)


def tensor(t, dtype=None):
    if t.dtype == torch.uint32:
        t = t.view(torch.int32)
        dtype = wp.uint32
    return wp.from_torch(t.detach(), dtype=dtype)


def stream():
    return wp.stream_from_torch(torch.cuda.current_stream())


def prepared_edges(points, adjacency, offsets):
    out = torch.empty((adjacency.numel(), 4), dtype=torch.float16, device=points.device)
    wp.launch(
        prepare_edges,
        len(points),
        inputs=[tensor(points, wp.vec3), tensor(adjacency), tensor(offsets), tensor(out, vec4h)],
        stream=stream(),
    )
    return out


def arguments(points, diffuse, specular, density, adj, offsets, edges, rays, starts):
    return [
        tensor(points, wp.vec3),
        tensor(diffuse.reshape(-1, 64, 3), wp.vec3),
        tensor(specular.reshape(-1, 64, 3), wp.vec3),
        tensor(density.reshape(-1)),
        tensor(adj),
        tensor(offsets),
        tensor(edges.view(torch.int64).reshape(-1), wp.uint64),
        tensor(rays.reshape(-1, 2, 3), wp.vec3),
        tensor(starts.reshape(-1)),
    ]


def check_storage(points, diffuse, specular, density, offsets, rays, starts, *, packed):
    channels = 256 if packed else 192
    dtype = torch.float16 if packed else torch.float32
    n = len(points)
    if n == 0 or points.shape != (n, 3) or points.dtype != torch.float32:
        raise ValueError("Expected nonempty FP32 positions of shape (sites, 3)")
    if any(t.shape != (n, channels) or t.dtype != dtype for t in [diffuse, specular]):
        raise ValueError(
            f"Expected two 8x8 appearance maps of shape ({n}, {channels}) and dtype {dtype}"
        )
    if density.numel() != n or density.dtype != torch.float32 or offsets.numel() != n + 1:
        raise ValueError("Density and CSR offsets must match the site count")
    if rays.shape[-1] != 6 or rays.dtype != torch.float32 or starts.numel() != rays.numel() // 6:
        raise ValueError("Expected FP32 (..., 6) rays and one starting cell per ray")


def infer(
    points,
    diffuse_fp16,
    specular_fp16,
    density,
    adjacency,
    offsets,
    rays,
    start_point,
    adjacent_diff,
    ray_perm,
    weight_threshold=0.001,
    max_intersections=1024,
    cell_skip_threshold=0.0,
):
    check_storage(
        points, diffuse_fp16, specular_fp16, density, offsets, rays, start_point, packed=True
    )
    n = rays.numel() // 6
    rgba = torch.empty((n, 4), dtype=torch.float16, device=points.device)
    wp.launch(
        trace_inference,
        n,
        inputs=[
            tensor(points, wp.vec3),
            tensor(diffuse_fp16.view(torch.int64).reshape(-1), wp.uint64),
            tensor(specular_fp16.view(torch.int64).reshape(-1), wp.uint64),
            tensor(density.reshape(-1)),
            tensor(adjacency),
            tensor(offsets),
            tensor(adjacent_diff.view(torch.int64).reshape(-1), wp.uint64),
            tensor(rays.contiguous().reshape(-1, 2, 3), wp.vec3),
            tensor(start_point.contiguous().reshape(-1)),
            tensor(ray_perm),
            weight_threshold,
            max_intersections,
            cell_skip_threshold,
            tensor(rgba, vec4h),
        ],
        stream=stream(),
        block_dim=128,
    )
    return rgba.reshape(*rays.shape[:-1], 4)


class Trace(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,
        diffuse,
        specular,
        density,
        adj,
        offsets,
        rays,
        starts,
        threshold=0.001,
        max_steps=1024,
    ):
        check_storage(points, diffuse, specular, density, offsets, rays, starts, packed=False)
        rays, starts = rays.contiguous(), starts.contiguous()
        n = rays.numel() // 6
        edges = prepared_edges(points, adj, offsets)
        rgba = torch.empty((n, 4), device=points.device)
        dist = torch.empty(n, device=points.device)
        totals = torch.empty((n, 2), device=points.device)
        counts = torch.empty(n, dtype=torch.int32, device=points.device)
        inputs = arguments(points, diffuse, specular, density, adj, offsets, edges, rays, starts)
        wp.launch(
            trace_forward,
            n,
            inputs=inputs
            + [
                threshold,
                max_steps,
                tensor(rgba, wp.vec4),
                tensor(dist),
                tensor(totals, wp.vec2),
                tensor(counts),
            ],
            stream=stream(),
        )
        ctx.save_for_backward(
            points,
            diffuse,
            specular,
            density,
            adj,
            offsets,
            edges,
            rays,
            starts,
            rgba,
            dist,
            totals,
        )
        ctx.threshold, ctx.max_steps = threshold, max_steps
        ctx.mark_non_differentiable(counts)
        shape = rays.shape[:-1]
        return rgba.reshape(*shape, 4), dist.reshape(shape), counts.reshape(shape)

    @staticmethod
    def backward(ctx, grad_rgba, grad_dist, grad_counts):
        p, diffuse, specular, density, adj, offsets, edges, rays, starts, rgba, dist, totals = (
            ctx.saved_tensors
        )
        grads = [torch.zeros_like(t) for t in [p, diffuse, specular, density]]
        if grad_rgba is None:
            grad_rgba = torch.zeros_like(rgba)
        if grad_dist is None:
            grad_dist = torch.zeros_like(dist)
        inputs = arguments(p, diffuse, specular, density, adj, offsets, edges, rays, starts)
        wp.launch(
            trace_backward,
            len(rgba),
            inputs=inputs
            + [
                ctx.threshold,
                ctx.max_steps,
                tensor(rgba, wp.vec4),
                tensor(dist),
                tensor(totals, wp.vec2),
                tensor(grad_rgba.contiguous().reshape(-1, 4), wp.vec4),
                tensor(grad_dist.contiguous().reshape(-1)),
                tensor(grads[0], wp.vec3),
                tensor(grads[1].reshape(-1, 64, 3), wp.vec3),
                tensor(grads[2].reshape(-1, 64, 3), wp.vec3),
                tensor(grads[3].reshape(-1)),
            ],
            stream=stream(),
        )
        for category, gradient in enumerate(grads):
            sanitize(gradient, category)
        return *grads, None, None, None, None, None, None


class Nearest:
    """Warp-owned site hierarchy; camera queries do not use cuBQL or RadFoam."""

    def build(self, points):
        self.points = points
        with wp.ScopedStream(stream()):
            self.hierarchy = SiteHierarchy(tensor(points, wp.vec3))
        self.active = torch.tensor([len(points)], dtype=torch.int32, device=points.device)
        self.occupied = torch.empty(0, dtype=torch.int32, device=points.device)

    def query(self, points, queries):
        output = torch.empty(len(queries), device=points.device, dtype=torch.uint32)
        inputs = [
            self.hierarchy.tree,
            tensor(points, wp.vec3),
            tensor(queries.contiguous(), wp.vec3),
            tensor(self.active),
            tensor(self.occupied),
        ]
        if len(queries) > 4096:
            # Camera origins repeat in random-ray training. Three linear passes
            # replace torch.unique(dim=0), without assuming a fixed camera set.
            representatives = torch.full(
                (8192,), len(queries), dtype=torch.int32, device=points.device
            )
            starts = torch.empty(8192, dtype=torch.uint32, device=points.device)
            wp.launch(
                camera_representatives,
                len(queries),
                inputs=[inputs[2], tensor(representatives)],
                stream=stream(),
            )
            wp.launch(
                query_representatives,
                8192,
                inputs=inputs + [tensor(representatives), tensor(starts)],
                stream=stream(),
            )
            wp.launch(
                expand_representatives,
                len(queries),
                inputs=inputs + [tensor(representatives), tensor(starts), tensor(output)],
                stream=stream(),
            )
            return output
        wp.launch(
            query_nearest,
            len(queries),
            inputs=inputs + [tensor(output)],
            stream=stream(),
        )
        return output
