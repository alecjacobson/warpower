"""Private #154 experiment; does not edit production modules."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from warpower import _boundary
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from boundary_reduction_kernel import accumulate_samples_reduced


@contextmanager
def variant(reduced, block):
    original = wp.launch
    kernel = accumulate_samples_reduced if reduced else _boundary.accumulate_samples
    wp.load_module(module=kernel.module, device="cuda:0", block_dim=block)

    def launch(k, *args, **kwargs):
        if k is _boundary.accumulate_samples:
            args = list(args)
            if reduced:
                if args:
                    args[0] = (args[0] + block - 1) // block * block
                else:
                    kwargs['dim'] = (kwargs['dim'] + block - 1) // block * block
            kwargs['block_dim'] = block
            k = kernel
        return original(k, *args, **kwargs)

    wp.launch = launch
    try:
        yield
    finally:
        wp.launch = original


def independent(points, samples, weights, sizing, power):
    points = points.astype(np.float64)
    mass = np.zeros(len(points))
    moment = np.zeros_like(points)
    owners = np.full(len(samples), -1, dtype=np.int32)
    energy, total, maximum = 0., 0., 0.
    for first in range(0, len(samples), 512):
        last = min(first + 512, len(samples))
        q = samples[first:last].astype(np.float64)
        d = np.sum((q[:, None, :] - points[None, :, :]) ** 2, axis=2)
        ids = d.argmin(axis=1)
        weight = weights[first:last] / sizing[first:last].astype(np.float64) ** power
        enabled = weight > 0
        owners[first:last] = np.where(enabled, ids, -1)
        np.add.at(mass, ids, weight)
        np.add.at(moment, ids, weight[:, None] * (q - points[ids]))
        energy += np.sum(weight * d[np.arange(len(q)), ids])
        total += np.sum(weight)
        maximum = max(maximum, np.sqrt(d[np.arange(len(q))[enabled], ids[enabled]]).max(initial=0))
    return owners, mass, moment, np.array([energy, total, maximum])


def probe():
    root = Path('results/validation/vtm/surface-radix/timings/after')
    all_cases = []
    configs = [(False, 256), (False, 32), (False, 64), (False, 128)] + [(True, b) for b in (32, 64, 128, 256)]
    for name, path, subdivisions in [('torus',root/'torus1000.npz',4),('plate',root/'counterbored-plate/mesh.npz',16)]:
        saved = np.load(path)
        domain = warpower.SurfaceDomain(saved['input_points'], saved['input_faces'],
                                        surface_subdivisions=subdivisions, crease_subdivisions=subdivisions)
        workspaces = []
        for reduced, block in configs:
            with variant(reduced, block):
                workspace = warpower.BoundaryWorkspace(domain, saved['points'])
            workspace.run()
            assert workspace.error.numpy()[0] == 0
            workspaces.append(workspace)
        all_cases.append((name, saved['points'], domain, workspaces))
    wp.synchronize()
    report = {'scope':'BoundaryWorkspace captured run including hierarchy rebuild, corner assignment and targets, on fixed saved fitted inputs; NOT complete mesher timings.', 'cases': {}}
    for name, points, domain, workspaces in all_cases:
        for _ in range(3):
            for workspace in workspaces:
                workspace.run()
        wp.synchronize()
        before = cpu_quota_snapshot()
        samples = [[] for _ in configs]
        start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for repeat in range(21):
            order = range(len(configs)) if repeat % 2 == 0 else reversed(range(len(configs)))
            for i in order:
                wp.record_event(start)
                workspaces[i].run()
                wp.record_event(stop)
                wp.synchronize_event(stop)
                samples[i].append(wp.get_event_elapsed_time(start, stop))
        report['cases'][name] = {'samples':len(domain.surface_points), 'crease_slots':len(domain.crease_points),
            'cpu_quota_delta': cpu_quota_delta(before,cpu_quota_snapshot()),
            'variants':[{'reduced':r,'block':b,'gpu_ms':s,'median_ms':float(np.median(s))} for (r,b),s in zip(configs,samples)]}
        print(name,[(r['reduced'],r['block'],r['median_ms']) for r in report['cases'][name]['variants']], flush=True)
    for name, points, domain, workspaces in all_cases:
        references=[]
        for sample,weight,power in [(domain.surface_points,domain.surface_weights,4),(domain.crease_points,domain.crease_weights,3)]:
            references.append(independent(points,sample.numpy(),weight.numpy(),np.ones(len(sample)),power))
        baseline = workspaces[0]
        for workspace, row in zip(workspaces,report['cases'][name]['variants']):
            assert workspace.error.numpy()[0] == 0
            for attr in ('surface_owners','crease_owners','corner_sites','corner_owners','roles','role_counts'):
                np.testing.assert_array_equal(getattr(workspace,attr).numpy(),getattr(baseline,attr).numpy())
            for component, (owners,mass,moment,scalars) in enumerate(references):
                prefix = 'surface' if component==0 else 'crease'
                np.testing.assert_array_equal(getattr(workspace,prefix+'_owners').numpy(),owners)
                np.testing.assert_allclose(getattr(workspace,prefix+'_mass').numpy(),mass,rtol=3e-13,atol=1e-13)
                np.testing.assert_allclose(getattr(workspace,'_'+prefix+'_moment').numpy(),moment,rtol=3e-13,atol=1e-13)
                actual=np.array([workspace.energy.numpy()[component],workspace.total_weight.numpy()[component],workspace.max_distance.numpy()[component]])
                np.testing.assert_allclose(actual,scalars,rtol=3e-13,atol=1e-13)
            target_error=float(np.max(np.abs(workspace.targets.numpy().astype(np.float64)-baseline.targets.numpy())))
            np.testing.assert_allclose(workspace.targets.numpy(),baseline.targets.numpy(),rtol=1e-6,atol=1e-7)
            row.update(owners_roles_corners_exact=True,independent_moments_and_diagnostics=True,target_max_difference=target_error,
                       energy=workspace.energy.numpy().tolist(),total=workspace.total_weight.numpy().tolist())
        print(name,'all independent comparisons passed',flush=True)
    Path('.work/vtm/boundary-reduction-probe.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests',action='store_true')
    parser.add_argument('--block',type=int,default=128)
    args=parser.parse_args()
    wp.init()
    if args.tests:
        import pytest
        with variant(True,args.block):
            raise SystemExit(pytest.main(['-q','tests/test_boundary.py']))
    probe()


if __name__=='__main__':
    main()
