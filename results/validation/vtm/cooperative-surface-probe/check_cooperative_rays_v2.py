"""Strong boundary/scale/caller-FMA comparison for the private #197 probe."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from examples.vtm_inputs import counterbored_plate, surface
from tests.test_domain import cube, solid_angle_inside, torus
from warpower import SurfaceDomain
import cooperative_rays_v2 as kernels


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('.work/vtm/cooperative-rays-check-v2.json')
    assert not out.exists()
    report = dict(runtime=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  scope='Private ray queries only; no production adoption or end-to-end claim.',
                  sources={p: sha(p) for p in [__file__, kernels.__file__, 'warpower/_surface_bvh.py']},
                  versions={'warp': wp.__version__, 'numpy': np.__version__}, cases=[])
    names = ['leaf_1', 'leaf_2', 'leaf_4', 'subtree_2', 'subtree_4', 'subtree_8', 'subtree_16', 'subtree_32']
    for fuse in [False, True]:
        wp.set_module_options({'fuse_fp': fuse}, module=kernels)
        wp.load_module(module=kernels, device='cuda:0', block_dim=32)
        for case in ['cube', 'torus', 'plate', 'thin-hand', 'cavity', 'disconnected']:
            for exponent in [-60, 0, 60]:
                p, f = cube() if case == 'cube' else torus(16, 8) if case == 'torus' else counterbored_plate(16) if case == 'plate' else surface(case)
                p = (p * np.float32(2.0**exponent)).astype(np.float32)
                domain = SurfaceDomain(p, f, surface_subdivisions=1, crease_subdivisions=1)
                assert domain.error.numpy()[0] == 0
                rng = np.random.default_rng(918)
                lo, hi = p.min(axis=0).astype(float), p.max(axis=0).astype(float)
                ordinary = rng.uniform(lo - .1*(hi-lo), hi + .1*(hi-lo), (100, 3))
                triangles = p[f].astype(float)
                centers = triangles.mean(axis=1)
                probes = np.concatenate([ordinary, p, triangles[:, :2].mean(axis=1), centers, centers])
                excluded = np.r_[np.full(len(probes)-len(f), -1, np.int32), np.arange(len(f), dtype=np.int32)]
                points = wp.array(probes, dtype=wp.vec3d)
                faces = wp.array(excluded, dtype=int)
                expected = wp.empty((len(probes), 3), dtype=int)
                actual = wp.empty_like(expected)
                args = [domain.tree, points, faces, actual]
                wp.launch(kernels.reference, len(probes), inputs=args[:-1]+[expected], block_dim=32)
                ref = expected.numpy()
                oracle = solid_angle_inside(p, f, ordinary).astype(np.int32)
                np.testing.assert_array_equal(ref[:100, 2], oracle)
                row = dict(case=case, exponent=exponent, fuse_fp=fuse, faces=len(f), queries=len(probes),
                           input_sha256=hashlib.sha256(p.tobytes()+f.tobytes()+probes.tobytes()+excluded.tobytes()).hexdigest(),
                           reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(), solid_angle_random_queries=100, variants={})
                for name in names:
                    group = int(name.rsplit('_', 1)[1])
                    dim = (len(probes)+(32//group)-1)//(32//group)
                    wp.launch_tiled(getattr(kernels, name), dim, inputs=args, block_dim=32)
                    values = actual.numpy()
                    bad = np.flatnonzero(np.any(values != ref, axis=1))
                    wp.launch_tiled(getattr(kernels, name+'_side'), dim, inputs=args, block_dim=32)
                    side = actual.numpy()[:, 2]
                    side_bad = np.flatnonzero(side != ref[:, 2])
                    row['variants'][name] = dict(diagnostic_mismatches=len(bad), side_only_mismatches=len(side_bad),
                        diagnostic_mismatch_fields=np.count_nonzero(values != ref, axis=0).tolist(),
                        diagnostic_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        side_sha256=hashlib.sha256(side.tobytes()).hexdigest(),
                        examples=[dict(i=int(i), point=probes[i].tolist(), excluded=int(excluded[i]), reference=ref[i].tolist(), candidate=values[i].tolist(), side_only=int(side[i])) for i in np.union1d(bad, side_bad)[:20]])
                report['cases'].append(row)
                out.write_text(json.dumps(report, indent=2)+'\n')
                print(case, exponent, fuse, {k:(v['diagnostic_mismatches'],v['side_only_mismatches']) for k,v in row['variants'].items()}, flush=True)
    report['finished'] = True
    report['all_equal'] = all(not v['diagnostic_mismatches'] and not v['side_only_mismatches'] for row in report['cases'] for v in row['variants'].values())
    out.write_text(json.dumps(report, indent=2)+'\n')
    print('ALL_EQUAL', report['all_equal'], flush=True)


if __name__ == '__main__':
    main()
