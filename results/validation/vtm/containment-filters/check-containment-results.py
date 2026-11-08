"""Offline checks/publication, after the paired timed jobs have finished."""
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from benchmarks.mesh_quality import measure

root = Path('/home/horde/projects/power-diagram')
work = root / '.work/vtm'
tree = root
baseline_tree = work / 'extraction-geometry'
out = tree / 'results/validation/vtm'
out.mkdir(parents=True, exist_ok=True)


def canonical(x):
    x = np.sort(x, axis=1)
    return x[np.lexsort(x.T[::-1])]


def oriented(x):
    return sorted(min(tuple(f), tuple(np.roll(f, 1)), tuple(np.roll(f, 2))) for f in x)


def check_quota(data, keys):
    for key in keys:
        assert all(v['throttled_periods'] == 0 for v in data[key].values()), (key, data[key])


def paths(case, phase):
    stem = f'containment-{phase}-{case}'
    if case.startswith('torus'):
        return work / (stem + '.json'), work / (stem + '.npz'), None
    return work / stem / 'result.json', work / stem / 'mesh.npz', work / stem / 'mesh.vtu'


def native_check(mesh, name):
    pp, tp = work / (name + '-check-points.txt'), work / (name + '-check-tets.txt')
    np.savetxt(pp, mesh['points'], fmt='%.17g', header=str(len(mesh['points'])), comments='')
    np.savetxt(tp, mesh['full_tetrahedra'], fmt='%d', header=str(len(mesh['full_tetrahedra'])), comments='')
    p = subprocess.run([str(work / 'check-delaunay'), str(pp), str(tp)], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def independent(mesh, name, vtu):
    record = dict(full_hull=native_check(mesh, name))
    assert record['full_hull']['valid']
    record['material_quality'] = measure(mesh['points'], mesh['tetrahedra'], mesh['input_points'], mesh['input_faces'])
    if 'radius_ratio' in mesh:
        p = record['material_quality']['radius_ratio_percentiles']
        np.testing.assert_allclose(list(p.values()), np.percentile(mesh['radius_ratio'], [0, 1, 5, 50, 95, 100]), rtol=1e-10, atol=1e-12)
    if vtu:
        xml = ET.parse(vtu).getroot()
        points = np.fromstring(xml.find('.//Points/DataArray').text, sep=' ', dtype=np.float32).reshape(-1, 3)
        tets = np.fromstring(xml.find(".//Cells/DataArray[@Name='connectivity']").text, sep=' ', dtype=np.int32).reshape(-1, 4)
        quality = np.fromstring(xml.find(".//CellData/DataArray[@Name='radius_ratio']").text, sep=' ')
        offsets = np.fromstring(xml.find(".//Cells/DataArray[@Name='offsets']").text, sep=' ', dtype=int)
        types = np.fromstring(xml.find(".//Cells/DataArray[@Name='types']").text, sep=' ', dtype=int)
        np.testing.assert_array_equal(points.view(np.uint32), mesh['points'].view(np.uint32))
        np.testing.assert_array_equal(tets, mesh['tetrahedra'])
        np.testing.assert_array_equal(quality, mesh['radius_ratio'])
        np.testing.assert_array_equal(offsets, 4 * np.arange(1, len(tets) + 1))
        np.testing.assert_array_equal(types, np.full(len(tets), 10))
        record['vtu_round_trip_exact'] = True
    return record


cases = {}
for case in ['torus1000', 'torus4096', 'graded-torus', 'thin-hand', 'cavity', 'disconnected']:
    rows, meshes = {}, {}
    for phase in ['before', 'after']:
        report, mesh, vtu = paths(case, phase)
        data = json.loads(report.read_text())
        assert data['error'] == 0 and data['valid'] == 1
        if case.startswith('torus'):
            check_quota(data, ['cpu_quota_delta', 'initial_replays_cpu_quota_delta'])
            milliseconds = data['median_gpu_milliseconds']
        else:
            check_quota(data, ['additional_replays_cpu_quota_delta', 'first_replay_cpu_quota_delta'])
            milliseconds = data['median_additional_replay_ms']
        rows[phase] = dict(median_ms=milliseconds, counts=data['counts'], setup_seconds=data['setup_seconds'], workspace_pool_bytes=data.get('persistent_workspace_pool_bytes'), material_recovery=data.get('material_recovery'))
        if 'quality' in data:
            rows[phase]['quality'] = data['quality']
        meshes[phase] = np.load(mesh)
        shutil.copyfile(report, out / f'containment-{phase}-{case}.json')
        if phase == 'after':
            shutil.copyfile(mesh, out / f'containment-after-{case}.npz')
            rows['independent'] = independent(meshes[phase], f'containment-after-{case}', vtu)
    a, b = meshes['before'], meshes['after']
    rows['points_bitwise_equal'] = bool(np.array_equal(a['points'].view(np.uint32), b['points'].view(np.uint32)))
    rows['roles_equal'] = bool(np.array_equal(a['roles'], b['roles']))
    rows['full_tet_sets_equal'] = bool(np.array_equal(canonical(a['full_tetrahedra']), canonical(b['full_tetrahedra'])))
    rows['material_tet_sets_equal'] = bool(np.array_equal(canonical(a['tetrahedra']), canonical(b['tetrahedra'])))
    rows['oriented_boundary_equal'] = oriented(a['boundary']) == oriented(b['boundary'])
    rows['time_change_percent'] = 100 * (rows['after']['median_ms'] / rows['before']['median_ms'] - 1)
    cases[case] = rows
    print(json.dumps(dict(case=case, **{k: v for k, v in rows.items() if k != 'independent'})), flush=True)

coarse_cases = {}
for case in ['cavity', 'disconnected', 'thin-hand']:
    coarse_dir = work / ('containment-coarse-' + case)
    coarse = json.loads((coarse_dir / 'result.json').read_text())
    shutil.copyfile(coarse_dir / 'result.json', out / f'containment-coarse-{case}.json')
    if coarse['valid']:
        assert coarse['error'] == 0
        check_quota(coarse, ['first_replay_cpu_quota_delta', 'additional_replays_cpu_quota_delta'])
        shutil.copyfile(coarse_dir / 'mesh.npz', out / f'containment-coarse-{case}.npz')
        coarse_checks = independent(np.load(coarse_dir / 'mesh.npz'), f'containment-coarse-{case}', coarse_dir / 'mesh.vtu')
        coarse_cases[case] = dict(result=coarse, independent=coarse_checks)
    else:
        assert all(v == 0 for v in coarse['counts'])
        coarse_cases[case] = dict(result=coarse, scope='Rejected output, not a successful mesh or comparable timing sample.')
assert coarse_cases['cavity']['result']['valid'] == 1

standalone = {}
for phase in ['before', 'after']:
    row = {}
    for n in [1000, 100000]:
        p = work / f'containment-{phase}-delaunay{n}.json'
        data = json.loads(p.read_text())
        assert data['error'] == 0
        assert all(v['throttled_periods'] == 0 for v in data['full']['cpu'].values())
        row[f'delaunay{n}_ms'] = data['full']['median_ms']
        shutil.copyfile(p, out / p.name)
    p = work / f'containment-{phase}-diagram100000.json'
    data = json.loads(p.read_text())
    assert not data['cpu_throttled']
    row['diagram100000_ms'] = data['median_seconds'] * 1000
    shutil.copyfile(p, out / p.name)
    standalone[phase] = row

isolation = {}
for p in sorted((tree / 'warpower').glob('*.py')):
    baseline = baseline_tree / 'warpower' / p.name
    isolation[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), same_as_main=baseline.exists() and p.read_bytes() == baseline.read_bytes())
assert {k for k, v in isolation.items() if not v['same_as_main']} == {'_containment.py', '_extraction.py', '_selective.py', '_quality_moves.py', '_vtm.py', '_mesher.py'}
assert (baseline_tree / 'README.md').read_bytes() == (tree / 'README.md').read_bytes()
for name, marker in [('isolated-memcheck.txt', '8 passed'), ('nested-memcheck.txt', '1 passed'), ('lifetime.txt', 'reference_quality_sum')]:
    text = (work / ('containment-' + name)).read_text()
    assert marker in text and 'ERROR SUMMARY: 0 errors' in text, name
summary = dict(
    scope='Complete fresh resident VTM solves, mainfa59aa8 versus runtimee85d086; three warm repeats, including domain/sizing refit and fresh sampling. Setup, upload, download, independent checks and export excluded. Material-recovery comparison against preceding Warpower, not CPU outperformance.',
    cases=cases, coarse_cases=coarse_cases, standalone=standalone,
    source_isolation=isolation, readme_unchanged=True,
    memory_check_scope='Eight isolated regressions, one small complete repaired graded quality trajectory, and the complete current 1,000-site disconnected four-round trajectory pass unfiltered padded instrumentation. The resident four-phase lifetime audit is also unfiltered. Earlier runtime1804b64 unfiltered/filtered runs were canceled before completion and remain incomplete; the current full-trajectory result does not retroactively validate them.',
    full_disconnected_memory_audit=json.loads((work / "containment-full-disconnected-audit.json").read_text()),
    native_checker=dict(source='benchmarks/cpu/check_delaunay.cpp', source_sha256=hashlib.sha256((tree / 'benchmarks/cpu/check_delaunay.cpp').read_bytes()).hexdigest(), binary_sha256=hashlib.sha256((work / 'check-delaunay').read_bytes()).hexdigest()),
)
(out / 'containment-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for name in ['suite.log', 'nested-test.log', 'full-disconnected-audit.json', 'full-disconnected-memcheck.txt', 'isolated-memcheck.txt', 'nested-memcheck.txt', 'lifetime.txt', 'device-audit.json', 'native-run.json', 'native.dot']:
    p = work / ('containment-' + name)
    shutil.copyfile(p, out / p.name)
manifest = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.glob('containment-*')) if 'provenance' not in p.name}
(out / 'containment-provenance.json').write_text(json.dumps(dict(before='fa59aa8', runtime_after='e85d086', audit_after='8c41ba3', artifacts=manifest), indent=2) + '\n')
print(json.dumps(standalone, indent=2), flush=True)
