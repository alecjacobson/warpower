"""Capture actual native geometry calls from a modest pinned RadFoam training run."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.replay.fixture import Fixture, digest
from benchmarks.replay.radfoam_capture import RadFoamRecorder, instrument_scene, numpy_copy


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/home/horde/references/radfoam"))
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--rays", type=int, default=8192)
    parser.add_argument("--initial-samples", type=int, default=8192)
    parser.add_argument("--background-sites", type=int, default=1024)
    parser.add_argument("--final-sites", type=int, default=32768)
    parser.add_argument("--densify-from", type=int, default=40)
    parser.add_argument("--densify-until", type=int, default=250)
    parser.add_argument("--freeze-points", type=int, default=400)
    parser.add_argument("--error-map-views", type=int, default=8)
    parser.add_argument("--error-map-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-capture-mb", type=int, default=150)
    args = parser.parse_args()
    if not (0 < args.initial_samples + args.background_sites < args.final_sites):
        parser.error("final site target must exceed the positive initialized site count")
    output, source, build, data = [
        p.resolve() for p in (args.output, args.source, args.build, args.data)
    ]
    output.mkdir(parents=True, exist_ok=False)

    import torch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    sys.path[:0] = [str(build), str(source)]
    native = importlib.import_module("radfoam_headless")
    sys.modules["radfoam"] = native
    upstream = importlib.import_module("train")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config_parser = upstream.configargparse.ArgParser()
    groups = [
        getattr(upstream, name)(config_parser)
        for name in ("PipelineParams", "ModelParams", "OptimizationParams", "DatasetParams")
    ]
    config_parser.add_argument("-c", "--config", is_config_file=True)
    config = config_parser.parse_args(["-c", str(source / "configs/mipnerf360_indoor.yaml")])
    overrides = dict(
        iterations=args.steps,
        final_points=args.final_sites,
        data_path=str(data),
        downsample=[4],
        downsample_iterations=[0],
        experiment_name="capture",
        densify_from=args.densify_from,
        densify_until=args.densify_until,
        freeze_points=args.freeze_points,
    )
    vars(config).update(overrides)
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "benchmarks/replay/radfoam_capture.py",
        Path(__file__).resolve().parents[2] / "benchmarks/replay/fixture.py",
        source / "train.py",
        source / "radfoam_model/scene.py",
        source / "radfoam_model/utils.py",
        source / "configs/mipnerf360_indoor.yaml",
        source / "data_loader/__init__.py",
        source / "data_loader/colmap.py",
    ]
    provenance = dict(
        producer="pinned upstream train.py with explicit reduced workload and producer capture hooks",
        radfoam_revision=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        upstream_status=subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True
        ),
        source_sha256={str(p): digest(p) for p in source_files},
        native_binary_sha256=digest(build / "radfoam_headless.so"),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        configuration=vars(config),
        config_overrides=overrides,
        initializer="SfM samples without replacement plus Gaussian background sites; exact recipe parameters recorded in arguments",
        identity="IDs assigned before initial native ordering; producer permutation/prune/append operations tracked and their coordinate row semantics checked exactly",
        input_scope="Every actual native constructor/rebuild input, including any retry perturbations; no synthetic motion",
        serialization="Model/optimizer checkpoints, PLY, TensorBoard and final images disabled; fitting and evaluations still execute",
        timings="Instrumented capture wall time only; synchronous input/output copies and compression preclude geometry or fitting speed claims",
    )
    recorder = RadFoamRecorder(
        output / "capture", provenance=provenance, max_bytes=args.max_capture_mb * 1024**2
    )
    state = dict(step=None, histories={}, losses=[], omitted_images=0, geometry_calls=0)
    report = dict(complete=False, provenance=provenance)
    write(output / "report.json", report)
    for i, path in enumerate(source_files):
        target = output / "executed-source" / f"{i:02d}-{path.name}"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())

    class CapturedTriangulation:
        def __init__(self, points):
            index, ids = recorder.input(points, incremental=False, constructor=True)
            try:
                self.inner = native.Triangulation(points)
            except BaseException as error:
                recorder.failure(index, error)
                raise
            recorder.success(index, ids, self.inner.tets(), self.inner.permutation())

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def rebuild(self, points, incremental=False):
            index, ids = recorder.input(points, incremental=incremental)
            try:
                permuted = self.inner.rebuild(points, incremental=incremental)
            except BaseException as error:
                recorder.failure(index, error)
                raise
            recorder.success(
                index, ids, self.inner.tets(), self.inner.permutation() if permuted else None
            )
            return permuted

    base = instrument_scene(upstream.RadFoamScene, recorder)

    class CapturedScene(base):
        def initialize_from_pcd(self, points, points_colors):
            generator = torch.Generator().manual_seed(args.seed)
            rows = torch.randperm(len(points), generator=generator)[: args.initial_samples]
            p = points[rows] + 0.01 * torch.randn((len(rows), 3), generator=generator)
            p = torch.cat([p, torch.randn((args.background_sites, 3), generator=generator) * 10])
            density = torch.cat(
                [torch.zeros((len(rows), 1)), torch.full((args.background_sites, 1), -0.5)]
            )
            recorder.initialize(len(p))
            recorder.begin_geometry(
                step=None, stage="initialization", rebuild=True, incremental=False
            )
            self.triangulation = CapturedTriangulation(p.to(self.device))
            permutation = self.triangulation.permutation().long()
            self.primal_points = torch.nn.Parameter(p.to(self.device)[permutation])
            self.density = torch.nn.Parameter(density.to(self.device)[permutation])
            recorder.permute(numpy_copy(permutation), source="initialize_from_pcd native ordering")
            self.num_init_points = len(p)
            self.faces = None
            self.update_triangulation(rebuild=False)

        def update_triangulation(self, rebuild=True, incremental=False):
            stage = (
                "initialization-attach"
                if state["step"] is None
                else "post-prune-densify"
                if recorder.pending_topology
                else "point-freeze"
                if not incremental and state["step"] == args.freeze_points
                else "optimizer-geometry-refresh"
                if incremental
                else "explicit-full-refresh"
            )
            recorder.begin_geometry(
                step=state["step"], stage=stage, rebuild=rebuild, incremental=incremental
            )
            state["geometry_calls"] += 1
            return super().update_triangulation(rebuild=rebuild, incremental=incremental)

        def declare_optimizer(self, *a, **kw):
            result = super().declare_optimizer(*a, **kw)
            original_step = self.optimizer.step

            def observed_step(*a, **kw):
                result = original_step(*a, **kw)
                recorder.optimizer_step()
                return result

            self.optimizer.step = observed_step
            return result

        def collect_error_map(self, data_handler, white_bkg=True, downsample=2):
            rows = torch.linspace(0, len(data_handler.rays) - 1, args.error_map_views).long()
            selected = SimpleNamespace(rays=data_handler.rays[rows], rgbs=data_handler.rgbs[rows])
            return super().collect_error_map(selected, white_bkg, downsample=args.error_map_stride)

        def save_pt(self, path):
            recorder.event("checkpoint-export-disabled", format="pt")

        def save_ply(self, path):
            recorder.event("checkpoint-export-disabled", format="ply")

    upstream.RadFoamScene = CapturedScene
    original_handler = upstream.DataHandler

    class CaptureDataHandler(original_handler):
        def __init__(self, dataset_args, rays_per_batch, device="cuda"):
            super().__init__(dataset_args, args.rays if rays_per_batch else 0, device=device)

    upstream.DataHandler = CaptureDataHandler
    original_trange = upstream.tqdm.trange

    class Progress:
        def __init__(self, *a, **kw):
            self.progress = original_trange(*a, disable=True, **kw)

        def __enter__(self):
            self.progress.__enter__()
            return self

        def __exit__(self, *a):
            return self.progress.__exit__(*a)

        def __iter__(self):
            for step in self.progress:
                state["step"] = int(step)
                recorder.context["step"] = int(step)
                yield step

        def set_postfix(self, **kw):
            state["losses"].append(dict(step=state["step"], **kw))
            if state["step"] % 50 == 0:
                print(
                    json.dumps(
                        dict(
                            step=state["step"],
                            sites=len(recorder.registry.ids),
                            native_inputs=len(recorder.writer.document["frames"]),
                            payload_bytes=recorder.payload_bytes,
                            **kw,
                        )
                    ),
                    flush=True,
                )

    upstream.tqdm = SimpleNamespace(trange=Progress)

    class ScalarWriter:
        def __init__(self, *a, **kw):
            pass

        def add_scalar(self, tag, value, step):
            state["histories"].setdefault(tag, []).append([int(step), float(value)])

        def close(self):
            pass

    class OmittedImage:
        def save(self, path):
            state["omitted_images"] += 1

    upstream.SummaryWriter = ScalarWriter
    upstream.Image = SimpleNamespace(fromarray=lambda _: OmittedImage())
    started = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        os.chdir(output)
        upstream.train(config, *(group.extract(config) for group in groups))
        torch.cuda.synchronize()
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        os.chdir(old_cwd)
        if recorder.writer.document["frames"]:
            recorder.finish()
            fixture = Fixture(recorder.writer.root)
            report.update(
                fixture=fixture.validate(),
                native_inputs=len(fixture),
                native_outputs=sum(r["success"] for r in recorder.outputs),
                native_failures=sum(not r["success"] for r in recorder.outputs),
                input_events=dict(Counter(r["event"] for r in fixture.records)),
                input_stages=dict(Counter(r["stage"] for r in fixture.records)),
                producer_events=dict(Counter(r["kind"] for r in recorder.events)),
                inserted_sites=sum(
                    r.get("count", 0) for r in recorder.events if r["kind"] == "insert"
                ),
                pruned_sites=sum(
                    r.get("count", 0) for r in recorder.events if r["kind"] == "prune"
                ),
                initial_sites=fixture.records[0]["active_count"],
                final_sites=len(recorder.registry.ids),
                optimizer_steps=recorder.optimizer_updates,
                capture_bytes=sum(
                    p.stat().st_size for p in recorder.root.rglob("*") if p.is_file()
                ),
            )
        report.update(
            instrumented_wall_seconds=time.perf_counter() - started,
            state=state,
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
        )
        write(output / "report.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "complete",
                    "native_inputs",
                    "native_outputs",
                    "inserted_sites",
                    "pruned_sites",
                    "capture_bytes",
                    "instrumented_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
