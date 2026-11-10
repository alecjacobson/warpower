"""Run upstream train.py with the reduced workload, retaining its training loop."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_upstream(source, build):
    sys.path.insert(0, str(build))
    sys.path.insert(0, str(source))
    sys.modules["radfoam"] = importlib.import_module("radfoam_headless")
    return importlib.import_module("train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--rays", type=int, default=65536)
    parser.add_argument("--final-sites", type=int, default=262144)
    parser.add_argument("--recipe", choices=["official", "adapter-settings"], default="official")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("RADFOAM_SOURCE_DIR", "/home/horde/references/radfoam")),
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=Path(os.environ.get("RADFOAM_BUILD_DIR", ".work/radfoam-fitting-20261110/native")),
    )
    args = parser.parse_args()
    output, data, source, build = [
        p.resolve() for p in (args.output, args.data, args.source, args.build)
    ]
    output.mkdir(parents=True, exist_ok=False)
    upstream = load_upstream(source, build)
    # Base workload overrides. The hardcoded million-ray training batch is
    # overridden at its constructor; the optional recipe is handled below.
    config_parser = upstream.configargparse.ArgParser()
    groups = [
        getattr(upstream, name)(config_parser)
        for name in ("PipelineParams", "ModelParams", "OptimizationParams", "DatasetParams")
    ]
    config_parser.add_argument("-c", "--config", is_config_file=True)
    config = config_parser.parse_args(["-c", str(source / "configs/mipnerf360_indoor.yaml")])
    overrides = dict(
        final_points=args.final_sites,
        downsample=[4],
        downsample_iterations=[0],
        data_path=str(data),
        experiment_name="fit",
    )
    vars(config).update(overrides)

    histories = {}
    state = {}
    original_handler = upstream.DataHandler

    class WorkloadDataHandler(original_handler):
        def __init__(self, dataset_args, rays_per_batch, device="cuda"):
            super().__init__(dataset_args, args.rays if rays_per_batch else 0, device=device)

    upstream.DataHandler = WorkloadDataHandler
    original_scene = upstream.RadFoamScene

    class AdapterSettingsScene(original_scene):
        """Match the small experiment's initialization and error-map budgets."""

        def initialize_from_pcd(self, points, points_colors):
            generator = torch.Generator().manual_seed(2026)
            ids = torch.randperm(len(points), generator=generator)[:8192]
            p = points[ids] + 0.01 * torch.randn((len(ids), 3), generator=generator)
            p = torch.cat([p, torch.randn((1024, 3), generator=generator) * 10])
            density = torch.cat([torch.zeros((len(ids), 1)), torch.full((1024, 1), -0.5)])
            initial = dict(
                primal_points=p,
                density=density,
                att_dc=torch.zeros((len(p), 3)),
                att_sh=torch.zeros((len(p), 45)),
            )
            state["initial_physical_sha256"] = hashlib.sha256(
                b"".join(initial[k].numpy().tobytes() for k in sorted(initial))
            ).hexdigest()
            self.triangulation = upstream.radfoam.Triangulation(p.to(self.device))
            perm = self.triangulation.permutation().long()
            self.primal_points = torch.nn.Parameter(p.to(self.device)[perm])
            self.density = torch.nn.Parameter(density.to(self.device)[perm])
            self.num_init_points = len(p)
            self.faces = None
            self.update_triangulation(rebuild=False)

        def collect_error_map(self, data_handler, white_bkg=True, downsample=2):
            ids = torch.linspace(0, len(data_handler.rays) - 1, 8).long()
            small = SimpleNamespace(rays=data_handler.rays[ids], rgbs=data_handler.rgbs[ids])
            return super().collect_error_map(small, white_bkg, downsample=4)

    def create_scene(*a, **kw):
        if args.recipe == "adapter-settings":
            torch.manual_seed(2026)
            np.random.seed(2026)
            model = AdapterSettingsScene(*a, **kw)
        else:
            model = original_scene(*a, **kw)
        state["model"] = model
        state["initial_points"] = len(model.primal_points)
        write(
            output / "initial-state.json",
            {
                "points": state["initial_points"],
                "physical_sha256": state.get("initial_physical_sha256"),
                "recipe": args.recipe,
            },
        )
        return model

    upstream.RadFoamScene = create_scene
    original_writer = upstream.SummaryWriter

    class RecordedWriter(original_writer):
        def add_scalar(self, tag, scalar_value, global_step=None, *a, **kw):
            result = super().add_scalar(tag, scalar_value, global_step, *a, **kw)
            histories.setdefault(tag, []).append([global_step, float(scalar_value)])
            if tag == "test/psnr":
                progress = dict(
                    step=global_step,
                    official_psnr=float(scalar_value),
                    sites=len(state["model"].primal_points),
                    elapsed_seconds=time.perf_counter() - state["loop_start"],
                )
                write(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            return result

    upstream.SummaryWriter = RecordedWriter
    original_trange = upstream.tqdm.trange

    class TimedProgress:
        def __init__(self, *a, **kw):
            self.progress = original_trange(*a, **kw)

        def __enter__(self):
            torch.cuda.synchronize()
            state["loop_start"] = time.perf_counter()
            return self.progress.__enter__()

        def __exit__(self, *a):
            torch.cuda.synchronize()
            state["loop_seconds"] = time.perf_counter() - state["loop_start"]
            return self.progress.__exit__(*a)

    # Replace only train.py's reference, not the tqdm module used by its loader.
    upstream.tqdm = SimpleNamespace(trange=TimedProgress)
    source_files = [
        source / "train.py",
        source / "configs/mipnerf360_indoor.yaml",
        source / "radfoam_model/scene.py",
        source / "radfoam_model/utils.py",
        source / "data_loader/__init__.py",
        source / "data_loader/colmap.py",
        Path(__file__).resolve(),
    ]
    report = {
        "complete": False,
        "scope": "Official native training loop at reduced resolution, ray and final-site budgets; asynchronous batches and every-100-step evaluation retained. Recipe selects official initialization/error maps or the small adapter's settings.",
        "recipe": args.recipe,
        "recipe_settings": {
            "seed": 2026 if args.recipe == "adapter-settings" else 42,
            "initialization": "8192 SfM samples without replacement + 1024 background sites; zero foreground raw density"
            if args.recipe == "adapter-settings"
            else "Official 90% SfM sample count with replacement + 5000 background sites; uniform foreground raw density",
            "error_map_views": 8 if args.recipe == "adapter-settings" else "all training views",
            "error_map_stride": 4 if args.recipe == "adapter-settings" else 2,
        },
        "configuration": vars(config),
        "rays_per_batch": args.rays,
        "config_overrides": overrides,
        "radfoam_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_status": subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True
        ),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "headless_binary_sha256": hashlib.sha256(
            (build / "radfoam_headless.so").read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "timing_scope": "Loop timer includes official evaluation every 100 steps and synchronous scalar logging; excludes loading, model setup, serialization and final image export. Not comparable to fit_scene.py's four training evaluations.",
        "unchanged": [
            "renderer/backward",
            "native geometry",
            "losses",
            "optimizer/schedules",
            "pruning/densification",
            "BatchFetcher",
            "held-out split",
            "PSNR function",
            "evaluation cadence",
            "serialization",
        ],
    }
    if args.recipe == "official":
        report["unchanged"].append("initialization and error-map selection")
    write(output / "report.json", report)
    (output / "executed_driver.py").write_bytes(Path(__file__).read_bytes())
    quota = cpu_quota_snapshot()
    start = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        os.chdir(output)
        upstream.train(config, *(group.extract(config) for group in groups))
        torch.cuda.synchronize()
        report.update(
            complete=True,
            total_seconds=time.perf_counter() - start,
            loop_seconds=state["loop_seconds"],
            initial_points=state["initial_points"],
            initial_physical_sha256=state.get("initial_physical_sha256"),
            final_points=len(state["model"].primal_points),
            metrics_text=(output / "output/fit/metrics.txt").read_text(),
            histories=histories,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except BaseException as error:
        report.update(
            error=repr(error), histories=histories, elapsed_seconds=time.perf_counter() - start
        )
        raise
    finally:
        os.chdir(old_cwd)
        report["cpu_quota"] = cpu_quota_delta(quota, cpu_quota_snapshot())
        write(output / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("complete", "loop_seconds", "final_points", "metrics_text")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
