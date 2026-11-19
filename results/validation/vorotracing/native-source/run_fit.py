"""Paired fitting in the original VoroTracing training loop with pinned inputs."""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numerics
import numpy as np
import torch
from upstream import load

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "warp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initialization",
        type=Path,
        default=Path("results/local/vorotracing-20261110/initialization/dense-points.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--sites", type=int, default=2000000)
    parser.add_argument("--rays", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--morton", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = load(args.backend, audit=True)
    import vorotracing.trainer as trainer_module
    from vorotracing.config.train_configs import _train_configs
    from vorotracing.datasets.datasets import Dataset

    config = copy.deepcopy(_train_configs["vorotracing-indoor"])
    config.dataset_config.data_path = str(args.data.resolve())
    config.dataset_config.init_points = "original"
    config.num_init_points = args.sites
    config.rays_per_batch = args.rays
    config.iterations = args.steps
    config.val_interval = args.eval_every
    config.iter2downsample = {0: args.downsample}
    config.viewer = False
    config.wandb = False
    config.save_checkpoint = not args.no_save
    config.out_dir = str(args.output)
    config.experiment_name = "fit"
    config.exit_on_completion = True

    # Use the original sampler with its own RNG, avoiding races with model and
    # metric initialization on the global Torch RNG. Same indices in both runs.
    generator = torch.Generator().manual_seed(2026)

    class SamplingTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def randint(self, *a, **kw):
            return torch.randint(*a, **kw, generator=generator)

    sampler = Dataset.sample_batch
    Dataset.sample_batch = FunctionType(
        sampler.__code__,
        {**sampler.__globals__, "torch": SamplingTorch()},
        sampler.__name__,
        sampler.__defaults__,
        sampler.__closure__,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    cloud = torch.load(args.initialization, weights_only=True)
    original_initialize = upstream.VoroTracing.from_pointcloud.__func__

    class Capture:
        def __init__(self, **kw):
            self.kw = kw

    initial = original_initialize(
        Capture,
        config=config.model_config,
        points=cloud["points"],
        points_colors=cloud["colors"],
        max_points=args.sites,
        num_random=5000,
        subsample_density_alpha=1.0,
        device=torch.device("cuda"),
    ).kw
    del cloud
    initializer_hash = hashlib.sha256()
    for name in ["points", "points_density", "points_diffuse"]:
        values = initial[name]
        if name == "points_diffuse":
            values = values[:, :3]
        initializer_hash.update(values.detach().cpu().contiguous().numpy().tobytes())

    def cached_initialize(cls, **kw):
        nonlocal initial
        model = cls(**initial)
        initial = None
        return model

    upstream.VoroTracing.from_pointcloud = classmethod(cached_initialize)

    state = {"step": -1, "evaluations": [], "geometry": []}
    original_trange = trainer_module.tqdm.trange

    class Progress:
        def __init__(self, *a, **kw):
            self.progress = original_trange(*a, **kw)

        def __iter__(self):
            torch.cuda.synchronize()
            state["loop_start"] = time.perf_counter()
            for step in self.progress:
                state["step"] = step
                yield step
            torch.cuda.synchronize()
            state["loop_seconds"] = time.perf_counter() - state["loop_start"]

        def set_postfix(self, *a, **kw):
            if not np.isfinite(float(kw["color_loss"])):
                raise RuntimeError("Nonfinite training color loss")
            return self.progress.set_postfix(*a, **kw)

    trainer_module.tqdm = SimpleNamespace(trange=Progress)

    original_validate = trainer_module.VoroTracingTrainer.validation
    original_metrics = trainer_module.VoroTracingTrainer._compute_metrics

    def recorded_metrics(self, rgb, target):
        result = original_metrics(self, rgb, target)
        state["per_view"].append(
            {
                "view": len(state["per_view"]),
                "pooled_psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                **{k: float(v) for k, v in result.items()},
            }
        )
        return result

    trainer_module.VoroTracingTrainer._compute_metrics = recorded_metrics

    def recorded_validate(self):
        state["per_view"] = []
        result = original_validate(self)
        row = {
            "step": state["step"],
            "seconds": time.perf_counter() - state["loop_start"],
            "metrics": {k: float(v) for k, v in result.items()},
            "numerics": numerics.snapshot(),
            "per_view": state["per_view"],
        }
        state["evaluations"].append(row)
        write(args.output / "progress.json", row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_view"}), flush=True)
        return result

    trainer_module.VoroTracingTrainer.validation = recorded_validate
    report = {
        "complete": False,
        "backend": args.backend,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "initial_base_sha256": initializer_hash.hexdigest(),
        "initial_hash_scope": "Original sampled positions, log densities and repeated diffuse RGB logit palette; specular texture is zero.",
        "source_commit": "75910484d6d46af25e180044ca6b9091204fe6fa",
        "training": "Original upstream loop; cached upstream dense initialization and isolated generator for the original uniform random-ray sampler; no external logging or viewer.",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    write(args.output / "report.json", report)
    quota = cpu_quota_snapshot()
    start = time.perf_counter()
    try:
        trainer = trainer_module.VoroTracingTrainer(config)
        trainer.out_dir = str(args.output / "model")
        model = trainer.model
        if args.morton:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radfoam-integration"))
            from spatial import morton_permutation

            model._permute_points(morton_permutation(model.primal_points))
            model.update_triangulation()
        update = model.update_triangulation

        def timed_update():
            torch.cuda.synchronize()
            t = time.perf_counter()
            update()
            torch.cuda.synchronize()
            state["geometry"].append({"step": state["step"], "seconds": time.perf_counter() - t})

        model.update_triangulation = timed_update
        # Compile the exact forward/backward specialization without an optimizer
        # step, before timing. This consumes one identical batch in both runs.
        rays = trainer.ray_batcher.next_batch()["rays"]
        report["warmup_batch_sha256"] = hashlib.sha256(rays.cpu().numpy().tobytes()).hexdigest()
        rgba, _, _, _, _, dist = model(rays[:2048])
        (rgba.sum() + dist.sum()).backward()
        model.optimizer.zero_grad(set_to_none=True)
        del rgba, dist, rays
        torch.cuda.synchronize()
        numerics.reset()
        trainer.train()
        report.update(
            complete=True,
            total_seconds=time.perf_counter() - start,
            loop_seconds=state["loop_seconds"],
            evaluations=state["evaluations"],
            geometry=state["geometry"],
            sites=len(model.primal_points),
            peak_torch_bytes=torch.cuda.max_memory_allocated(),
            native_extension_loaded="vorotracing.cuda._backend" in sys.modules,
            checkpoint=str(args.output / "model/model.pt") if not args.no_save else None,
            numerics=numerics.snapshot(),
        )
    except BaseException as error:
        report.update(
            error=repr(error), evaluations=state["evaluations"], geometry=state["geometry"]
        )
        raise
    finally:
        report["cpu_quota"] = cpu_quota_delta(quota, cpu_quota_snapshot())
        write(args.output / "report.json", report)
    print(
        json.dumps({k: report[k] for k in ["complete", "loop_seconds", "total_seconds", "sites"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
