"""Complete reduced-budget Bonsai fitting example with stage timings and resume."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from integration import UPSTREAM, COLMAPDataset, RadFoamScene, Scene, checkpoint, initialize_runtime
from PIL import Image

from benchmarks.memory import MemorySampler
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["native", "warpower", "warpower-hints", "warpower-morton", "paragram"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", choices=["small", "radfoam"], default="small")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--initial-sites", type=int, help="SfM sample count for the small recipe")
    parser.add_argument("--final-sites", type=int)
    parser.add_argument("--rays", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--densify-factor", type=float, default=1.15)
    parser.add_argument("--densify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-trajectory", action="store_true")
    args = parser.parse_args()
    official_recipe = args.recipe == "radfoam"
    if official_recipe and args.initial_sites is not None:
        parser.error(
            "--initial-sites applies only to --recipe small; radfoam uses its SfM initializer"
        )
    defaults = {
        "seed": 42 if official_recipe else 2026,
        "steps": 20000 if official_recipe else 2000,
        "initial_sites": None if official_recipe else 8192,
        "final_sites": 262144 if official_recipe else 65536,
        "rays": 65536 if official_recipe else 16384,
        "eval_every": 5000 if official_recipe else 500,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.output.mkdir(parents=True, exist_ok=False)
    initialize_runtime()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    train = COLMAPDataset("data/radfoam-training/bonsai", "train", 4)
    test = COLMAPDataset("data/radfoam-training/bonsai", "test", 4)
    load_seconds = time.perf_counter() - started
    model_args = SimpleNamespace(
        sh_degree=3, init_points=131072, final_points=args.final_sites, activation_scale=1.0
    )
    initialization_start = time.perf_counter()
    if official_recipe:
        # Reuse the pinned initializer verbatim, including its native point
        # ordering and density assignment. Both geometry backends receive the
        # same physical state. This one-time native bootstrap is setup cost.
        bootstrap = RadFoamScene(
            model_args,
            points=train.points3D,
            points_colors=train.points3D_color,
            device=torch.device("cuda:0"),
        )
        initial = {
            name: getattr(bootstrap, name).detach().cpu().clone()
            for name in ["primal_points", "density", "att_dc", "att_sh"]
        }
        del bootstrap
    else:
        # Same deterministic physical initialization for all backends.
        generator = torch.Generator().manual_seed(args.seed)
        ids = torch.randperm(len(train.points3D), generator=generator)[: args.initial_sites]
        p = train.points3D[ids] + 0.01 * torch.randn((len(ids), 3), generator=generator)
        guards = torch.randn((1024, 3), generator=generator) * 10
        p = torch.cat([p, guards])
        initial = {
            "primal_points": p,
            "density": torch.cat([torch.zeros((len(ids), 1)), torch.full((1024, 1), -0.5)]),
            "att_dc": torch.zeros((len(p), 3)),
            "att_sh": torch.zeros((len(p), 45)),
        }
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_start
    if args.densify and args.final_sites <= len(initial["primal_points"]):
        parser.error(
            "--final-sites must exceed the initialized site count when densification is enabled"
        )
    initial_hash = hashlib.sha256(
        b"".join(initial[k].numpy().tobytes() for k in sorted(initial))
    ).hexdigest()
    recipe = {
        "name": args.recipe,
        "seed": args.seed,
        "initializer": "upstream RadFoamScene.initialize_from_pcd"
        if official_recipe
        else "small sampled SfM initialization",
        "native_initialization_bootstrap": official_recipe,
        "initial_sites": len(initial["primal_points"]),
        "initial_sha256": initial_hash,
        "error_map_views": len(train.all_rays) if official_recipe else 8,
        "error_map_stride": 2 if official_recipe else 4,
        "prune_and_densify": "inherited upstream RadFoamScene.prune_and_densify",
    }
    write(args.output / "recipe.json", recipe)
    for source in [
        Path(__file__),
        Path(__file__).with_name("integration.py"),
        Path(__file__).with_name("spatial.py"),
    ]:
        (args.output / f"executed-{source.name}").write_bytes(source.read_bytes())
    warmup = max(10, args.steps // 10)
    freeze = int(args.steps * 0.9)
    until = int(args.steps * 0.55)
    optimizer_args = SimpleNamespace(
        points_lr_init=2e-4,
        points_lr_final=5e-6,
        density_lr_init=1e-1,
        density_lr_final=1e-2,
        attributes_lr_init=5e-3,
        attributes_lr_final=5e-4,
        sh_factor=0.1,
        freeze_points=freeze,
    )
    model = Scene(initial, model_args, args.backend, optimizer_args, args.steps, warmup)
    train_rays = train.all_rays.reshape(-1, 6).cuda()
    train_rgb = train.all_rgbs.reshape(-1, 3).cuda()
    rng = torch.Generator(device="cuda").manual_seed(12345)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    stages = {
        k: 0.0
        for k in [
            "sampling",
            "forward",
            "backward",
            "optimizer",
            "geometry",
            "densification",
            "checkpoint_resume",
            "evaluation",
        ]
    }
    records = []
    evals = []
    events = []
    frames = []
    current_step = [-1]
    if args.save_trajectory:

        def save_frame(points, incremental):
            # Early, mid and late refreshes; keep the last preceding state too.
            if len(frames) < 40 or current_step[0] % 100 < 20:
                frames.append(
                    {
                        "step": current_step[0],
                        "incremental": incremental,
                        "points": points.cpu().numpy(),
                    }
                )

        model.frame_callback = save_frame

    def evaluate(step, save=False):
        t = time.perf_counter()
        values = []
        official_values = []
        with torch.no_grad():
            for view in range(len(test.all_rays)):
                rays = test.all_rays[view].reshape(-1, 6).cuda()
                target = test.all_rgbs[view].reshape(-1, 3).cuda()
                pred = []
                for chunk in rays.split(65536):
                    rgba = model(chunk)[0]
                    pred.append((rgba[:, :3] + 1 - rgba[:, 3:]).clamp(0, 1))
                pred = torch.cat(pred)
                mse = (pred - target).square().mean().item()
                values.append(-10 * np.log10(max(mse, 1e-12)))
                channel_mse = (pred - target).square().mean(dim=0)
                official_values.append(float((-10 * torch.log10(channel_mse)).mean()))
                if save and view in [0, len(test.all_rays) // 2]:
                    h, w = test.all_rgbs.shape[1:3]
                    display = torch.cat([target.reshape(h, w, 3), pred.reshape(h, w, 3)], dim=1)
                    Image.fromarray((display.cpu().numpy() * 255).astype(np.uint8)).save(
                        args.output / f"view-{view:02d}-target-render.png"
                    )
        torch.cuda.synchronize()
        stages["evaluation"] += time.perf_counter() - t
        item = {
            "step": step,
            "mean_test_psnr": float(np.mean(values)),
            "per_view_psnr": values,
            "mean_official_psnr": float(np.mean(official_values)),
            "per_view_official_psnr": official_values,
        }
        evals.append(item)
        print(json.dumps({"evaluation": item}), flush=True)
        return item

    # Compilation/warmup and first evaluation are outside training timing.
    warmup_start = time.perf_counter()
    dummy = model(train_rays[:1024])[0].sum()
    dummy.backward()
    model.optimizer.zero_grad(set_to_none=True)
    evaluate(-1)
    warmup_seconds = time.perf_counter() - warmup_start
    initial_evaluation_seconds = stages["evaluation"]
    stages["evaluation"] = 0.0
    period = 1
    since_update = 1
    since_densify = 0
    next_densify = 1
    memory = MemorySampler()
    memory.start()
    quota_before = cpu_quota_snapshot()
    fit_start = time.perf_counter()
    resume_verified = False
    for step in range(args.steps):
        current_step[0] = step
        torch.cuda.synchronize()
        t = time.perf_counter()
        ids = torch.randint(len(train_rays), (args.rays,), generator=rng, device="cuda")
        rays = train_rays[ids]
        target = train_rgb[ids]
        quantiles = (
            torch.rand((args.rays, 2), generator=rng, device="cuda")
            .sort(dim=1, descending=True)
            .values
        )
        torch.cuda.synchronize()
        now = time.perf_counter()
        stages["sampling"] += now - t
        t = now
        rgba, depth, _, _, _ = model(rays, depth_quantiles=quantiles)
        rgb = rgba[:, :3] + 1 - rgba[:, 3:]
        color_loss = torch.nn.functional.smooth_l1_loss(rgb, target)
        opacity_loss = (1 - rgba[:, 3]).square().mean()
        valid = (depth > 0).all(dim=1)
        depth_loss = ((depth[:, 0] - depth[:, 1]).abs() * valid).mean()
        loss = color_loss + opacity_loss + 1e-4 * min(2 * step / args.steps, 1) * depth_loss
        torch.cuda.synchronize()
        now = time.perf_counter()
        stages["forward"] += now - t
        t = now
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        now = time.perf_counter()
        stages["backward"] += now - t
        t = now
        model.optimizer.step()
        model.update_learning_rate(step)
        torch.cuda.synchronize()
        now = time.perf_counter()
        stages["optimizer"] += now - t
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite fitting loss")
        if since_update >= period:
            t = time.perf_counter()
            model.update_triangulation(incremental=True)
            stages["geometry"] += time.perf_counter() - t
            since_update = 0
            if period < 100:
                period += 2
        since_update += 1
        if step + 1 >= warmup:
            since_densify += 1
        if (
            args.densify
            and since_densify == next_densify
            and len(model.primal_points) < 0.9 * args.final_sites
        ):
            t = time.perf_counter()
            if official_recipe:
                handler = SimpleNamespace(rays=train.all_rays, rgbs=train.all_rgbs)
            else:
                view_ids = torch.linspace(0, len(train.all_rays) - 1, 8).long()
                handler = SimpleNamespace(
                    rays=train.all_rays[view_ids], rgbs=train.all_rgbs[view_ids]
                )
            error, contribution = model.collect_error_map(
                handler, True, downsample=recipe["error_map_stride"]
            )
            before = len(model.primal_points)
            model.prune_and_densify(error, contribution, args.densify_factor)
            stages["densification"] += time.perf_counter() - t
            t = time.perf_counter()
            model.update_triangulation(incremental=False)
            stages["geometry"] += time.perf_counter() - t
            events.append(
                {
                    "step": step,
                    "kind": "densify_prune",
                    "before": before,
                    "after": len(model.primal_points),
                    "error_map_views": recipe["error_map_views"],
                    "error_map_stride": recipe["error_map_stride"],
                }
            )
            period = 1
            since_densify = 0
            next_densify = max(
                100,
                int(
                    (args.densify_factor - 1)
                    * len(model.primal_points)
                    * (until - warmup)
                    / (args.final_sites - model.num_init_points)
                ),
            )
        if step == freeze:
            t = time.perf_counter()
            model.update_triangulation(incremental=False)
            stages["geometry"] += time.perf_counter() - t
            events.append({"step": step, "kind": "freeze_refresh"})
        if step == args.steps // 2:
            t = time.perf_counter()
            model.update_triangulation(incremental=False)
            with torch.no_grad():
                before_resume = model(rays[:1024])[0].clone()
            path = args.output / "resume.pt"
            checkpoint(
                model, model.optimizer, step, path, vars(args) | {"output": str(args.output)}
            )
            saved = torch.load(path, weights_only=False, map_location="cpu")
            old_geometry, old_callback = model.geometry_records, model.frame_callback
            old_audit = model.pipeline.counts
            # Fresh scene object; load Adam before a native full rebuild can
            # permute sites, so moments follow the same permutation as parameters.
            model = Scene(
                saved["parameters"],
                model_args,
                args.backend,
                optimizer_args,
                args.steps,
                warmup,
                defer_geometry=True,
            )
            model.optimizer.load_state_dict(saved["optimizer"])
            model.update_triangulation(incremental=False)
            permutation = (
                model.last_permutation
                if model.last_permutation is not None
                else torch.arange(len(model.primal_points), device="cuda")
            )
            for group, saved_group in zip(
                model.optimizer.param_groups, saved["optimizer"]["param_groups"]
            ):
                original = saved["optimizer"]["state"][saved_group["params"][0]]
                restored = model.optimizer.state[group["params"][0]]
                for key in ["exp_avg", "exp_avg_sq"]:
                    assert torch.equal(restored[key], original[key].cuda()[permutation])
            with torch.no_grad():
                resume_difference = float((model(rays[:1024])[0] - before_resume).abs().max())
            assert resume_difference < 1e-4, resume_difference
            model.geometry_records = old_geometry + model.geometry_records
            for key, value in old_audit.items():
                model.pipeline.counts[key] += value
            model.frame_callback = old_callback
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"])
            resume_verified = True
            stages["checkpoint_resume"] += time.perf_counter() - t
            events.append(
                {
                    "step": step,
                    "kind": "checkpoint_fresh_scene_resume",
                    "render_max_abs_difference": resume_difference,
                    "adam_moments_identical": True,
                }
            )
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "rgb_mse": float((rgb - target).square().mean().detach()),
            "points": len(model.primal_points),
        }
        records.append(row)
        if step % 50 == 0 or step == args.steps - 1:
            print(json.dumps(row), flush=True)
            write(
                args.output / "progress.json",
                {"last": row, "stages_seconds": stages, "events": events},
            )
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            evaluate(step, save=step == args.steps - 1)
    torch.cuda.synchronize()
    fit_wall = time.perf_counter() - fit_start
    quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
    memory_record = memory.finish()
    model.save_pt(str(args.output / "model.pt"))
    checkpoint(
        model,
        model.optimizer,
        args.steps,
        args.output / "training-state.pt",
        vars(args) | {"output": str(args.output)},
    )
    if frames:
        np.savez_compressed(
            args.output / "trajectory.npz",
            **{f"points_{i}": r["points"] for i, r in enumerate(frames)},
        )
        write(
            args.output / "trajectory.json",
            [{k: v for k, v in r.items() if k != "points"} for r in frames],
        )
    report = {
        "complete": True,
        "backend": args.backend,
        "config": vars(args) | {"output": str(args.output)},
        "scope": "Real Bonsai images_4 and standard every-eighth-view test split; reduced site/ray budget; configurable iteration count (the reported 20,000-step runs match upstream); explicit initialization/error-map recipe; synchronized stage profiling. Upstream RadFoam renderer, losses, optimizer schedules, pruning/densification and checkpoint format retained.",
        "recipe": recipe,
        "psnr_metric": "mean per-view pooled-RGB PSNR; upstream channel-averaged PSNR reported separately",
        "initialization_seconds": initialization_seconds,
        "initial_sha256": initial_hash,
        "load_seconds": load_seconds,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "initial_evaluation_seconds": initial_evaluation_seconds,
        "fit_wall_seconds": fit_wall,
        "stages_seconds": stages,
        "records": records,
        "evaluations": evals,
        "events": events,
        "geometry": model.geometry_records,
        "raw_render_gradient_checks": model.pipeline.counts,
        "memory": memory_record,
        "cpu_quota": quota,
        "cpu_throttled": any(x["throttled_periods"] for x in quota.values()),
        "checkpoint_continuation": resume_verified,
        "final_points": len(model.primal_points),
        "gpu": torch.cuda.get_device_name(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("integration.py"),
                Path(__file__).with_name("spatial.py"),
                UPSTREAM / "radfoam_model/scene.py",
                UPSTREAM / "configs/mipnerf360_indoor.yaml",
            ]
        },
        "warpower_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "radfoam_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "complete": True,
                "backend": args.backend,
                "seconds": fit_wall,
                "final_test_psnr": evals[-1]["mean_test_psnr"],
                "final_points": len(model.primal_points),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
