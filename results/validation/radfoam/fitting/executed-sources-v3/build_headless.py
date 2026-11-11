"""Build native RadFoam training APIs without its optional OpenGL viewer."""

import json
import os
from pathlib import Path

from torch.utils.cpp_extension import load

UPSTREAM = Path("/home/horde/references/radfoam")
BUILD = Path(".work/radfoam-fitting-20261110/native").resolve()


def remove_function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[:start] + source[end:]


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    source = (UPSTREAM / "torch_bindings/pipeline_bindings.cpp").read_text()
    source = source.replace('#include "viewer/viewer.h"', "")
    source = remove_function(source, "void update_scene(")
    source = remove_function(source, "void run_with_viewer(")
    begin = source.index("    py::class_<Viewer,")
    end = source.index("\n}\n\n} // namespace radfoam_bindings", begin)
    source = source[:begin] + source[end:]
    generated = BUILD / "pipeline_bindings_headless.cpp"
    generated.write_text(source)
    binding = (UPSTREAM / "torch_bindings/torch_bindings.cpp").read_text()
    binding = binding.replace(
        "PYBIND11_MODULE(torch_bindings, module)", "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)"
    )
    entry = BUILD / "module.cpp"
    entry.write_text(binding)
    names = [
        "aabb_tree/aabb_tree.cu",
        "delaunay/delaunay.cu",
        "delaunay/sample_initial_tets.cu",
        "delaunay/growth_iteration.cu",
        "delaunay/delete_violations.cu",
        "delaunay/triangulation_ops.cu",
        "tracing/pipeline.cu",
        "utils/batch_fetcher.cpp",
    ]
    sources = [str(UPSTREAM / "src" / name) for name in names] + [
        str(generated),
        str(entry),
        str(UPSTREAM / "torch_bindings/triangulation_bindings.cpp"),
    ]
    includes = [
        "/usr/include/eigen3",
        str(UPSTREAM / "src"),
        str(UPSTREAM / "torch_bindings"),
        str(UPSTREAM / "external/include"),
        "/home/horde/references/atomic_queue/include",
    ]
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    record = {
        "sources": sources,
        "include_paths": includes,
        "scope": "Only viewer bindings removed; upstream renderer, AABB, native full/incremental Delaunay, farthest-neighbor, allocator and batch-fetcher code retained.",
    }
    (BUILD / "build-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    module = load(
        name="radfoam_headless",
        sources=sources,
        extra_include_paths=includes,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_ldflags=["-lcuda", "-ltbb"],
        build_directory=str(BUILD),
        verbose=True,
    )
    print("HEADLESS BUILD COMPLETE", module.__file__, flush=True)


if __name__ == "__main__":
    main()
