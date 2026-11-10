// Headless feasibility probe: calls the unchanged upstream RadFoam CUDA pipeline.
// This C ABI is only a small test harness, not a replacement for its torch bindings.
#include "tracing/pipeline.h"
#include "utils/cuda_array.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// The normal binding supplies a PyTorch-backed allocator. For this synchronous
// probe, use cudaMalloc/cudaFree; rendering arithmetic is still upstream code.
namespace radfoam {
struct ProbeBuffer : OpaqueBuffer {
    void *pointer = nullptr;
    explicit ProbeBuffer(size_t bytes) { cuda_check(cudaMalloc(&pointer, bytes)); }
    ~ProbeBuffer() override { cudaFree(pointer); }
    void *data() override { return pointer; }
};
std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes) {
    return std::make_unique<ProbeBuffer>(bytes);
}
}

struct Buffers {
    uint32_t sites, edges, rays, quantiles;
    void *points, *attributes, *adjacency, *offsets, *ray_data, *start;
    void *depth_quantiles, *rgba, *depth, *depth_indices, *intersections, *contribution;
    void *rgba_grad, *depth_grad, *point_grad, *attribute_grad, *ray_grad;
};

static thread_local std::string error;
static auto &pipeline() {
    static auto p = radfoam::create_pipeline(0, radfoam::Float32);
    return p;
}

extern "C" const char *rf_error() { return error.c_str(); }

extern "C" int rf_forward(const Buffers *b) {
    try {
        pipeline()->trace_forward(radfoam::default_trace_settings(), b->sites,
            static_cast<const radfoam::Vec3f *>(b->points), b->attributes, b->edges,
            static_cast<const uint32_t *>(b->adjacency), static_cast<const uint32_t *>(b->offsets),
            b->rays, static_cast<const radfoam::Ray *>(b->ray_data),
            static_cast<const uint32_t *>(b->start), b->quantiles,
            static_cast<const float *>(b->depth_quantiles), b->rgba,
            static_cast<float *>(b->depth), static_cast<uint32_t *>(b->depth_indices),
            static_cast<uint32_t *>(b->intersections), b->contribution);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception &e) { error = e.what(); return 1; }
}

extern "C" int rf_backward(const Buffers *b) {
    try {
        pipeline()->trace_backward(radfoam::default_trace_settings(), b->sites,
            static_cast<const radfoam::Vec3f *>(b->points), b->attributes, b->edges,
            static_cast<const uint32_t *>(b->adjacency), static_cast<const uint32_t *>(b->offsets),
            b->rays, static_cast<const radfoam::Ray *>(b->ray_data),
            static_cast<const uint32_t *>(b->start), b->quantiles,
            static_cast<const float *>(b->depth_quantiles),
            static_cast<const uint32_t *>(b->depth_indices), b->rgba, b->rgba_grad,
            static_cast<const float *>(b->depth_grad), nullptr,
            static_cast<radfoam::Ray *>(b->ray_grad),
            static_cast<radfoam::Vec3f *>(b->point_grad), b->attribute_grad, nullptr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception &e) { error = e.what(); return 1; }
}
