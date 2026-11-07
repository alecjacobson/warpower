#define CUB_STDERR
#include <cub/cub.cuh>
#include "delaunay_solver.h"
#include "geometry.h"

template <typename T>
__global__ void GPUExactPredictInit(T* d_predConsts)
{
    T half;
    T check, lastcheck;
    int every_other;
    T Epsilon, Splitter;

    every_other = 1;
    half = 0.5;
    Epsilon = 1.0;
    Splitter = 1.0;
    check = 1.0;
    do {
        lastcheck = check;
        Epsilon = Epsilon * half;
        if (every_other) {
            Splitter *= 2.0;
        }
        every_other = !every_other;
        check = 1.0 + Epsilon;
    } while ((check != 1.0) && (check != lastcheck));
    Splitter += 1.0;

    d_predConsts[epsilon] = Epsilon;
    d_predConsts[resulterrbound] = (3.0 + 8.0 * Epsilon) * Epsilon;
    d_predConsts[ccwerrboundA] = (3.0 + 16.0 * Epsilon) * Epsilon;
    d_predConsts[ccwerrboundB] = (2.0 + 12.0 * Epsilon) * Epsilon;
    d_predConsts[ccwerrboundC] = (9.0 + 64.0 * Epsilon) * Epsilon * Epsilon;
    d_predConsts[isperrboundA] = (16.0 + 224.0 * Epsilon) * Epsilon;
    d_predConsts[isperrboundB] = (5.0 + 72.0 * Epsilon) * Epsilon;
    d_predConsts[isperrboundC] = (71.0 + 1408.0 * Epsilon) * Epsilon * Epsilon;
    d_predConsts[o3derrboundA] = (7.0 + 56.0 * Epsilon) * Epsilon;
    d_predConsts[o3derrboundB] = (3.0 + 28.0 * Epsilon) * Epsilon;
    d_predConsts[o3derrboundC] = (26.0 + 288.0 * Epsilon) * Epsilon * Epsilon;
    d_predConsts[o3derrboundAlifted] = (11.0 + 112.0 * Epsilon) * Epsilon;
}



__global__ void cast(REAL3* input, float3* output, int n) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed < n) {
        output[seed] = { (float)input[seed].x, (float)input[seed].y, (float)input[seed].z };
    }
}

__global__ void find_boundary(int nv, DTStruct* output, uchar* boundary, uchar* stat, uchar* exact_result) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nv) return;
    if (stat[seed] != success) return;

    if (exact_result) {
        if (exact_result[seed] == 2) {
            for (int i = 0; i < output[seed].nv[0]; i++) {
               /* if (seed == 487872)
                    printf("%d\n", exact_result[decode_index(output[seed].vertex_data[i])]);*/
                if (exact_result[decode_index(output[seed].vertex_data[i])] == 0)
                    boundary[seed] = 1;
            }
            for (int i = 0; i < output[seed].nv[1]; i++) {
              /*  if (seed == 487872)
                    printf("%d\n", exact_result[decode_index(output[seed].vertex_data[MAX_P + i])]);*/
                if (exact_result[decode_index(output[seed].vertex_data[MAX_P + i])] == 0)
                    boundary[seed] = 1;
            }
        }
        else {
            if (exact_result[seed] == 1) {
                boundary[seed] = 1;
            }
            else {
                boundary[seed] = 0;
                for (int i = 0; i < output[seed].nv[0]; i++) {
                    int id = decode_index(output[seed].vertex_data[i]);
                    if (exact_result[id] == 2)
                        boundary[id] = 1;
                }
                for (int i = 0; i < output[seed].nv[1]; i++) {                    
                    int id = decode_index(output[seed].vertex_data[MAX_P + i]);
                    if (exact_result[id] == 2)
                        boundary[id] = 1;
                }
            }
        }
    }
    else {
        boundary[seed] = 2;
        for (int i = 0; i < output[seed].nv[0]; i++) {
            if (decode_index(output[seed].vertex_data[i]) >= nv)
                boundary[seed] = 1;
        }
        for (int i = 0; i < output[seed].nv[1]; i++) {
            if (decode_index(output[seed].vertex_data[MAX_P + i]) >= nv)
                boundary[seed] = 1;
        }
    }

}



__global__ void find_boundary_post(int nv, int npost, int* unsuccessIds, DTStructPost* output, uchar* boundary, uchar* exact_result) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= npost) return;
    DTStructPost* dt = &(output[seed]);
    seed = unsuccessIds[seed];
    if (exact_result) {
        if (exact_result[seed] == 2) {
            for (int i = 0; i < dt->nt; i++) {
                uchar4 tr = dt->tr_data[i];
                if (exact_result[dt->index_data[tr.x]] == 0 || exact_result[dt->index_data[tr.y]] == 0 || exact_result[dt->index_data[tr.z]] == 0)
                    boundary[seed] = 1;
            }
        }
        else {
            if (exact_result[seed] == 1) {
                boundary[seed] = 1;
            }
            else {
                boundary[seed] = 0;
                for (int i = 0; i < dt->nt; i++) {
                    uchar4 tr = dt->tr_data[i];
                    if (exact_result[dt->index_data[tr.x]] == 2)
                        boundary[dt->index_data[tr.x]] = 1;
                    if (exact_result[dt->index_data[tr.y]] == 2)
                        boundary[dt->index_data[tr.x]] = 1;
                    if (exact_result[dt->index_data[tr.z]] == 2)
                        boundary[dt->index_data[tr.x]] = 1;
                }
            }
        }
    }
    else {
        boundary[seed] = 2;
        for (int i = 0; i < dt->nt; i++) {
            uchar4 tr = dt->tr_data[i];
            if (dt->index_data[tr.x] >= nv || dt->index_data[tr.y] >= nv || dt->index_data[tr.z] >= nv)
                boundary[seed] = 1;
        }
    }
}

__global__ void get_nt(int nv, int startseed, DTStruct* output, int* ntet, uchar* stat, uchar* exact_result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = tid >> 1;                // tid / 2
    int side = tid & 1;                 // tid % 2

    if (seed + startseed >= nv) return;
    if (stat[seed] != success) return;

    DTStruct& dt = output[seed];

    int num = 0;

    int idx_base = side * MAX_P;
    int last1 = dt.vertex_data[idx_base];
    int last2 = decode_index(dt.vertex_data[idx_base + 1]);

    bool use_exact = (exact_result != nullptr) &&
        (exact_result[seed + startseed] != 2);

    bool in1 = false, in2 = false;
    if (use_exact) {
        in1 = (exact_result[last1] == 2);
        in2 = (exact_result[last2] == 2);
    }

    int nv_local = dt.nv[side];

    for (int i = 2; i < nv_local; ++i) {
        int v = dt.vertex_data[idx_base + i];

        if (v < 0) {  
            int last3 = decode_index(v);

            if (last1 < nv && last2 < nv && last3 < nv) {
                if (!use_exact) {
                    num++;
                }
                else {
                    bool in3 = (exact_result[last3] == 2);
                    if (in1 || in2 || in3)
                        num++;
                    in2 = in3;
                }
            }

            last2 = last3;
        }
        else {
            last1 = v;
            if (use_exact)
                in1 = (exact_result[last1] == 2);
        }
    }

    ntet[tid] = num;
}

__global__ void post_init(int npts, int npost, DTStructPost* output) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= npost) return;
    output[seed].nv = 4;
    output[seed].nt = 4;
    output[seed].index_data[0] = npts;
    output[seed].index_data[1] = -(npts + 1);
    output[seed].index_data[2] = npts + 2;
    output[seed].index_data[3] = npts + 6;
    output[seed].tr_data[0] = { 0, 2, 3, 1 };
    output[seed].tr_data[1] = { 0, 1, 2, 0 };
    output[seed].tr_data[2] = { 1, 0, 3, 0 };
    output[seed].tr_data[3] = { 2, 1, 3, 0 };
}

__global__ void get_inverse(int npost, int* unsuccessIds, int* inverseIds) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= npost) return;
    inverseIds[unsuccessIds[seed]] = seed;
}

__global__ void make_tet_list(int nv, int startseed, DTStruct* output, int* accnt, int4* tet, size_t* totaltet, uchar* stat, uchar* exact_result) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed/2 + startseed >= nv) return;
    if (stat[seed / 2] != success) return;
   

    size_t start = totaltet[0] + accnt[seed];
    int lastid1 = output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P];
    int lastid2 = output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P + 1];
    lastid2 = decode_index(lastid2);
    int globalidx = start;
    if (exact_result && (exact_result[seed / 2 + startseed] != 2)) {
        bool inboundary1 = (exact_result[lastid1 ] == 2);
        bool inboundary2 = (exact_result[lastid2 ] == 2);
        for (int i = 2; i < output[seed / 2].nv[threadIdx.x % 2]; i++) {
            int signednow = output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P + i];
            if (signednow < 0) {
                int lastid3 = decode_index(signednow);
                bool inboundary3 = (exact_result[lastid3 ] == 2);
                if (lastid3 < nv && lastid1 < nv && lastid2 < nv) {
                    if (inboundary1 || inboundary2 || inboundary3) {
                        tet[globalidx] = make_int4(lastid1, lastid2, lastid3, seed / 2 + startseed);
                        globalidx++;
                    }
                }
                lastid2 = lastid3;
                inboundary2 = inboundary3;
            }
            else {
                lastid1 = signednow;
                inboundary1 = (exact_result[lastid1] == 2);
            }

        }
    }
    else {
        for (int i = 2; i < output[seed / 2].nv[threadIdx.x % 2]; i++) {
            int signednow = output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P + i];
            if (signednow < 0) {
                int lastid3 = decode_index(signednow);
                if (lastid3 < nv && lastid1 < nv && lastid2 < nv) {
                    tet[globalidx] = make_int4(lastid1, lastid2, decode_index(signednow), seed / 2 + startseed);
                    globalidx++;
                }
                lastid2 = lastid3;
            }
            else {
                lastid1 = signednow;
            }

        }
    }
    
       
}

__global__ void gather_information(int nv, int startseed, int nv_process, DTStruct* output, uchar* stat, DTStructPost* outputpost, int* inverseIds) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed / 2 + startseed >= nv) return;
    if (stat[seed / 2] != success) {
        for (int i = 0; i < output[seed / 2].nv[threadIdx.x % 2]; i++) {
            int now = decode_index(output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P + i]);
            if (now < nv) {
                int id = inverseIds[seed / 2];
                int old_nv = atomicAdd(&outputpost[id].nv, 1);
                if (old_nv < MAX_POST_P)
                    outputpost[id].index_data[old_nv] = now;
            }
        }
        return;
    }

    for (int i = 0; i < output[seed / 2].nv[threadIdx.x % 2]; i++) {
        int now = decode_index(output[seed / 2].vertex_data[(threadIdx.x % 2) * MAX_P + i]);
        if (now < nv && now >= startseed && now < startseed + nv_process) {
            if (stat[now - startseed] != success) {
                int id = inverseIds[now - startseed];
                int old_nv = atomicAdd(&outputpost[id].nv, 1);
                if (old_nv < MAX_POST_P)
                    outputpost[id].index_data[old_nv] = seed / 2 + startseed;
            }
        }
    }
}

__global__ void post_remove(int npost, DTStructPost* output) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= npost) return;
    if (output[seed].nv > MAX_POST_P)
        output[seed].nv = MAX_POST_P;
    for (int i = 4; i < output[seed].nv; i++) {
        for (int j = i + 1; j < output[seed].nv; ) {
            if (output[seed].index_data[j] == output[seed].index_data[i]) {
                output[seed].index_data[j] = output[seed].index_data[output[seed].nv - 1];
                output[seed].nv--;  
            }
            else {
                j++;
            }
        }
    }
}


__global__ void get_total_tet(int nv, int* accnt, int* ntet, size_t* totaltet) {
    totaltet[0] = totaltet[0] + accnt[nv-1] + ntet[nv-1];
}


void DelaunaySolver::addinfpoints() {
    std::vector<REAL3> infpoints(7);
    REAL infdist = 1;
    infpoints[0] = { -infdist, -infdist, -infdist };
    infpoints[1] = { infdist, -infdist, -infdist };
    infpoints[2] = { 0, infdist, -infdist };
    infpoints[3] = { -infdist, infdist, infdist };
    infpoints[4] = { infdist, infdist, infdist };
    infpoints[5] = { 0, -infdist, infdist };


    REAL3 infnormal[8];
    infnormal[0] = inf_normal2(infpoints[0], infpoints[1], infpoints[2]);
    infnormal[1] = inf_normal(infpoints[1], infpoints[2]);
    infnormal[2] = inf_normal(infpoints[2], infpoints[0]);
    infnormal[3] = inf_normal(infpoints[0], infpoints[1]);

    infnormal[4] = inf_normal2(infpoints[3], infpoints[4], infpoints[5]);
    infnormal[5] = inf_normal(infpoints[4], infpoints[5]);
    infnormal[6] = inf_normal(infpoints[5], infpoints[3]);
    infnormal[7] = inf_normal(infpoints[3], infpoints[4]);

    cuda_check(cudaMemcpyToSymbol(infvector, infnormal, 8 * sizeof(REAL3)));
    infdist = 1.e8f;
    infpoints[0] = { -infdist, -infdist, -infdist };
    infpoints[1] = { infdist, -infdist, -infdist };
    infpoints[2] = { 0, infdist, -infdist };
    infpoints[3] = { -infdist, infdist, infdist };
    infpoints[4] = { infdist, infdist, infdist };
    infpoints[5] = { 0, -infdist, infdist };
    infpoints[6] = { 0, 0, infdist };
    cuda_check(cudaMemcpy(sortedpts.data() + nv, infpoints.data(), 7 * sizeof(REAL3), cudaMemcpyHostToDevice));
}

void DelaunaySolver::postinit(int startseed) {
    unsuccessIds.zero_fill();
    nunsuccess.zero_fill();
    void* d_temp_storage = NULL; size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, thrust::make_counting_iterator(0), stat.data(), unsuccessIds.data(), nunsuccess.data(), nv_process);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, thrust::make_counting_iterator(0), stat.data(), unsuccessIds.data(), nunsuccess.data(), nv_process);
    npost = nunsuccess.at(0);
    outputpost = cuVector< DTStructPost>(npost);
    post_init << < divide(npost, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (nv, npost, outputpost.data());
    cuVector<int> inverseIds(nv_process);
    get_inverse << < divide(npost, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (npost, unsuccessIds.data(), inverseIds.data());

    
    gather_information << < divide(2 * nv_process, 32), 32 >> > (nv, startseed, nv_process, output.data(), stat.data(), outputpost.data(), inverseIds.data());
    
    post_remove << < divide(npost, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (npost, outputpost.data());
    cudaFree(d_temp_storage);

}


void DelaunaySolver::compressoutput(int startseed) {
   
    void* d_temp_storage = NULL; size_t temp_storage_bytes = 0; 

    ntet.zero_fill();
    get_nt << < divide(2 * nv_process, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (nv, startseed, output.data(), ntet.data(), stat.data(), exact_result.data());
    d_temp_storage = NULL; temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ntet.data(), accnt.data(), 2 * nv_process);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ntet.data(), accnt.data(), 2 * nv_process);   
    make_tet_list << < divide(2 * nv_process, 32), 32 >> > (nv, startseed, output.data(), accnt.data(), tet.data(), totaltet.data(),stat.data(), exact_result.data());
    get_total_tet << <1, 1 >> > (2 * nv_process, accnt.data(), ntet.data(), totaltet.data());
    cudaFree(d_temp_storage);
     
}

__global__ void get_nt_post(int npost, int startseed, int* unsuccessIds, DTStructPost* output, int* ntet, uchar* stat, uchar* exact_result) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed  >= npost) return;
    if (stat[unsuccessIds[seed]] != success) return;
    if (exact_result) {
        bool inboundary = (exact_result[unsuccessIds[seed] + startseed] == 2);
        int i = 0;
        while (i < output[seed].nt) {
            uchar4 tr = output[seed].tr_data[i];
            bool inboundary1 = (exact_result[output[seed].index_data[tr.x]] == 2);
            bool inboundary2 = (exact_result[output[seed].index_data[tr.y]] == 2);
            bool inboundary3 = (exact_result[output[seed].index_data[tr.z]] == 2);
            if (!(inboundary || inboundary1 || inboundary2 || inboundary3)) {
                output[seed].nt--;
                output[seed].tr_data[i] = output[seed].tr_data[output[seed].nt];
            }
            else
                i++;

        }
    }

    ntet[seed] = output[seed].nt;
}

__global__ void make_tet_list_post(int npost, int startseed, int* unsuccessIds, DTStructPost* output, int* accnt, int4* tet, size_t* totaltet, uchar* stat) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= npost) return;
    if (stat[unsuccessIds[seed]] != success) return;
    DTStructPost* dt = &(output[seed]);
    size_t start = totaltet[0] + accnt[seed];
    for (int i = 0; i < dt->nt; i++) {
        uchar4 tr = dt->tr_data[i];
        tet[start + i] = make_int4(dt->index_data[tr.x], dt->index_data[tr.y], dt->index_data[tr.z], unsuccessIds[seed] + startseed);
    }
}

void DelaunaySolver::compressoutputpost(int startseed) {
    ntet.zero_fill();
    get_nt_post << < divide(npost, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (npost, startseed, unsuccessIds.data(), outputpost.data(), ntet.data(), stat.data(), exact_result.data());

    void* d_temp_storage = NULL; size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ntet.data(), accnt.data(), npost);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ntet.data(), accnt.data(), npost);
    cudaFree(d_temp_storage);
    make_tet_list_post << < divide(npost, 32), 32 >> > (npost, startseed, unsuccessIds.data(), outputpost.data(), accnt.data(), tet.data(), totaltet.data(), stat.data());
    get_total_tet << <1, 1 >> > (npost, accnt.data(), ntet.data(), totaltet.data());

}

void DelaunaySolver::init(const std::vector<REAL3>& h_pts, std::vector<uchar>& boundary, int n) {
    exact_result = boundary;
    nv = h_pts.size();
    if (n == 0)
        n = nv;
    nv_process = n;
   /* M = (int)round(pow(nv / 0.7f, 1.0f / 3.0));*/
    M = (int)round(pow(nv / 4.0f, 1.0f / 3.0));
    pts = h_pts;
    sortedpts = cuVector<REAL3>(nv + 7);
    stat = cuVector<uchar>(nv_process);
    stat.zero_fill();
    cellStartIds = cuVector<int>(M * M * M + 1);
    cellStartIds.zero_fill();
    output = cuVector<DTStruct>( nv_process);
    unsuccessIds = cuVector<int>( nv_process );
    nunsuccess = cuVector<int>(1);
    tet = cuVector<int4>(7 * nv);
    totaltet = cuVector<size_t>(1);
    totaltet.zero_fill();
    ntet = cuVector<int>(2 * nv_process);
    accnt = cuVector<int>(2 * nv_process);
#ifdef OUTPUT_INFORMATION   
    time = cuVector<clock_t>(2 * nv_process);
    ncycles = cuVector<int>(2 * nv_process);
#endif
#ifndef SORTING_POINTS
    permutation = cuVector<int>(2 * nv_process);
#endif
    addinfpoints();
}

void DelaunaySolver::cvt(int Iter, const char* outfile) {

    cuVector<float> predConsts(numpredConsts);
    GPUExactPredictInit<float> << <1, 1 >> > (predConsts.data());
    cuVector<REAL> predConstsR(numpredConsts);
    GPUExactPredictInit<REAL> << <1, 1 >> > (predConstsR.data());

    cuVector<REAL3> newpts(nv);//vol*bary
    cuVector<REAL> vol(nv);  
    cuVector<REAL> distance(nv);

    PostProcessingPost_kernel << < 1, 1 >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());

    gpu_timer dt_time("Delaunay Triangulation");
    gpu_timer dt_time2("Delaunay Triangulation Post");
    gpu_timer moving_time("Moving Points");
    gpu_timer moving_time2("Moving Points");
    float dt = 0;
    float moving = 0;

    for (int i = 0; i < Iter; i++) {
        newpts.zero_fill();
        vol.zero_fill();
        cuVector<uchar> boundary(std::vector<uchar>(nv, 2));
        dt_time.start();
        buildGrids_GPU(pts, sortedpts, cellStartIds, M, exact_result, permutation);
        /*buildGrids_CPU2GPU(pts, sortedpts, cellStartIds, M, permutation,exact_result);*/

        DT_kernel << < divide(2 * nv, BLOCK_SIZE_MAIN), BLOCK_SIZE_MAIN   >> > (sortedpts.data(), nv, 0, predConsts.data(), stat.data(), output.data(), cellStartIds.data(), M, exact_result.data(), permutation.data(), ncycles.data(), time.data());
        find_boundary << <divide(nv, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (nv, output.data(), boundary.data(), stat.data(), exact_result.data());
        dt_time.stop();
        moving_time2.start();
        Voronoi_center << < divide(2 * nv, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (sortedpts.data(), nv, output.data(), stat.data(), newpts.data(), vol.data(), exact_result.data(), boundary.data());
        moving_time2.stop();
        dt_time2.start();
        postinit(0);
        PostProcessingMain_kernel << < divide(npost, BLOCK_SIZE_POST), BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), permutation.data(), ncycles.data(), time.data());
        PostProcessingPost_kernel << < npost, BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());
        dt_time2.stop();
       
        moving_time.start();
        
        find_boundary_post << < divide(npost, BLOCK_SIZE_POST), BLOCK_SIZE_POST >> > (nv, npost, unsuccessIds.data(), outputpost.data(), boundary.data(), exact_result.data());
        Voronoi_center_post << < divide(npost, BLOCK_SIZE_POST), BLOCK_SIZE_POST >> > (sortedpts.data(), npost, unsuccessIds.data(), outputpost.data(), newpts.data(), vol.data(), exact_result.data(), boundary.data());

        update_pts << <divide(nv, THREADS_PER_BLOCK), THREADS_PER_BLOCK >> > (pts.data(), sortedpts.data(), nv, newpts.data(), vol.data(), boundary.data(), distance.data());
        moving_time.stop();
        cudaDeviceSynchronize();

        dt_time.report();
        dt_time2.report();
        dt = dt + dt_time.time + dt_time2.time;
        moving_time.report();
        moving_time2.report();
        moving = moving + moving_time.time + moving_time2.time;
    }
     float total = dt+moving;
     std::cout << "dt total time:" << dt << " msec" << std::endl;
     std::cout << "moving points time:" << moving << " msec" << std::endl;
     std::cout << "cvt total time:" <<total << " msec" << std::endl;

    cuVector<REAL> sumdistance(1);
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, distance.data(), sumdistance.data(), nv);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, distance.data(), sumdistance.data(), nv);
    cudaFree(d_temp_storage);

    std::cout << "distance average:" << sumdistance.at(0) / nv << std::endl;

   /* compute_DT_GPU(outfile);*/

}

void DelaunaySolver::compute_DT_GPU_large(const char* outfile) {
    cuVector<float> predConsts(numpredConsts);
    GPUExactPredictInit<float> << <1, 1 >> > (predConsts.data());
    cuVector<REAL> predConstsR(numpredConsts);
    GPUExactPredictInit<REAL> << <1, 1 >> > (predConstsR.data());

    gpu_timer Buildgrids_time("Build grids");
    buildGrids_GPU(pts, sortedpts, cellStartIds, M, exact_result, permutation);
    pts.clear();
    /*buildGrids_CPU2GPU(pts, sortedpts, cellStartIds, M, permutation);*/
    Buildgrids_time.stop();
#ifdef USINGDOUBLE
    cuVector<float3> sortedpts2(nv + 4);
    cast << <(nv - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (sortedpts.data(), sortedpts2.data(), nv + 4);
#endif
    PostProcessingPost_kernel << < 1, BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());
    exact_result = std::vector<uchar>(nv_process, 2);
    gpu_timer dt_time("Delaunay Triangulation");
    std::cout << "Delaunay Triangulation Batches of " << nv_process << " points at a time" << std::endl;
    for (int i = 0; i < nv; i = i + nv_process) {
        int idxend = (i + nv_process < nv) ? (i + nv_process) : nv;
        std::cout << "processing points:" << i << "-" << idxend << std::endl;
        DT_kernel << < divide(2 * nv_process, BLOCK_SIZE_MAIN), BLOCK_SIZE_MAIN  >> > (sortedpts.data(), nv, i, predConsts.data(), stat.data(), output.data(), cellStartIds.data(), M, exact_result.data(), permutation.data(), ncycles.data(), time.data()); 
        compressoutput(i);
        postinit(i);
        PostProcessingMain_kernel << < divide(npost, BLOCK_SIZE_POST), BLOCK_SIZE_POST >> > (sortedpts.data(), nv, i, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), permutation.data(), ncycles.data(), time.data());
        PostProcessingPost_kernel << < npost, BLOCK_SIZE_POST >> > (sortedpts.data(), nv, i, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());
        compressoutputpost(i);
    }
    dt_time.stop();

    cudaDeviceSynchronize();
    Buildgrids_time.report();
    dt_time.report();
    std::cout << "Delaunay Triangulation on GPU total time:" << (Buildgrids_time.time + dt_time.time) / 1000 << " s" << std::endl;
    std::cout << "Total tet:" << totaltet.at(0) << std::endl;
    /*copywrite(outfile);*/

}

void DelaunaySolver::compute_DT_GPU(const char* outfile) {
    cuVector<float> predConsts(numpredConsts);
    GPUExactPredictInit<float> << <1, 1 >> > (predConsts.data());
    cuVector<REAL> predConstsR(numpredConsts);
    GPUExactPredictInit<REAL> << <1, 1 >> > (predConstsR.data());

    cuVector<int> unsuccess(1);
    unsuccess.zero_fill();

    gpu_timer Buildgrids_time("Build grids");
    buildGrids_GPU(pts, sortedpts, cellStartIds, M, exact_result, permutation);
    /*buildGrids_CPU2GPU(pts, sortedpts, cellStartIds, M, permutation, exact_result);*/
    Buildgrids_time.stop();   

    gpu_timer Bodyprogram_time("Delaunay Triangulation on GPU Body program");
#ifdef USINGDOUBLE
    cuVector<float3> sortedpts2(nv+4);
    cast << <(nv - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> >(sortedpts.data(), sortedpts2.data(), nv+4);
    DT_kernel << < (nv_process - 1) / BLOCK_SIZE_MAIN + 1, BLOCK_SIZE_MAIN >> > (sortedpts2.data(), nv, 0, predConsts.data(), stat.data(), output.data(), cellStartIds.data(), M, exact_result.data(), permutation.data(), ncycles.data(), time.data());
#else
    DT_kernel << < divide(2 * nv_process, BLOCK_SIZE_MAIN), BLOCK_SIZE_MAIN  /*1,2*/>> > (sortedpts.data(), nv, 0, predConsts.data(), stat.data(), output.data(), cellStartIds.data(), M, exact_result.data(), permutation.data(), ncycles.data(), time.data());
#endif
    Bodyprogram_time.stop(); 
    
    gpu_timer compress_time("Compress Output");
    compressoutput(0);
    compress_time.stop();

    gpu_timer Postprocessingmain_time("Delaunay Triangulation on GPU Post-processing Main");
    postinit(0);
    std::cout << "Postprocessing points:" << npost << std::endl;
    PostProcessingMain_kernel << < divide(npost, BLOCK_SIZE_POST), BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), permutation.data(), ncycles.data(), time.data());
    Postprocessingmain_time.stop();

    PostProcessingPost_kernel << < 1, BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());
    gpu_timer Postprocessingpost_time("Delaunay Triangulation on GPU Post-processing Post");
    PostProcessingPost_kernel << < npost, BLOCK_SIZE_POST >> > (sortedpts.data(), nv, 0, predConstsR.data(), stat.data(), outputpost.data(), cellStartIds.data(), M, npost, unsuccessIds.data(), exact_result.data(), permutation.data(), ncycles.data(), time.data());
    Postprocessingpost_time.stop();
   
    gpu_timer compress_time2("Compress Output post");
    compressoutputpost(0);
    compress_time2.stop();
   
    cudaDeviceSynchronize(); 
    Buildgrids_time.report();
    Bodyprogram_time.report();
    Postprocessingmain_time.report();
    Postprocessingpost_time.report();
    compress_time.report();
    compress_time2.report();
    std::cout << "Delaunay Triangulation on GPU total time:" << Buildgrids_time.time + Bodyprogram_time.time + Postprocessingmain_time.time + compress_time.time+ Postprocessingpost_time.time + compress_time2.time << " msec" << std::endl;

    copywrite(outfile);
    record();
}


void DelaunaySolver::copywrite(const char* outfile) {
    size_t ntet = totaltet.at(0);
    std::cout << "Number of Tets:" << ntet << std::endl;
    std::cout << "copy data back to the cpu..." << std::endl;    
    if (nv == nv_process) {
        std::vector<uchar> h_stat = stat;
        show_status_stats(h_stat);
    }    
    if (outfile) {
        std::cout << "write data to the file..." << std::endl;
        std::vector< REAL3> h_sortedpts = sortedpts;
        std::vector<int4> h_tets = tet;
        std::vector< uchar> h_exact = exact_result;
        writeMat((outfile + std::string("x.bin")).c_str(), (const REAL*)h_sortedpts.data(), nv, 3);
        writeMat((outfile + std::string("t.bin")).c_str(), (const int*)h_tets.data(), ntet, 4);
        if (exact_result.data()) {
            std::vector< uchar> h_exact = exact_result;
            writeMat((outfile + std::string("b.bin")).c_str(), (const uchar*)h_exact.data(), nv, 1);
        }
    }
}


void DelaunaySolver::record() {
    if(!time.empty()|| !ncycles.empty())
        std::cout << "record information..." << std::endl;
    std::fstream file;
    if (!time.empty()) {
        std::vector<clock_t> h_time = time;
        int clockRateKHz;
        cudaError_t err = cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
        file.open("time.txt", std::ios_base::out);
        file.precision(9);
        for (int i = 0; i < 2 * nv_process; i++) {
            file << (double)h_time[i] / clockRateKHz << std::endl;

        }
        file.close();
    }

    if (!ncycles.empty()) {
        std::vector<int> h_ncycles = ncycles;
        file.open("num.txt", std::ios_base::out);
        for (int i = 0; i < 2 * nv_process; i++) {
            file << h_ncycles[i] << std::endl;

        }
        file.close();
    }
}
