#include "spatial_hash.h"
#include <omp.h> 
#include <cub/cub.cuh>
#include "geometry.h"

__device__ __host__ int cellFromPoint(REAL3 p, int M) {
    return getcellidx_flat(p, M);
}

void buildGrids_CPU(std::vector<REAL3>& pts, int nv, int &M, std::vector<int>& cellStartIds, std::vector<REAL3>& sortedpts, std::vector<int>& permutation, std::vector<uchar>&exact_result) {
    int cellnum = M * M * M;   

    std::vector<int> bktSizes(cellnum, 0);    

#pragma omp parallel for 
    for (int i = 0; i < nv; i++) {
        const int idx = cellFromPoint(pts[i], M);

#pragma omp atomic 
        ++bktSizes[idx];

    }
    int cmin = INT_MAX, cmax = 0; int nempty = 0;
    for (int c = 0; c < cellnum; c++) {
        if (bktSizes[c] == 0)
            nempty++;
        cmin = min(cmin, bktSizes[c]);
        cmax = max(cmax, bktSizes[c]);
    }
    std::cout << "Grid:  points per cell: " << cmin << " (min), " << cmax << " (max), "  << std::endl;
    std::cout << "empty cells :" << nempty << "/" << cellnum << std::endl;
    /*int M2 = (int)round(pow(cellnum * (cellnum / double(cellnum - nempty)), 1.0 / 3));
    if (M2 > M) {
        M = M2;
        cellnum = M * M * M;
        std::cout << "using new cells :" << cellnum << std::endl;
        bktSizes = std::vector<int>(cellnum , 0);
#pragma omp parallel for 
        for (int i = 0; i < nv; i++) {
            const int idx = cellFromPoint(pts[i], M);

#pragma omp atomic 
            ++bktSizes[idx];

        }
    }*/
    cellStartIds = std::vector<int>(cellnum + 1, 0);
    std::vector<int> mark(cellnum, 0);
    std::vector<uchar> new_exact(nv,0);
    std::partial_sum(bktSizes.cbegin(), bktSizes.cend(), cellStartIds.begin() + 1);
    for (int i = 0; i < nv; i++) {
        const int idx = cellFromPoint(pts[i], M);
        int pos = cellStartIds[idx] + mark[idx]++;
        sortedpts[pos] = pts[i];
        permutation[pos] = i;
        if (exact_result.size()==nv)
            new_exact[pos] = exact_result[i];
    }
    if (exact_result.size() == nv)
        for(int i=0; i<nv;i++)
            exact_result[i] = new_exact[i];
}

void buildGrids_CPU2GPU(cuVector<REAL3>& d_pts, cuVector<REAL3>& d_sortedpts, cuVector<int>& d_cellStartIds, int& M, cuVector<int>& d_permutation, cuVector<uchar>& exact_result) {
    int nv = d_pts.size();
    std::vector<REAL3> h_pts = d_pts;
    std::vector<int>h_cellstartIds(d_cellStartIds.size());
    std::vector<int>h_permutation(nv);
    std::vector<REAL3>h_sortedpts(nv);
    std::vector<uchar> h_exact = exact_result;
    buildGrids_CPU(h_pts, nv, M, h_cellstartIds, h_sortedpts, h_permutation, h_exact);
    d_cellStartIds = h_cellstartIds;
    if (d_permutation.empty()) {
        cuda_check(cudaMemcpy(d_sortedpts.data(), h_sortedpts.data(), nv * sizeof(REAL3), cudaMemcpyHostToDevice));//d_sortedpts have four infinite points in the end
    }
    else {
        d_permutation = h_permutation;
        cuda_check(cudaMemcpy(d_sortedpts.data(), h_pts.data(), nv * sizeof(REAL3), cudaMemcpyHostToDevice));
    }
    exact_result = h_exact;
}


__global__ void countpoints(const REAL3* d_pts, int nv, int M, int* d_counters) {
    int seed = blockDim.x * blockIdx.x + threadIdx.x;
    if (seed < nv) {
        int cell = cellFromPoint(d_pts[seed], M);
        atomicAdd(d_counters + cell, 1);
    }
}

__global__ void storepoints(const REAL3* d_pts, int nv, int M, const int* d_cellStartIds, int* d_counters, REAL3* d_sortedpts, uchar* new_exact, int* d_permutation, uchar* exact_result) {
    int seed = blockDim.x * blockIdx.x + threadIdx.x;
    if (seed < nv) {
        REAL3 p = d_pts[seed];
        int cell = cellFromPoint(p, M);
        int pos = d_cellStartIds[cell] + atomicAdd(d_counters + cell, 1);
        if (d_permutation) {
            d_permutation[pos] = seed;
            d_sortedpts[seed] = p;            
        }
        else {
            d_sortedpts[pos] = p;
            if (exact_result)
                new_exact[pos] = exact_result[seed];
        }
    }
}

__global__ void emptycell( int cellnum, const int* d_counters, int* d_counter) {
    int seed = blockDim.x * blockIdx.x + threadIdx.x;
    if (seed < cellnum) {
        if (d_counters[seed] == 0)
            atomicAdd(d_counter, 1);
    }
}

__global__ void copyexact(int nv, uchar* exact_result, uchar* new_exact) {
    int seed = blockDim.x * blockIdx.x + threadIdx.x;
    if (seed < nv) 
        exact_result[seed] = new_exact[seed];

}


void buildGrids_GPU(cuVector<REAL3>& d_pts, cuVector<REAL3>& d_sortedpts, cuVector<int>& d_cellStartIds, int& M, cuVector<uchar>& exact_result, cuVector<int>& d_permutation) {
    int nv = d_pts.size();
    int cellnum = M * M * M;
    cuVector<uchar> new_exact(exact_result.size());
    cuVector<int> d_counters(cellnum);
    d_counters.zero_fill();
    std::cout << "using cells in each dim :" << M << std::endl;    
    countpoints << <(nv - 1) / 128 + 1, 128 >> > (d_pts.data(), nv, M, d_counters.data());

    cuVector<int> d_max(1);
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
   /* cub::DeviceReduce::Max( d_temp_storage, temp_storage_bytes, d_counters.data(), d_max.data(), cellnum);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_counters.data(), d_max.data(), cellnum);
    int maxp = d_max.at(0);
    std::cout << "max points per cell :" << maxp  << std::endl;*/
  /*  int M2 = (int)round(pow(maxp*nv / 8.0f, 1.0f / 3.0));*/
   /* cuVector<int> d_counter(1);
    d_counter.zero_fill();
    emptycell << <(cellnum - 1) / 128 + 1, 128 >> > (cellnum, d_counters.data(), d_counter.data());
    int nempty = d_counter.at(0);
    std::cout << "empty cells :" << nempty << "/" << cellnum << std::endl;*/
    /*int M2= (int)round(pow(cellnum * (cellnum / double(cellnum - nempty)), 1.0 / 3));*/   
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_counters.data(), d_cellStartIds.data() + 1, cellnum);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_counters.data(), d_cellStartIds.data() + 1, cellnum);
    cudaFree(d_temp_storage);

    d_counters.zero_fill();
    storepoints << <(nv - 1) / 128 + 1, 128 >> > (d_pts.data(), nv, M, d_cellStartIds.data(), d_counters.data(), d_sortedpts.data(),  new_exact.data(), d_permutation.data(), exact_result.data());
    if(exact_result.size()>0 && d_permutation.size()==0)
        copyexact << <(nv - 1) / 128 + 1, 128 >> > (nv, exact_result.data(), new_exact.data());

}