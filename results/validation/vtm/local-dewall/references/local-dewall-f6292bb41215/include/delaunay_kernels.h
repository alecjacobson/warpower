#pragma once
#include "spatial_hash.h"
#include "types.h"



#define SORTING_POINTS
//#define OUTPUT_INFORMATION

const int THREADS_PER_BLOCK = 256;
const int BLOCK_SIZE_MAIN = 128;
const int BLOCK_SIZE_POST = 128;
const int MAX_CELL = 100;
const int  MAX_P = 32;
const int  MAX_T = MAX_P * 2;
const int  MAX_POST_P = 128;
const int  MAX_POST_T = MAX_POST_P * 2;

const int  MAX_CYCLES = 100;
const int  MAX_POST_CELL = 200;
const int  MAX_POST_CYCLES1 = 40;
const int  MAX_POST_CYCLES2 = 500;


extern __constant__ REAL3 infvector[8];


typedef unsigned char uchar;

#define BitGet(p,n) ((p)& (1 <<(n )))
#define BitSet(p,n) (p)=((p)| (1 <<(n )))
#define BitClear(p,n) (p)=((p)& (~(1 <<(n ))))

class BitMap {
public:
    __device__ BitMap() { init(); }
    __device__ void init() {
        for (int i = 0; i < ((MAX_POST_P - 1) / 32 + 1); i++)
            map[i] = 0;
    }
    __device__ void set(int i) {
        BitSet(map[i / 32], i % 32);
    }
    __device__ void clear(int i) {
        BitClear(map[i / 32], i % 32);
    }
    __device__ bool check(int i) {
        return BitGet(map[i / 32], i % 32) != 0;
    }
private:
    int map[(MAX_POST_P - 1) / 32 + 1];
};

class DTStruct {
public:
    uchar nv[2];
    int vertex_data[2 * MAX_P];
};

class DTStructPost {
public:
    int nv;
    uchar nt;
    uchar4 tr_data[MAX_POST_T];
    int index_data[MAX_POST_P];
};



template < typename R3>
class Delaunay1Ring {
public:
    using R = decltype(R3::x);
    __device__ Delaunay1Ring() {};
    __device__ void  init(int p_seed, R3* p_pts, int nbpts, int* p_cellStartIds, int dim, int* p_vertex, R* p_predConsts);
    __device__ void shrink();
    __device__ void reoder();
    __device__ void dt_init();
    __device__ int encode_index(int v);
    __device__ int decode_index(int v);
    __device__ bool compare_position(int v);
    __device__ void update();
    __device__ bool intriangle(int3 tri);

    __device__ void  cell_range(int3* lowbound, int3* upbound, int3* startcell);
    __device__ void  cell_range2(int3* lowbound, int3* upbound, int3* startcell);
    __device__ int ballsearch();
    __device__ bool isfinished();

   /* __device__ int ballsearch_post(int3 lowbound, int3 upbound, int i, R* detpre, R* boundpre);
    __device__ void update_post();
    __device__ bool intriangle_post(int3 tri);
    __device__ void shrink_post();*/
    
  
    inline  __device__ int& vertex(int k){ return  vertex_data[k]; }
    inline  __device__ int ridx(int k) { return  decode_index(vertex_data[k]); }
    inline  __device__ int idx_before(int k) {
        for (int i = k - 1; i >= 0; i--)
            if ((vertex(k) < 0 && vertex(i) >= 0) || (vertex(k) >= 0 && vertex(i) < 0))
                return decode_index(vertex(i));
    }

public:
    int3 act_triangle;  //the triangle we are processing
    uchar status;
    int nv;   
    bool trust_tag;
    int3 aim;
    R3* pts;
    int target_id;
    R3 target_coords;    
    int npts;
    int* cellStartIds;
    int M;//dim of the grid in one dimension
#ifndef SORTING_POINTS
    int* permutation;
#endif
    R* predConsts;
    int* vertex_data;

};

class MainPhase {

public:
    __device__ MainPhase() { num = 0; };
    __device__ void  update();
public:
    Delaunay1Ring<float3> dr;
    int num;
};



template < typename T4, typename R3>
class Delaunay1RingPost {
public:
    using T = decltype(T4::x);
    using R = decltype(R3::x);
    __device__ Delaunay1RingPost() {};
    __device__ void  init(int p_seed, R3* p_pts, int nbpts, int* p_cellStartIds, int dim, T4* p_tr, int* p_idx, int p_maxp, R* p_predConsts);
    __device__ void dt_init(int n);
    __device__ void remove_triangle(int i);
    __device__ bool halfspace_intersection(int k);
    __device__ void halfspace_intersection_post(int vid, uchar* predict_tag);
    __device__ inline  T& ith_plane(T t, int i) { return reinterpret_cast<T*>(&(tr(t)))[i]; };
    __device__ int encode_index(int v);
    __device__ int decode_index(int v);
    __device__ int add_vertex(int vid);
    __device__ void add_triangle(T i, T j, T k);
    __device__ T compute_boundary();
    __device__ void  cell_range(int3* lowbound, int3* upbound, R3* start);
    __device__ bool compare_position(int v);
    __device__ int ballsearch_post(int3 lowbound, int3 upbound, int3 lowbound0, int3 upbound0, R* detpre, R* boundpre, int* npredict);
    __device__ void output();

    inline  __device__ int& index(int v) { return  index_data[v]; }
    inline  __device__ int rindex(int v) { return  decode_index(index_data[v]); }
    inline  __device__ T4& tr(int t) { return  tr_data[t]; }
    inline  __device__ T& tag(int t) { return  tr_data[t].w; }
public:
    int3 act_triangle;  //the triangle we are processing 
    Status status;
    T nv;
    T nt;
    T nr;
    T noutput;
    R3* pts;
    int target_id;
    R3 target_coords;
    int npts;
    int* cellStartIds;
    int M;//dim of the grid in one dimension
    int maxp;
    uchar exact_tag;
#ifndef SORTING_POINTS
    int* permutation;
#endif
    T4* tr_data;
    int* index_data;
    R* predConsts;
    T* boundary_next;
};


template <typename T4>
class PostProcessingPost {
    using T = decltype(T4::x);

public:
    __device__ PostProcessingPost() {};
    __device__ void  init();
    __device__ void  update_activetr();
    __device__ void  search_prepare();
    __device__ void  idx_reduction(int idx);
    __device__ void  do_predict(uchar* predict_tag);
    __device__ int  ballsearch(int* npredict);
public:
    int idx_block;
    T activetr_id;//the tri_id we are processing
    bool finished;
    int num;
    Delaunay1RingPost<T4, REAL3> dr;
    int3 lowbound;
    int3 upbound;
    REAL3 center;
    bool tag;
    int idx_warp[BLOCK_SIZE_POST / 32];
    REAL dist_warp[BLOCK_SIZE_POST / 32];
    REAL detpre[4];
    REAL boundpre[4];
};




__global__ void DT_kernel(REAL3* pts, int nv, int startseed, REAL* predConsts, uchar* stat, DTStruct* output, int* cellStartIds, int M, uchar* exact, int* permutation, int* ncycles, clock_t* time);

__global__ void PostProcessingMain_kernel(float3* pts, int nv, int startseed, float* predConsts, uchar* stat, DTStructPost* output, int* cellStartIds, int M, int nunsuccess, int* unsuccessIds, int* permutation, int* ncycles, clock_t* time);

__global__ void PostProcessingPost_kernel(REAL3* pts, int nv, int startseed, REAL* predConsts, uchar* stat, DTStructPost* output, int* cellStartIds, int M, int nunsuccess, int* unsuccessIds, uchar* exact, int* permutation, int* ncycles, clock_t* time);

__global__ void  Voronoi_center(REAL3* pts, int nv, DTStruct* output, uchar* stat, REAL3* newpts, REAL* cellvol, uchar* exact_result, uchar* boundary);
__global__ void  Voronoi_center_post(REAL3* pts, int nunsuccess, int* unsuccessIds, DTStructPost* output, REAL3* newpts, REAL* cellvol, uchar* exact_result, uchar* boundary);

__global__ void update_pts(REAL3* pts, REAL3* sortedpts, int nv, REAL3* newpts, REAL* tetvol, uchar* boundary, REAL* distance);
