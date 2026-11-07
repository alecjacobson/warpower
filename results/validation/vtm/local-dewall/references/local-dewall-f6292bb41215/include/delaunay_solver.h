#pragma once
#include "delaunay_kernels.h"
class DelaunaySolver {

public:
    DelaunaySolver(const std::vector<REAL3>& h_pts, std::vector<uchar>& boundary, int n=0) {
        init(h_pts, boundary, n);
    }
    void cvt(int Iter, const char* outfile = nullptr);    
    void compute_DT_GPU_large(const char* outfile = nullptr);
    void compute_DT_GPU(const char* outfile = nullptr);

private:
    void init(const std::vector<REAL3>& h_pts, std::vector<uchar>& boundary, int n);
    void addinfpoints();
    void postinit(int startseed);
    void compressoutput(int startseed);
    void compressoutputpost(int startseed);
    void copywrite(const char* outfile = nullptr);
    void record();

private:
    int nv;
    int nv_process;
    int npost;
    int M;
    cuVector<uchar> exact_result;
    cuVector<REAL3> pts;
    cuVector<REAL3> sortedpts;

    cuVector<uchar> stat;
    cuVector<int> cellStartIds;
    cuVector<DTStruct> output;

    cuVector<int> unsuccessIds;
    cuVector<int> nunsuccess;
    cuVector< DTStructPost> outputpost;

    cuVector<clock_t> time;
    cuVector<int> ncycles;
    cuVector<int> permutation;

    cuVector<int4> tet;//tetrahedra list
    cuVector<int> ntet;
    cuVector<int> accnt;    
    cuVector<size_t> totaltet;
};

