#pragma once
// Expansion arithmetic and robust-predicate routines in this file are based
// on Jonathan Richard Shewchuk's public-domain robust predicates:
// https://www.cs.cmu.edu/~quake/robust.html
// See THIRD_PARTY_NOTICES.md for attribution.
#include <cuda_runtime.h>

#define REALPRINT doubleprint
#define REALRAND doublerand
#define NARROWRAND narrowdoublerand
#define UNIFORMRAND uniformdoublerand
#define Absolute(a)  fabs(a)

#define Fast_Two_Sum_Tail(a, b, x, y) \
  bvirt = x - a; \
  y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
  Two_Sum(a, b, x, y)

#define Fast_Two_Diff_Tail(a, b, x, y) \
  bvirt = a - x; \
  y = bvirt - b

#define Fast_Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Fast_Two_Diff_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
  bvirt = (REAL) (x - a); \
  avirt = x - bvirt; \
  bround = b - bvirt; \
  around = a - avirt; \
  y = around + bround

#define Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
  bvirt = (REAL) (a - x); \
  avirt = x + bvirt; \
  bround = bvirt - b; \
  around = a - avirt; \
  y = around + bround

#define Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Two_Diff_Tail(a, b, x, y)

#define Two_Product(a, b, x, y) \
  x = (REAL) (a * b); \
  y = fma(a, b, -x);


//#define Two_Product_Presplit(a, b, bhi, blo, x, y) Two_Product(a, b, x, y)

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
  Two_Product(a0, b, _i, x0); \
  Two_Product(a1, b, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, x3, x2)


#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
  Two_Sum(a0, b , _i, x0); \
  Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
  Two_Diff(a0, b , _i, x0); \
  Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b0, _j, _0, x0); \
  Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Diff(a1, a0, b0, _j, _0, x0); \
  Two_One_Diff(_j, _0, b1, x3, x2, x1)

#define Four_One_Sum(a3, a2, a1, a0, b, x4, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b , _j, x1, x0); \
  Two_One_Sum(a3, a2, _j, x4, x3, x2)

#define Four_Two_Sum(a3, a2, a1, a0, b1, b0, x5, x4, x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b0, _k, _2, _1, _0, x0); \
  Four_One_Sum(_k, _2, _1, _0, b1, x5, x4, x3, x2, x1)

#define Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0, x7, x6, x5, x4, x3, x2, \
                      x1, x0) \
  Four_Two_Sum(a3, a2, a1, a0, b1, b0, _l, _2, _1, _0, x1, x0); \
  Four_Two_Sum(_l, _2, _1, _0, b4, b3, x7, x6, x5, x4, x3, x2)

#define Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b, x8, x7, x6, x5, x4, \
                      x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b , _j, x3, x2, x1, x0); \
  Four_One_Sum(a7, a6, a5, a4, _j, x8, x7, x6, x5, x4)

#define Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, x9, x8, x7, \
                      x6, x5, x4, x3, x2, x1, x0) \
  Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b0, _k, _6, _5, _4, _3, _2, \
                _1, _0, x0); \
  Eight_One_Sum(_k, _6, _5, _4, _3, _2, _1, _0, b1, x9, x8, x7, x6, x5, x4, \
                x3, x2, x1)

#define Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0, x11, \
                       x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0) \
  Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, _l, _6, _5, _4, _3, \
                _2, _1, _0, x1, x0); \
  Eight_Two_Sum(_l, _6, _5, _4, _3, _2, _1, _0, b4, b3, x11, x10, x9, x8, \
                x7, x6, x5, x4, x3, x2)


template <typename T> __device__ void inline swap(T& a, T& b) { T c(a); a = b; b = c; }

__device__ int d_scale_expansion_zeroelim(int elen, REAL* e, REAL b, REAL* h)
/* e and h cannot be the same. */
{
    REAL Q, sum;
    REAL hh;
    REAL product1;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    REAL bvirt;
    REAL avirt, bround, around;

    Two_Product(e[0], b, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product(enow, b, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

__device__ int d_fast_expansion_sum_zeroelim(int elen, REAL* e, int flen, REAL* f, REAL* h)
/* h cannot be e or f. */
{
    REAL Q;
    REAL Qnew;
    REAL hh;
    REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex, hindex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    }
    else {
        Q = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        }
        else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            }
            else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}


__device__ REAL d_fast_expansion_sum_sign(int elen, REAL* e, int flen, REAL* f){
    REAL Q;
    REAL lastTerm;
    REAL Qnew;
    REAL hh;
    REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    }
    else {
        Q = fnow;
        fnow = f[++findex];
    }
    lastTerm = 0.0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        }
        else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            }
            else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                lastTerm = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            lastTerm = hh;
        }
    }
    if ((Q != 0.0) || (lastTerm == 0.0)) {
        lastTerm = Q;
    }
    return lastTerm;
}

__device__ REAL d_fast_expansion_sum_estimate(int elen, REAL* e, int flen, REAL* f){
    REAL Q;
    REAL est;
    REAL Qnew;
    REAL hh;
    REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    }
    else {
        Q = fnow;
        fnow = f[++findex];
    }
    est = 0.0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        }
        else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        est += hh;
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            }
            else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            est += hh;
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        est += hh;
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        est += hh;
    }

    est += Q;

    return est;
}

__device__ int d_squared_scale_expansion_zeroelim(int elen, REAL* e, REAL b, REAL* h){
    REAL Q, sum, Q2, sum2;
    REAL hh;
    REAL product1, product2;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    REAL bvirt;
    REAL avirt, bround, around;

    hindex = 0;

    Two_Product(e[0], b, Q, hh);
    Two_Product(hh, b, Q2, hh);

    if (hh != 0) {
        h[hindex++] = hh;
    }

    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product(enow, b, product1, product0);
        Two_Sum(Q, product0, sum, hh);

        Two_Product(hh, b, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product1, sum, Q, hh);

        Two_Product(hh, b, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if (Q != 0) {
        Two_Product(Q, b, product2, product0);
        Two_Sum(Q2, product0, sum2, hh);

        if (hh != 0) {
            h[hindex++] = hh;
        }

        Fast_Two_Sum(product2, sum2, Q2, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }

    if ((Q2 != 0) || (hindex == 0)) {
        h[hindex++] = Q2;
    }

    return hindex;
}

__device__ REAL estimate(int elen, REAL* e)
{
    REAL Q;
    int eindex;

    Q = e[0];
    for (eindex = 1; eindex < elen; eindex++) {
        Q += e[eindex];
    }
    return Q;
}

__device__ REAL orient2dadapt(const REAL* predConsts, REAL* pa, REAL* pb, REAL* pc, REAL detsum)
{
    REAL acx, acy, bcx, bcy;
    REAL acxtail, acytail, bcxtail, bcytail;
    REAL detleft, detright;
    REAL detlefttail, detrighttail;
    REAL det, errbound;
    REAL B[4], C1[8], C2[12], D[16];
    REAL B3;
    int C1length, C2length, Dlength;
    REAL u[4];
    REAL u3;
    REAL s1, t1;
    REAL s0, t0;

    REAL bvirt;
    REAL avirt, bround, around;
    REAL _i, _j;
    REAL _0;

    acx = (REAL)(pa[0] - pc[0]);
    bcx = (REAL)(pb[0] - pc[0]);
    acy = (REAL)(pa[1] - pc[1]);
    bcy = (REAL)(pb[1] - pc[1]);

    Two_Product(acx, bcy, detleft, detlefttail);
    Two_Product(acy, bcx, detright, detrighttail);

    Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
        B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = estimate(4, B);
    errbound = predConsts[ccwerrboundB] * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pc[0], acx, acxtail);
    Two_Diff_Tail(pb[0], pc[0], bcx, bcxtail);
    Two_Diff_Tail(pa[1], pc[1], acy, acytail);
    Two_Diff_Tail(pb[1], pc[1], bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0)
        && (bcxtail == 0.0) && (bcytail == 0.0)) {
        return det;
    }

    errbound = predConsts[ccwerrboundC] * detsum + predConsts[resulterrbound] * Absolute(det);
    det += (acx * bcytail + bcy * acxtail)
        - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Product(acxtail, bcy, s1, s0);
    Two_Product(acytail, bcx, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = d_fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    Two_Product(acx, bcytail, s1, s0);
    Two_Product(acy, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = d_fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    Two_Product(acxtail, bcytail, s1, s0);
    Two_Product(acytail, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = d_fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    return(D[Dlength - 1]);
}

__device__ REAL orient2d(const REAL* predConsts, REAL* pa, REAL* pb, REAL* pc)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        }
        else {
            detsum = detleft + detright;
        }
    }
    else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        }
        else {
            detsum = -detleft - detright;
        }
    }
    else {
        return det;
    }

    errbound = predConsts[ccwerrboundA] * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return orient2dadapt(predConsts, pa, pb, pc, detsum);
}

__device__ REAL orient2dfast(const REAL* predConsts, REAL* pa, REAL* pb, REAL* pc, bool* tag)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    det = detleft - detright;

    /*if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        }
        else {
            detsum = detleft + detright;
        }
    }
    else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        }
        else {
            detsum = -detleft - detright;
        }
    }
    else {
        return det;
    }*/
    detsum = Absolute(detleft) + Absolute(detright);
    errbound = predConsts[ccwerrboundA] * detsum;
    if ((det <= errbound) && (det >= -errbound))
        *tag = false;

    return det;
}

__device__ bool orient3dfast(const REAL* predConsts, REAL3& pa, REAL3& pb, REAL3& pc, REAL3& pd, bool* tag)
{
    REAL adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL det;


    adx = pa.x - pd.x;
    ady = pa.y - pd.y;
    adz = pa.z - pd.z;
    bdx = pb.x - pd.x;
    bdy = pb.y - pd.y;
    bdz = pb.z - pd.z;
    cdx = pc.x - pd.x;
    cdy = pc.y - pd.y;
    cdz = pc.z - pd.z;

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy)
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    REAL permanent, errbound;

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
        + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
        + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);
    errbound = predConsts[o3derrboundA] * permanent;
    if (det < -errbound)
        return true;

    return false;


}

__device__  void orient3dprepare(REAL3& pb, REAL3& pc, REAL3& pd, REAL* detpre, REAL* boundpre) {
    REAL bdx, cdx, bdy, cdy, bdz, cdz;
    REAL bdycdz, cdybdz, bdxcdz, cdxbdz, bdxcdy, cdxbdy;

    bdx = pb.x - pd.x;
    bdy = pb.y - pd.y;
    bdz = pb.z - pd.z;
    cdx = pc.x - pd.x;
    cdy = pc.y - pd.y;
    cdz = pc.z - pd.z; 

    bdycdz = bdy * cdz;
    cdybdz = cdy * bdz;

    bdxcdz = bdx * cdz;
    cdxbdz = cdx * bdz;

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    detpre[0] = bdycdz - cdybdz;
    detpre[1] = bdxcdz - cdxbdz;
    detpre[2] = bdxcdy - cdxbdy;

    boundpre[0] = Absolute(bdycdz) + Absolute(cdybdz);
    boundpre[1] = Absolute(bdxcdz) + Absolute(cdxbdz);
    boundpre[2] = Absolute(bdxcdy) + Absolute(cdxbdy);

}

__device__ bool orient3dfast(const REAL* predConsts, REAL3& pa, REAL3& pd, REAL* detpre, REAL* boundpre, bool* tag)
{
    REAL adx, ady, adz;
    REAL det;

    adx = pa.x - pd.x;
    ady = pa.y - pd.y;
    adz = pa.z - pd.z;

    det = adx * detpre[0]
        + ady * detpre[1]
        + adz * detpre[2];

    REAL permanent, errbound;

    permanent = boundpre[0] * Absolute(adx)
        + boundpre[1] * Absolute(ady)
        + boundpre[2] * Absolute(adz);
    errbound = predConsts[o3derrboundA] * permanent;
    if ((det <= errbound) && (det >= -errbound))
        *tag = false;

    if (det < -errbound)
        return true;

    return false;


}

__device__ REAL orient1d_Lifted( const REAL* pa, const REAL* pb, bool lifted){
    if (!lifted)
        return (pa[0] - pb[0]);

    REAL axax1, ayay1, azaz1, bxbx1, byby1, bzbz1;
    REAL axax0, ayay0, azaz0, bxbx0, byby0, bzbz0;
    REAL aterms[4], bterms[4], cterms[4];
    REAL aterms3, bterms3, cterms3;
    REAL v[8];
    int vlength;

    REAL bvirt;
    REAL avirt, bround, around;
    REAL _i, _j;
    REAL _0;

    Two_Product(pa[0], pa[0], axax1, axax0);
    Two_Product(pb[0], pb[0], bxbx1, bxbx0);
    Two_Two_Diff(axax1, axax0, bxbx1, bxbx0,
        aterms3, aterms[2], aterms[1], aterms[0]);
    aterms[3] = aterms3;

    Two_Product(pa[1], pa[1], ayay1, ayay0);
    Two_Product(pb[1], pb[1], byby1, byby0);
    Two_Two_Diff(ayay1, ayay0, byby1, byby0,
        bterms3, bterms[2], bterms[1], bterms[0]);
    bterms[3] = bterms3;

    Two_Product(pa[2], pa[2], azaz1, azaz0);
    Two_Product(pb[2], pb[2], bzbz1, bzbz0);
    Two_Two_Diff(azaz1, azaz0, bzbz1, bzbz0,
        cterms3, cterms[2], cterms[1], cterms[0]);
    cterms[3] = cterms3;

    vlength = d_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);

    return d_fast_expansion_sum_sign(vlength, v, 4, cterms);
}

__device__ REAL orient2d_Lifted(REAL* predData, const REAL* pa, const REAL* pb, const REAL* pc, bool lifted){
    REAL* aterms = predData; predData += 24;
    REAL* bterms = predData; predData += 24;
    REAL* cterms = predData; predData += 24;
    REAL* v = predData;

    REAL aax1, aax0, aay1, aay0, aaz[2];
    REAL temp[4];
    REAL palift[6], pblift[6], pclift[6];
    REAL xy1terms[12], xy2terms[12];

    int palen, pblen, pclen;
    int xy1len, xy2len;
    int alen, blen, clen;
    int vlen;

    REAL bvirt;
    REAL avirt, bround, around;
    REAL _i, _j;
    REAL _0;

    // Compute the lifted coordinate
    if (lifted)
    {
        Two_Product(pa[0], pa[0], aax1, aax0);
        Two_Product(-pa[1], pa[1], aay1, aay0);
        Two_Product(pa[2], pa[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        palen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);

        Two_Product(pb[0], pb[0], aax1, aax0);
        Two_Product(-pb[1], pb[1], aay1, aay0);
        Two_Product(pb[2], pb[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        pblen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);

        Two_Product(pc[0], pc[0], aax1, aax0);
        Two_Product(-pc[1], pc[1], aay1, aay0);
        Two_Product(pc[2], pc[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        pclen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);
    }
    else {
        palen = 1; palift[0] = pa[1];
        pblen = 1; pblift[0] = pb[1];
        pclen = 1; pclift[0] = pc[1];
    }

    // Compute the determinant as usual
    xy1len = d_scale_expansion_zeroelim(pblen, pblift, pa[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pclen, pclift, -pa[0], xy2terms);
    alen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, aterms);

    xy1len = d_scale_expansion_zeroelim(pclen, pclift, pb[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(palen, palift, -pb[0], xy2terms);
    blen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bterms);

    xy1len = d_scale_expansion_zeroelim(palen, palift, pc[0], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pblen, pblift, -pc[0], xy2terms);
    clen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cterms);

    vlen = d_fast_expansion_sum_zeroelim(alen, aterms, blen, bterms, v);

    return d_fast_expansion_sum_sign(vlen, v, clen, cterms);
}
__device__ REAL orient3dExact_Lifted(const REAL* predConsts, REAL* predData, const REAL* pa, const REAL* pb, const REAL* pc, const REAL* pd, bool lifted){
    REAL* ab = predData; predData += 24;
    REAL* bc = predData; predData += 24;
    REAL* cd = predData; predData += 24;
    REAL* da = predData; predData += 24;
    REAL* ac = predData; predData += 24;
    REAL* bd = predData; predData += 24;
    REAL* temp48 = predData; predData += 48;
    REAL* cda = predData; predData += 72;
    REAL* dab = predData; predData += 72;
    REAL* abc = predData; predData += 72;
    REAL* bcd = predData; predData += 72;
    REAL* adet = predData; predData += 144;
    REAL* bdet = predData; predData += 144;
    REAL* cdet = predData; predData += 144;
    REAL* ddet = predData; predData += 144;
    REAL* abdet = predData; predData += 288;
    REAL* cddet = predData;

    REAL aax1, aax0, aay1, aay0, aaz[2];
    REAL temp[4];
    REAL palift[6], pblift[6], pclift[6], pdlift[6];
    REAL xy1terms[12], xy2terms[12];

    int templen;
    int palen, pblen, pclen, pdlen;
    int xy1len, xy2len;
    int ablen, bclen, cdlen, dalen, aclen, bdlen;
    int cdalen, dablen, abclen, bcdlen;
    int alen, blen, clen, dlen;

    REAL bvirt;
    REAL avirt, bround, around;
    REAL _i, _j;
    REAL _0;

    // Compute the lifted coordinate
    if (lifted) {
        Two_Product(pa[0], pa[0], aax1, aax0);
        Two_Product(-pa[1], pa[1], aay1, aay0);
        Two_Product(pa[2], pa[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        palen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);

        Two_Product(pb[0], pb[0], aax1, aax0);
        Two_Product(-pb[1], pb[1], aay1, aay0);
        Two_Product(pb[2], pb[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        pblen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);

        Two_Product(pc[0], pc[0], aax1, aax0);
        Two_Product(-pc[1], pc[1], aay1, aay0);
        Two_Product(pc[2], pc[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        pclen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);

        Two_Product(pd[0], pd[0], aax1, aax0);
        Two_Product(-pd[1], pd[1], aay1, aay0);
        Two_Product(pd[2], pd[2], aaz[1], aaz[0]);
        Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]);
        pdlen = d_fast_expansion_sum_zeroelim(4, temp, 2, aaz, pdlift);
    }
    else
    {
        palen = 1; palift[0] = pa[2];
        pblen = 1; pblift[0] = pb[2];
        pclen = 1; pclift[0] = pc[2];
        pdlen = 1; pdlift[0] = pd[2];
    }

    // Compute the determinant as usual
    xy1len = d_scale_expansion_zeroelim(pblen, pblift, pa[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(palen, palift, -pb[1], xy2terms);
    ablen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ab);

    xy1len = d_scale_expansion_zeroelim(pclen, pclift, pb[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pblen, pblift, -pc[1], xy2terms);
    bclen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bc);

    xy1len = d_scale_expansion_zeroelim(pdlen, pdlift, pc[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pclen, pclift, -pd[1], xy2terms);
    cdlen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cd);

    xy1len = d_scale_expansion_zeroelim(palen, palift, pd[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pdlen, pdlift, -pa[1], xy2terms);
    dalen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, da);

    xy1len = d_scale_expansion_zeroelim(pclen, pclift, pa[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(palen, palift, -pc[1], xy2terms);
    aclen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ac);

    xy1len = d_scale_expansion_zeroelim(pdlen, pdlift, pb[1], xy1terms);
    xy2len = d_scale_expansion_zeroelim(pblen, pblift, -pd[1], xy2terms);
    bdlen = d_fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bd);

    templen = d_fast_expansion_sum_zeroelim(cdlen, cd, dalen, da, temp48);
    cdalen = d_fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, cda);
    templen = d_fast_expansion_sum_zeroelim(dalen, da, ablen, ab, temp48);
    dablen = d_fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, dab);

    for (int i = 0; i < bdlen; i++)
        bd[i] = -bd[i];
    for (int i = 0; i < aclen; i++)
        ac[i] = -ac[i];

    templen = d_fast_expansion_sum_zeroelim(ablen, ab, bclen, bc, temp48);
    abclen = d_fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, abc);
    templen = d_fast_expansion_sum_zeroelim(bclen, bc, cdlen, cd, temp48);
    bcdlen = d_fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, bcd);

    alen = d_scale_expansion_zeroelim(bcdlen, bcd, pa[0], adet);
    blen = d_scale_expansion_zeroelim(cdalen, cda, -pb[0], bdet);
    clen = d_scale_expansion_zeroelim(dablen, dab, pc[0], cdet);
    dlen = d_scale_expansion_zeroelim(abclen, abc, -pd[0], ddet);

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

    return d_fast_expansion_sum_sign(ablen, abdet, cdlen, cddet);
}


__device__ REAL orient3d_Lifted(const REAL* predConsts, REAL* predData, const REAL* pa, const REAL* pb, const REAL* pc, const REAL* pd,bool lifted){
    REAL adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL det;
    REAL permanent, errbound;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];
    adz = pa[2] - pd[2];
    bdz = pb[2] - pd[2];
    cdz = pc[2] - pd[2];

    if (lifted)
    {
        adz = adx * adx + ady * ady + adz * adz;
        bdz = bdx * bdx + bdy * bdy + bdz * bdz;
        cdz = cdx * cdx + cdy * cdy + cdz * cdz;
    }

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;

    det = adz * (bdxcdy - cdxbdy)
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
        + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
        + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);

    if (lifted)
        errbound = predConsts[o3derrboundAlifted] * permanent;
    else
        errbound = predConsts[o3derrboundA] * permanent;

    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return orient3dExact_Lifted(predConsts, predData, pa, pb, pc, pd, lifted);
}

__noinline__ __device__ void two_mult_sub(const REAL* pa, const REAL* pb, REAL* ab){
   REAL axby1, axby0, bxay1, bxay0;

   REAL bvirt;
   REAL avirt, bround, around;
   REAL _i, _j;
   REAL _0;

    Two_Product(pa[0], pb[1], axby1, axby0);
    Two_Product(pb[0], pa[1], bxay1, bxay0);
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);
}

__noinline__ __device__ int mult_add(REAL* a,int lena, REAL fa, REAL* b, int lenb, REAL fb, REAL* c, int lenc, REAL fc, REAL* tempa, REAL* tempb, REAL* tempab, REAL* ret){
    int tempalen = d_scale_expansion_zeroelim(lena, a, fa, tempa);
    int tempblen = d_scale_expansion_zeroelim(lenb, b, fb, tempb);
    int tempablen = d_fast_expansion_sum_zeroelim(tempalen, tempa, tempblen, tempb, tempab);
    tempalen = d_scale_expansion_zeroelim(lenc, c, fc, tempa);
    return d_fast_expansion_sum_zeroelim(tempalen, tempa, tempablen, tempab, ret);
}

__noinline__ __device__ int calc_det(REAL* a, int alen, REAL* b, int blen, REAL* c, int clen, REAL* d, int dlen, const REAL* f, REAL* temp1a, REAL* temp1b, REAL* temp2, REAL* temp8x, REAL* temp8y, REAL* temp8z, REAL* temp16, REAL* ret24){
    int temp1alen = d_fast_expansion_sum_zeroelim(alen, a, blen, b, temp1a);
    int temp1blen = d_fast_expansion_sum_zeroelim(clen, c, dlen, d, temp1b);
    for (int i = 0; i < temp1blen; i++) {
        temp1b[i] = -temp1b[i];
    }
    int temp2len = d_fast_expansion_sum_zeroelim(temp1alen, temp1a, temp1blen, temp1b, temp2);
    int xlen = d_squared_scale_expansion_zeroelim(temp2len, temp2, f[0], temp8x);
    int ylen = d_squared_scale_expansion_zeroelim(temp2len, temp2, f[1], temp8y);
    int len = d_fast_expansion_sum_zeroelim(xlen, temp8x, ylen, temp8y, temp16);
    int zlen = d_squared_scale_expansion_zeroelim(temp2len, temp2, f[2], temp8z);

    return d_fast_expansion_sum_zeroelim(len, temp16, zlen, temp8z, ret24);
}

__noinline__ __device__ int calc_det_adapt(REAL* a, int alen, const REAL x, const REAL y, const REAL z, REAL* tempx, REAL* tempy, REAL* tempz, REAL* temp2, REAL* ret3){
    int xlen = d_squared_scale_expansion_zeroelim(alen, a, x, tempx);
    int ylen = d_squared_scale_expansion_zeroelim(alen, a, y, tempy);
    int zlen = d_squared_scale_expansion_zeroelim(alen, a, z, tempz);
    int len = d_fast_expansion_sum_zeroelim(xlen, tempx, ylen, tempy, temp2);

    return d_fast_expansion_sum_zeroelim(len, temp2, zlen, tempz, ret3);
}

__device__ REAL insphereExact(const REAL* predConsts, REAL* predData, const REAL* pa, const REAL* pb, const REAL* pc, const REAL* pd, const REAL* pe){
    // Index into global memory
   REAL* temp96 = predData; predData += Temp96Size;     //  abcd, bcde, cdea, deab, eabc;
   REAL* det384x = predData; predData += Det384xSize;
   REAL* det384y = predData; predData += Det384ySize;
   REAL* detxy = predData; predData += DetxySize;
   REAL* adet = predData; predData += AdetSize;
   REAL* bdet = predData; predData += BdetSize;
   REAL* abdet = predData; predData += AbdetSize;
   REAL* cdedet = predData;
   REAL* det384z = det384x;
   REAL* cdet = adet;
   REAL* ddet = bdet;
   REAL* edet = adet;
   REAL* cddet = abdet;
   REAL* temp48a = det384x;
   REAL* temp48b = det384y;

   REAL ab[4], bc[4], cd[4], de[4], ea[4];
   REAL ac[4], bd[4], ce[4], da[4], eb[4];
   REAL temp8a[8], temp8b[8], temp16[16];
   REAL abc[24], bcd[24], cde[24], dea[24], eab[24];
   REAL abd[24], bce[24], cda[24], deb[24], eac[24];
    int abclen, bcdlen, cdelen, dealen, eablen;
    int abdlen, bcelen, cdalen, deblen, eaclen;
    int alen, blen, clen, dlen, elen;
    int ablen, cdlen, cd_elen;

    // (pa[0] * pb[1]) # (pb[0] * pa[1]) => ab[0..3]
    two_mult_sub(pa, pb, ab);
    two_mult_sub(pb, pc, bc);
    two_mult_sub(pc, pd, cd);
    two_mult_sub(pd, pe, de);
    two_mult_sub(pe, pa, ea);
    two_mult_sub(pa, pc, ac);
    two_mult_sub(pb, pd, bd);
    two_mult_sub(pc, pe, ce);
    two_mult_sub(pd, pa, da);
    two_mult_sub(pe, pb, eb);

    // pa[2] # pb[2] # pc[2] => abc[24]
    abclen = mult_add(bc, 4, pa[2], ac, 4, -pb[2], ab, 4, pc[2], temp8a, temp8b, temp16, abc);
    bcdlen = mult_add(cd, 4, pb[2], bd, 4, -pc[2], bc, 4, pd[2], temp8a, temp8b, temp16, bcd);
    cdelen = mult_add(de, 4, pc[2], ce, 4, -pd[2], cd, 4, pe[2], temp8a, temp8b, temp16, cde);
    dealen = mult_add(ea, 4, pd[2], da, 4, -pe[2], de, 4, pa[2], temp8a, temp8b, temp16, dea);
    eablen = mult_add(ab, 4, pe[2], eb, 4, -pa[2], ea, 4, pb[2], temp8a, temp8b, temp16, eab);
    abdlen = mult_add(bd, 4, pa[2], da, 4, pb[2], ab, 4, pd[2], temp8a, temp8b, temp16, abd);
    bcelen = mult_add(ce, 4, pb[2], eb, 4, pc[2], bc, 4, pe[2], temp8a, temp8b, temp16, bce);
    cdalen = mult_add(da, 4, pc[2], ac, 4, pd[2], cd, 4, pa[2], temp8a, temp8b, temp16, cda);
    deblen = mult_add(eb, 4, pd[2], bd, 4, pe[2], de, 4, pb[2], temp8a, temp8b, temp16, deb);
    eaclen = mult_add(ac, 4, pe[2], ce, 4, pa[2], ea, 4, pc[2], temp8a, temp8b, temp16, eac);

    ///////////////////////////

    // => deab
    clen = calc_det(eab, eablen, deb, deblen, abd, abdlen, dea, dealen, pc,
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, cdet);

    // => eabc
    dlen = calc_det(abc, abclen, eac, eaclen, bce, bcelen, eab, eablen, pd,
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, ddet);

    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

    // => abcd
    elen = calc_det(bcd, bcdlen, abd, abdlen, cda, cdalen, abc, abclen, pe,
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, edet);

    cd_elen = d_fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet);

    // => bcde
    alen = calc_det(cde, cdelen, bce, bcelen, deb, deblen, bcd, bcdlen, pa,
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, adet);

    // => cdea
    blen = calc_det(dea, dealen, cda, cdalen, eac, eaclen, cde, cdelen, pb,
        temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, bdet);

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

   REAL ret = d_fast_expansion_sum_sign(ablen, abdet, cd_elen, cdedet);

    return ret;
}

__device__ REAL insphereAdaptExact(const REAL* predConsts, REAL* predData, const REAL* pa, const REAL* pb, const REAL* pc, const REAL* pd, const REAL* pe, const REAL permanent){
   REAL* tempData = predData;

    // Index into global memory
   REAL* xdet = tempData; tempData += 96;
   REAL* ydet = tempData; tempData += 96;
   REAL* zdet = tempData; tempData += 96;
   REAL* xydet = tempData; tempData += 192;
   REAL* adet = tempData; tempData += 288;
   REAL* bdet = tempData; tempData += 288;
   REAL* abdet = tempData; tempData += 576;
   REAL* cddet = tempData;
   REAL* cdet = adet;
   REAL* ddet = bdet;

   REAL aex, bex, cex, dex, aey, bey, cey, dey, aez, bez, cez, dez;
   REAL det, errbound;

   REAL aexbey1, bexaey1, bexcey1, cexbey1;
   REAL cexdey1, dexcey1, dexaey1, aexdey1;
   REAL aexcey1, cexaey1, bexdey1, dexbey1;
   REAL aexbey0, bexaey0, bexcey0, cexbey0;
   REAL cexdey0, dexcey0, dexaey0, aexdey0;
   REAL aexcey0, cexaey0, bexdey0, dexbey0;
   REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
   REAL ab3, bc3, cd3, da3, ac3, bd3;
   REAL abeps, bceps, cdeps, daeps, aceps, bdeps;
   REAL temp8a[8], temp8b[8], temp16[16], temp24[24];
    int temp24len;

    int alen, blen, clen, dlen;
    int ablen, cdlen;

   REAL aextail, bextail, cextail, dextail;
   REAL aeytail, beytail, ceytail, deytail;
   REAL aeztail, beztail, ceztail, deztail;

   REAL bvirt;
   REAL avirt, bround, around;
   REAL _i, _j;
   REAL _0;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    Two_Product(aex, bey, aexbey1, aexbey0);
    Two_Product(bex, aey, bexaey1, bexaey0);
    Two_Two_Diff(aexbey1, aexbey0, bexaey1, bexaey0, ab3, ab[2], ab[1], ab[0]);
    ab[3] = ab3;

    Two_Product(bex, cey, bexcey1, bexcey0);
    Two_Product(cex, bey, cexbey1, cexbey0);
    Two_Two_Diff(bexcey1, bexcey0, cexbey1, cexbey0, bc3, bc[2], bc[1], bc[0]);
    bc[3] = bc3;

    Two_Product(cex, dey, cexdey1, cexdey0);
    Two_Product(dex, cey, dexcey1, dexcey0);
    Two_Two_Diff(cexdey1, cexdey0, dexcey1, dexcey0, cd3, cd[2], cd[1], cd[0]);
    cd[3] = cd3;

    Two_Product(dex, aey, dexaey1, dexaey0);
    Two_Product(aex, dey, aexdey1, aexdey0);
    Two_Two_Diff(dexaey1, dexaey0, aexdey1, aexdey0, da3, da[2], da[1], da[0]);
    da[3] = da3;

    Two_Product(aex, cey, aexcey1, aexcey0);
    Two_Product(cex, aey, cexaey1, cexaey0);
    Two_Two_Diff(aexcey1, aexcey0, cexaey1, cexaey0, ac3, ac[2], ac[1], ac[0]);
    ac[3] = ac3;

    Two_Product(bex, dey, bexdey1, bexdey0);
    Two_Product(dex, bey, dexbey1, dexbey0);
    Two_Two_Diff(bexdey1, bexdey0, dexbey1, dexbey0, bd3, bd[2], bd[1], bd[0]);
    bd[3] = bd3;

    temp24len = mult_add(cd, 4, -bez, bd, 4, cez, bc, 4, -dez, temp8a, temp8b, temp16, temp24);
    alen = calc_det_adapt(temp24, temp24len, aex, aey, aez, xdet, ydet, zdet, xydet, adet);

    temp24len = mult_add(da, 4, cez, ac, 4, dez, cd, 4, aez, temp8a, temp8b, temp16, temp24);
    blen = calc_det_adapt(temp24, temp24len, bex, bey, bez, xdet, ydet, zdet, xydet, bdet);

    ablen = d_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

    temp24len = mult_add(ab, 4, -dez, bd, 4, -aez, da, 4, -bez, temp8a, temp8b, temp16, temp24);
    clen = calc_det_adapt(temp24, temp24len, cex, cey, cez, xdet, ydet, zdet, xydet, cdet);

    temp24len = mult_add(bc, 4, aez, ac, 4, -bez, ab, 4, cez, temp8a, temp8b, temp16, temp24);
    dlen = calc_det_adapt(temp24, temp24len, dex, dey, dez, xdet, ydet, zdet, xydet, ddet);

    cdlen = d_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
    det = d_fast_expansion_sum_estimate(ablen, abdet, cdlen, cddet);

    errbound = predConsts[isperrboundB] * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa[0], pe[0], aex, aextail);
    Two_Diff_Tail(pa[1], pe[1], aey, aeytail);
    Two_Diff_Tail(pa[2], pe[2], aez, aeztail);
    Two_Diff_Tail(pb[0], pe[0], bex, bextail);
    Two_Diff_Tail(pb[1], pe[1], bey, beytail);
    Two_Diff_Tail(pb[2], pe[2], bez, beztail);
    Two_Diff_Tail(pc[0], pe[0], cex, cextail);
    Two_Diff_Tail(pc[1], pe[1], cey, ceytail);
    Two_Diff_Tail(pc[2], pe[2], cez, ceztail);
    Two_Diff_Tail(pd[0], pe[0], dex, dextail);
    Two_Diff_Tail(pd[1], pe[1], dey, deytail);
    Two_Diff_Tail(pd[2], pe[2], dez, deztail);
    if ((aextail == 0.0) && (aeytail == 0.0) && (aeztail == 0.0)
        && (bextail == 0.0) && (beytail == 0.0) && (beztail == 0.0)
        && (cextail == 0.0) && (ceytail == 0.0) && (ceztail == 0.0)
        && (dextail == 0.0) && (deytail == 0.0) && (deztail == 0.0)) {
        return det;
    }

    errbound = predConsts[isperrboundC] * permanent + predConsts[resulterrbound] * Absolute(det);
    abeps = (aex * beytail + bey * aextail)
        - (aey * bextail + bex * aeytail);
    bceps = (bex * ceytail + cey * bextail)
        - (bey * cextail + cex * beytail);
    cdeps = (cex * deytail + dey * cextail)
        - (cey * dextail + dex * ceytail);
    daeps = (dex * aeytail + aey * dextail)
        - (dey * aextail + aex * deytail);
    aceps = (aex * ceytail + cey * aextail)
        - (aey * cextail + cex * aeytail);
    bdeps = (bex * deytail + dey * bextail)
        - (bey * dextail + dex * beytail);
    det += (((bex * bex + bey * bey + bez * bez)
        * ((cez * daeps + dez * aceps + aez * cdeps)
            + (ceztail * da3 + deztail * ac3 + aeztail * cd3))
        + (dex * dex + dey * dey + dez * dez)
        * ((aez * bceps - bez * aceps + cez * abeps)
            + (aeztail * bc3 - beztail * ac3 + ceztail * ab3)))
        - ((aex * aex + aey * aey + aez * aez)
            * ((bez * cdeps - cez * bdeps + dez * bceps)
                + (beztail * cd3 - ceztail * bd3 + deztail * bc3))
            + (cex * cex + cey * cey + cez * cez)
            * ((dez * abeps + aez * bdeps + bez * daeps)
                + (deztail * ab3 + aeztail * bd3 + beztail * da3))))
        + 2.0 * (((bex * bextail + bey * beytail + bez * beztail)
            * (cez * da3 + dez * ac3 + aez * cd3)
            + (dex * dextail + dey * deytail + dez * deztail)
            * (aez * bc3 - bez * ac3 + cez * ab3))
            - ((aex * aextail + aey * aeytail + aez * aeztail)
                * (bez * cd3 - cez * bd3 + dez * bc3)
                + (cex * cextail + cey * ceytail + cez * ceztail)
                * (dez * ab3 + aez * bd3 + bez * da3)));

    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return insphereExact(predConsts, predData, pa, pb, pc, pd, pe);
}

__device__ REAL insphere(REAL* predConsts, REAL* predData, REAL3& pa, REAL3& pb, REAL3& pc, REAL3& pd, REAL3& pe, REAL* detpre, REAL* boundpre)
{
    REAL aex, aey, aez, alift;
    REAL det;

    aex = pa.x - pe.x;
    aey = pa.y - pe.y;
    aez = pa.z - pe.z;
    alift = aex * aex + aey * aey + aez * aez;

    det = (aex * detpre[2] - aey * detpre[1]) + (aez * detpre[0] - alift * detpre[3]);

    REAL aexplus, aeyplus, aezplus;
    REAL permanent, errbound;

    aexplus = Absolute(aex);
    aeyplus = Absolute(aey);
    aezplus = Absolute(aez);

    permanent = boundpre[0] * alift + boundpre[1] * aezplus + boundpre[2] * aeyplus + boundpre[3] * aexplus;
    errbound = predConsts[isperrboundA] * permanent;
    if (det > errbound || -det > errbound)
        return det;

    return insphereAdaptExact(predConsts, predData, &pa.x, &pb.x, &pc.x, &pd.x, &pe.x, permanent);

}

template <typename T3, typename T = decltype(T3::x)>
__device__ void insphereprepare0(T3& pc, T3& pd, T3& pe, T* detpre, T* boundpre) {
    T cex, dex;
    T cey, dey;
    T cez, dez;
    T cexdey, dexcey, cexdez, dexcez, ceydez, deycez;
    T cexdlift, dexclift, ceydlift, deyclift, cezdlift, dezclift;
    T clift, dlift;
    T xy, xz, yz, xl, yl, zl;


    cex = pc.x - pe.x;
    dex = pd.x - pe.x;
    cey = pc.y - pe.y;
    dey = pd.y - pe.y;
    cez = pc.z - pe.z;
    dez = pd.z - pe.z;

    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;


    cexdey = cex * dey;
    dexcey = dex * cey;
    xy = cexdey - dexcey;
    cexdez = cex * dez;
    dexcez = dex * cez;
    xz = cexdez - dexcez;
    ceydez = cey * dez;
    deycez = dey * cez;
    yz = ceydez - deycez;

    cexdlift = cex * dlift;
    dexclift = dex * clift;
    xl = cexdlift - dexclift;
    ceydlift = cey * dlift;
    deyclift = dey * clift;
    yl = ceydlift - deyclift;
    cezdlift = cez * dlift;
    dezclift = dez * clift;
    zl = cezdlift - dezclift;

    detpre[0] = xy;
    detpre[1] = yz;
    detpre[2] = xz;
    detpre[3] = xl;
    detpre[4] = yl;
    detpre[5] = zl;


    T cexdeyplus, dexceyplus, cexdezplus, dexcezplus, ceydezplus, deycezplus;
    T cexdliftplus, dexcliftplus, ceydliftplus, deycliftplus, cezdliftplus, dezcliftplus;



    cexdeyplus = Absolute(cexdey);
    dexceyplus = Absolute(dexcey);
    cexdezplus = Absolute(cexdez);
    dexcezplus = Absolute(dexcez);
    ceydezplus = Absolute(ceydez);
    deycezplus = Absolute(deycez);
    cexdliftplus = Absolute(cexdlift);
    dexcliftplus = Absolute(dexclift);
    ceydliftplus = Absolute(ceydlift);
    deycliftplus = Absolute(deyclift);
    cezdliftplus = Absolute(cezdlift);
    dezcliftplus = Absolute(dezclift);

    boundpre[0] = cexdeyplus + dexceyplus;
    boundpre[1] = ceydezplus + deycezplus;
    boundpre[2] = cexdezplus + dexcezplus;
    boundpre[3] = cexdliftplus + dexcliftplus;
    boundpre[4] = ceydliftplus + deycliftplus;
    boundpre[5] = cezdliftplus + dezcliftplus;
}

template <typename T3, typename T = decltype(T3::x)>
__device__ void insphereprepare(T3& be, T* detpre0, T* boundpre0, T* detpre, T* boundpre) {


    T blift = be.x * be.x + be.y * be.y + be.z * be.z;

    detpre[0] = be.x * detpre0[4] - detpre0[3] * be.y + blift * detpre0[0];
    detpre[1] = be.x * detpre0[5] - detpre0[3] * be.z + blift * detpre0[2];
    detpre[2] = be.y * detpre0[5] - detpre0[4] * be.z + blift * detpre0[1];
    detpre[3] = be.x * detpre0[1] - be.y * detpre0[2] + be.z * detpre0[0];

    T bexplus, beyplus, bezplus;

    bexplus = Absolute(be.x);
    beyplus = Absolute(be.y);
    bezplus = Absolute(be.z);

    boundpre[0] = boundpre0[0] * bezplus + boundpre0[2] * beyplus + boundpre0[1] * bexplus;
    boundpre[1] = boundpre0[0] * blift + boundpre0[3] * beyplus + boundpre0[4] * bexplus;
    boundpre[2] = boundpre0[2] * blift + boundpre0[3] * bezplus + boundpre0[5] * bexplus;
    boundpre[3] = boundpre0[1] * blift + boundpre0[4] * bezplus + boundpre0[5] * beyplus;

}

template <typename T3, typename T = decltype(T3::x)>
__device__  void insphereprepare(T3& pb, T3& pc, T3& pd, T3& pe, T* detpre, T* boundpre) {
    T bex, cex, dex;
    T bey, cey, dey;
    T bez, cez, dez;
    T cexdey, dexcey, cexdez, dexcez, ceydez, deycez;
    T cexdlift, dexclift, ceydlift, deyclift, cezdlift, dezclift;
    T blift, clift, dlift;
    T xy, xz, yz, xl, yl, zl;

    bex = pb.x - pe.x;
    cex = pc.x - pe.x;
    dex = pd.x - pe.x;
    bey = pb.y - pe.y;
    cey = pc.y - pe.y;
    dey = pd.y - pe.y;
    bez = pb.z - pe.z;
    cez = pc.z - pe.z;
    dez = pd.z - pe.z;

    blift = bex * bex + bey * bey + bez * bez;
    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;


    cexdey = cex * dey;
    dexcey = dex * cey;
    xy = cexdey - dexcey;
    cexdez = cex * dez;
    dexcez = dex * cez;
    xz = cexdez - dexcez;
    ceydez = cey * dez;
    deycez = dey * cez;
    yz = ceydez - deycez;

    cexdlift = cex * dlift;
    dexclift = dex * clift;
    xl = cexdlift - dexclift;
    ceydlift = cey * dlift;
    deyclift = dey * clift;
    yl = ceydlift - deyclift;
    cezdlift = cez * dlift;
    dezclift = dez * clift;
    zl = cezdlift - dezclift;

    detpre[0] = bex * yl - xl * bey + blift * xy;
    detpre[1] = bex * zl - xl * bez + blift * xz;
    detpre[2] = bey * zl - yl * bez + blift * yz;
    detpre[3] = bex * yz - bey * xz + bez * xy;

    T bexplus, beyplus, bezplus;
    T cexdeyplus, dexceyplus, cexdezplus, dexcezplus, ceydezplus, deycezplus;
    T cexdliftplus, dexcliftplus, ceydliftplus, deycliftplus, cezdliftplus, dezcliftplus;

    bexplus = Absolute(bex);
    beyplus = Absolute(bey);
    bezplus = Absolute(bez);

    cexdeyplus = Absolute(cexdey);
    dexceyplus = Absolute(dexcey);
    cexdezplus = Absolute(cexdez);
    dexcezplus = Absolute(dexcez);
    ceydezplus = Absolute(ceydez);
    deycezplus = Absolute(deycez);
    cexdliftplus = Absolute(cexdlift);
    dexcliftplus = Absolute(dexclift);
    ceydliftplus = Absolute(ceydlift);
    deycliftplus = Absolute(deyclift);
    cezdliftplus = Absolute(cezdlift);
    dezcliftplus = Absolute(dezclift);

    boundpre[0] = (cexdeyplus + dexceyplus) * bezplus + (cexdezplus + dexcezplus) * beyplus + (ceydezplus + deycezplus) * bexplus;
    boundpre[1] = (cexdeyplus + dexceyplus) * blift + (cexdliftplus + dexcliftplus) * beyplus + (ceydliftplus + deycliftplus) * bexplus;
    boundpre[2] = (cexdezplus + dexcezplus) * blift + (cexdliftplus + dexcliftplus) * bezplus + (cezdliftplus + dezcliftplus) * bexplus;
    boundpre[3] = (ceydezplus + deycezplus) * blift + (ceydliftplus + deycliftplus) * bezplus + (cezdliftplus + dezcliftplus) * beyplus;

}

__device__ REAL insphere(REAL* predConsts, REAL* predData, REAL3& pa, REAL3& pb, REAL3& pc, REAL3& pd, REAL3& pe)
{
    REAL detpre[4];
    REAL boundpre[4];
    insphereprepare(pb, pc, pd, pe, detpre, boundpre);
    return insphere(predConsts, predData, pa, pb, pc, pd, pe, detpre, boundpre);
}

template <typename T3, typename T = decltype(T3::x)>
__device__  bool inspherefast(T* predConsts, T3& pa, T3& pe, T* detpre, T* boundpre, bool* tag)
{
    T aex, aey, aez, alift;
    T det;

    aex = pa.x - pe.x;
    aey = pa.y - pe.y;
    aez = pa.z - pe.z;
    alift = aex * aex + aey * aey + aez * aez;

    det = (aex * detpre[2] - aey * detpre[1]) + (aez * detpre[0] - alift * detpre[3]);

    T errbound;

    errbound = predConsts[isperrboundA] * (boundpre[0] * alift + boundpre[1] * Absolute(aez) + boundpre[2] * Absolute(aey) + boundpre[3] * Absolute(aex));
    if ((det <= errbound) && (det >= -errbound))
        *tag = false;

    if (det < -errbound)
        return true;

    return false;

}

template <typename T3, typename T = decltype(T3::x)>
__device__  bool inspherefast(T* predConsts, T3& pa, T3& pb, T3& pc, T3& pd, T3& pe, bool* tag)
{
    T detpre[4];
    T boundpre[4];
    insphereprepare(pb, pc, pd, pe, detpre, boundpre);
    return inspherefast(predConsts, pa, pe, detpre, boundpre, tag);
}

__device__ REAL InSphereSoSOnly(const REAL* predConsts, REAL* predData, int indices[5], REAL3& pa, REAL3& pb, REAL3& pc, REAL3& pd, REAL3& pe) {


    // Sort indices & note their order
    const REAL* ip[5] = { &pa.x, &pb.x, &pc.x, &pd.x, &pe.x };
    int swaps = 0; // Record the total number of swaps.

    if (indices[0] > indices[4]) { swap(indices[0], indices[4]); swap(ip[0], ip[4]); ++swaps; }
    if (indices[1] > indices[3]) { swap(indices[1], indices[3]); swap(ip[1], ip[3]); ++swaps; }
    if (indices[0] > indices[2]) { swap(indices[0], indices[2]); swap(ip[0], ip[2]); ++swaps; }
    if (indices[2] > indices[4]) { swap(indices[2], indices[4]); swap(ip[2], ip[4]); ++swaps; }
    if (indices[0] > indices[1]) { swap(indices[0], indices[1]); swap(ip[0], ip[1]); ++swaps; }
    if (indices[2] > indices[3]) { swap(indices[2], indices[3]); swap(ip[2], ip[3]); ++swaps; }
    if (indices[1] > indices[4]) { swap(indices[1], indices[4]); swap(ip[1], ip[4]); ++swaps; }
    if (indices[1] > indices[2]) { swap(indices[1], indices[2]); swap(ip[1], ip[2]); ++swaps; }
    if (indices[3] > indices[4]) { swap(indices[3], indices[4]); swap(ip[3], ip[4]); ++swaps; }

    // Note: This is a check placed only to overcome a CUDA compiler bug!
    // As of CUDA 4 and sm_21, this bug happens if there are many SoS checks
    // caused by co-planar input points. Comment this if you wish, but do NOT
    // delete it, so it can be turned on when needed.
    //if ( ( p0 == p1 ) || ( p1 == p2 ) || ( p2 == p3 ) || ( p3 == p4 ) )
    //    printf( "Duplicate!\n" );

    /*CudaAssert((pi0 != pi1) && (pi0 != pi2) && (pi0 != pi3) && (pi0 != pi4)
        && (pi1 != pi2) && (pi1 != pi3) && (pi1 != pi4)
        && (pi2 != pi3) && (pi2 != pi4)
        && (pi3 != pi4)
        && "Duplicate indices in SoS orientation!");*/

    // Calculate determinants

    REAL op[4][3];
    REAL det = 0;
    int depth = 0;
    // Setup determinants
    while (det == 0)
    {
        ++depth;    // Increment depth, begins from 1

        bool lifted = false;

        switch (depth)
        {
        case 0:
            break;

        case 1:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 2: lifted = true;
            //op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            //op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            //op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            //op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 3: lifted = true;
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];    op[0][2] = ip[1][1];
            /*op[1][0] = ip[2][0];*/    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            /*op[2][0] = ip[3][0];*/    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            /*op[3][0] = ip[4][0];*/    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 4: lifted = true;
            // We know this is hit sometime
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/    op[0][2] = ip[1][0];
            op[1][0] = ip[2][1];    /*op[1][1] = ip[2][2];*/    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    /*op[2][1] = ip[3][2];*/    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    /*op[3][1] = ip[4][2];*/    op[3][2] = ip[4][0];
            break;

        case 5:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 6:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 7:
            /*op[0][0] = ip[2][0];*/    op[0][1] = ip[2][2];
            /*op[1][0] = ip[3][0];*/    op[1][1] = ip[3][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 8:
            op[0][0] = ip[2][1];    /*op[0][1] = ip[2][2];*/
            op[1][0] = ip[3][1];    /*op[1][1] = ip[3][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 9: lifted = true;
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 10: lifted = true;
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 11: lifted = true;
            op[0][0] = ip[2][1];    op[0][1] = ip[2][0];    /*op[0][2] = ip[2][2];*/
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    /*op[1][2] = ip[3][2];*/
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    /*op[2][2] = ip[4][2];*/
            break;

        case 12: lifted = true;
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 13: lifted = true;
            op[0][0] = ip[2][2];    op[0][1] = ip[2][0];    op[0][2] = ip[2][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 14: lifted = true;
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 15:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 16:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 17:
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];
            /*op[1][0] = ip[3][0];*/    op[1][1] = ip[3][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 18:
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/
            op[1][0] = ip[3][1];    /*op[1][1] = ip[3][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 19:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 20:
            op[0][0] = ip[3][0];
            op[1][0] = ip[4][0];
            break;

        case 21:
            op[0][0] = ip[3][1];
            op[1][0] = ip[4][1];
            break;

        case 22:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 23:
            op[0][0] = ip[3][2];
            op[1][0] = ip[4][2];
            break;

        case 24:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 25: lifted = true;
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 26: lifted = true;
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 27: lifted = true;
            op[0][0] = ip[1][1];    op[0][1] = ip[1][0];    /*op[0][2] = ip[1][2];*/
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    /*op[1][2] = ip[3][2];*/
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    /*op[2][2] = ip[4][2];*/
            break;

        case 28: lifted = true;
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 29: lifted = true;
            op[0][0] = ip[3][0];    op[0][1] = ip[3][1];    op[0][2] = ip[3][2];
            op[1][0] = ip[4][0];    op[1][1] = ip[4][1];    op[1][2] = ip[4][2];
            break;

        case 30: lifted = true;
            op[0][0] = ip[0][1];    op[0][1] = ip[0][0];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;

        case 31: lifted = true;
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][2];    op[1][2] = ip[1][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 32: lifted = true;
            op[0][0] = ip[1][2];    op[0][1] = ip[1][0];    op[0][2] = ip[1][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 33: lifted = true;
            op[0][0] = ip[0][2];    op[0][1] = ip[0][0];    op[0][2] = ip[0][1];
            //op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            //op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 34: lifted = true;
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[1][1];    op[1][1] = ip[1][2];    op[1][2] = ip[1][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 35:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[2][0];    op[2][1] = ip[2][1];    op[2][2] = ip[2][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 36:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 37:
            /*op[0][0] = ip[1][0];*/    op[0][1] = ip[1][2];
            /*op[1][0] = ip[2][0];*/    op[1][1] = ip[2][2];
            /*op[2][0] = ip[4][0];*/    op[2][1] = ip[4][2];
            break;

        case 38:
            op[0][0] = ip[1][1];    /*op[0][1] = ip[1][2];*/
            op[1][0] = ip[2][1];    /*op[1][1] = ip[2][2];*/
            op[2][0] = ip[4][1];    /*op[2][1] = ip[4][2];*/
            break;

        case 39:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 40:
            op[0][0] = ip[2][0];
            op[1][0] = ip[4][0];
            break;

        case 41:
            op[0][0] = ip[2][1];
            op[1][0] = ip[4][1];
            break;

        case 42:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 43:
            op[0][0] = ip[2][2];
            op[1][0] = ip[4][2];
            break;

        case 44:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 45:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 46:
            op[0][0] = ip[1][0];
            op[1][0] = ip[4][0];
            break;

        case 47:
            op[0][0] = ip[1][1];
            op[1][0] = ip[4][1];
            break;

        case 48:
            op[0][0] = ip[0][0];
            op[1][0] = ip[4][0];
            break;

        case 49:
            // See below for result
            break;

        default:
            break;

        }   // switch ( depth )

        // *** Calculate determinant

        switch (depth)
        {
            // 3D orientation determinant
        case 1: case 5: case 15: case 35:
            // 3D orientation involving the lifted coordinate
        case 2: case 3: case 4: case 9: case 12: case 14: case 25: case 31: case 34:
            det = orient3d_Lifted(predConsts, predData, op[0], op[1], op[2], op[3], lifted);
            break;

            // 2D orientation determinant
        case 6: case 7: case 8: case 16: case 17: case 18: case 19: case 22: case 24:
        case 36: case 37: case 38: case 39: case 42: case 44: case 45:
            // 2D orientation involving the lifted coordinate
        case 10: case 11: case 13: case 26: case 27: case 28: case 30: case 32: case 33:
            det = orient2d_Lifted(predData, op[0], op[1], op[2], lifted);
            break;

            // 1D orientation determinant
        case 20: case 21: case 23: case 40: case 41: case 43: case 46: case 47: case 48:
            // 1D orientation involving the lifted coordinate
        case 29:
            det = orient1d_Lifted(op[0], op[1], lifted);
            break;

        case 49:
            // Last depth, always POS
            det = 1;
            break;

        default:
            break;
        }
    }   

    ////
    // Flip result for certain depths. (See SoS paper.)
    ////

    switch (depth)
    {
        // -ve result
    case 1: case 3: case 7: case 9: case 11: case 14: case 16: case 15: case 18:
    case 20: case 22: case 23: case 26: case 30: case 31: case 32: case 37: case 39:
    case 41: case 44: case 46:
        det = -det;
        break;

    default:
        // Do nothing
        break;
    }

    ////
    // Flip result for odd swap count
    ////

    if ((swaps % 2) != 0)
        det = -det;
    return det;
}
