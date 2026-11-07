#include "delaunay_kernels.h"
#include "predicates.h"
#include "geometry.h"

__constant__ REAL3 infvector[8];



template <typename R3>
__device__ void Delaunay1Ring<R3>::init(int p_seed, R3* p_pts, int nbpts, int* p_cellStartIds, int dim, int* p_vertex, R* p_predConsts) {

    pts = p_pts;
    npts = nbpts;
    target_id = p_seed;
    target_coords = p_pts[p_seed];
    status = success;
    cellStartIds = p_cellStartIds;
    M = dim;
   
    vertex_data = p_vertex;
    trust_tag = true;
    predConsts = p_predConsts;

    nv = 3;   

}



template <typename T>
__device__  void intersection2d(T* center, T r1, T* lowboundr, T* upboundr, int i) {//coordinate range of the intersection of circle and unit square
    int jk[2] = { (i + 1) % 3 ,(i + 2) % 3 };
    int j = jk[0]; int k = jk[1];
    if ((center[j] + r1 > 0 || center[j] - r1 < 1) && (center[k] + r1 > 0 || center[k] - r1 < 1)) {

        for (int s = 0; s < 2; s++) {//min/max point is in the square
            j = jk[s % 2]; k = jk[(s + 1) % 2];
            if (center[j] > 0. && center[j] < 1.) {
                T bound = center[k] - r1;
                if (lowboundr[k] < 0 ||  bound < lowboundr[k])
                    lowboundr[k] = max(bound, 0.);
                bound = center[k] + r1;
                if (upboundr[k] > 1 ||  bound > upboundr[k])
                    upboundr[k] = min(bound, 1.);
            }
        }
        for (int s = 0; s < 2; s++) {//min/max point is on the square
            j = jk[s % 2]; k = jk[(s + 1) % 2];
            T updown = (center[j] > 0.5) ? 1 : 0;
            if (center[j] - r1 < updown && center[j] + r1> updown) {
                T r2 = sqrt(r1 * r1 - (center[j] - updown) * (center[j] - updown));//intersection of circle and line
                T upstate = center[k] + r2;
                T lowstate = center[k] - r2;
                if (upstate > 0. && lowstate < 1.) {
                    if (lowboundr[k] < 0 ||  lowstate < lowboundr[k])
                        lowboundr[k] = max(lowstate, 0.);
                    if (upboundr[k] > 1 || upstate > upboundr[k])
                        upboundr[k] = min(upstate,1.);
                }
            }
        }
    }
}

template <typename T>
__device__  void intersection_range(T* center, T r, T* lowboundr, T* upboundr) { //coordinate range of the intersection of ball and boundingbox(unit cube)
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;int k = (i + 2) % 3;
        if (center[j] >= 0 && center[j] <= 1 && center[k] >= 0 && center[k] <= 1) { //min/max point is in the cube    
            lowboundr[i] = max(center[i] - r, 0.);
            upboundr[i] = min(center[i] + r,1.);
        }
    }

    for (int i = 0; i < 3; i++) {//min/max point is on the cube
        T updown = (center[i] > 0.5) ? 1 : 0;//choose z=0 or z=1
        if (center[i] + r > updown && center[i] - r < updown) {
            T r1 = sqrt(r * r - (center[i] - updown) * (center[i] - updown));
            intersection2d(center, r1, lowboundr, upboundr, i);//(y,z),r1
        }
    }
}


//template <typename R3>
//__device__ void  Delaunay1Ring<R3>::cell_range(int3* lowbound, int3* upbound, int3* startcell) {
//    //we need to search the points in the cells within the range lowbound to upbound, here we calculate the range
//    R3 realcenter = ballcenter<R, R3>(target_coords, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z]);
//    R realr = norm<R, R3>(realcenter - target_coords);
//
//    if (realcenter.x < 0. || realcenter.y < 0. || realcenter.z < 0. || realcenter.x >1. || realcenter.y >1. || realcenter.z >1.) {// the center is not in the unit cube
//        double3 center = ballcenter<double, double3>(cast<R3>(target_coords), cast<R3>(pts[act_triangle.x]), cast<R3>(pts[act_triangle.y]), cast<R3>(pts[act_triangle.z]));
//        double r = norm<double, double3>(center - cast(target_coords));
//        double3 lowboundr{ -1.,-1.,-1. };
//        double3 upboundr{ 2.,2.,2. };
//
//        intersection_range<double>(&(center.x), r, &(lowboundr.x), &(upboundr.x));
//        startcell[0] = getcellidx(target_coords, M);
//        lowbound[0] = getcellidx(lowboundr, M);
//        upbound[0] = getcellidx(upboundr, M);
//    }
//    else {
//        R3 lowboundr{ realcenter.x - realr, realcenter.y - realr, realcenter.z - realr };
//        R3 upboundr{ realcenter.x + realr, realcenter.y + realr, realcenter.z + realr };
//        startcell[0] = getcellidx(realcenter, M);
//        lowbound[0] = getcellidx(lowboundr, M);
//        upbound[0] = getcellidx(upboundr, M);
//    }
//
//}

template <typename R3>
__device__ void  Delaunay1Ring<R3>::cell_range(int3* lowbound, int3* upbound, int3* startcell) {
    //we need to search the points in the cells within the range lowbound to upbound, here we calculate the range
    startcell[0] = getcellidx(target_coords, M);
    int inf_tag = 0;
    if (act_triangle.x >= npts)
        BitSet(inf_tag, 0);
    if (act_triangle.y >= npts)
        BitSet(inf_tag, 1);
    if (act_triangle.z >= npts)
        BitSet(inf_tag, 2);
    R3 lowboundr = { -1,-1,-1 };
    R3 upboundr = { 2,2,2 };
    if (inf_tag == 0) {
        R3 center = ballcenter<R, R3>(target_coords, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z]);
        R r = norm<R, R3>(center - target_coords);
        if (center.x < 0. || center.y < 0. || center.z < 0. || center.x >1. || center.y >1. || center.z >1.) {
            intersection_range<R>(&center.x, r, &lowboundr.x, &upboundr.x);
        }
        else {
            lowboundr = { center.x - r, center.y - r, center.z - r };
            upboundr = { center.x + r, center.y + r, center.z + r };
        }
        startcell[0] = getcellidx(center, M);
    }
    else {
        REAL3 p1,p2,px,py,pz;
        px = pts[act_triangle.x] - target_coords;
        py = pts[act_triangle.y] - target_coords;
        pz = pts[act_triangle.z] - target_coords;
        if (inf_tag == 1) {
            p1 = pz;
            p2 = py;
        }
        if (inf_tag == 2) {
            p1 = px;
            p2 = pz;
        }
        if (inf_tag == 3) {
            p1 = infvector[4 * (threadIdx.x % 2) + 3];
            p2 = pz;
        }
        if (inf_tag == 4) {
            p1 = py;
            p2 = px;
        }
        if (inf_tag == 5) {
            p1 = infvector[4 * (threadIdx.x % 2) + 2];
            p2 = py;
        }
        if (inf_tag == 6) {
            p1 = infvector[4 * (threadIdx.x % 2) + 1];
            p2 = px;
        }

        REAL3 normal;
        if (inf_tag == 7)          
            normal = infvector[4 * (threadIdx.x % 2)];
        else
            normal = cross3(p1, p2);
        REAL det = dot3(normal, target_coords);
        REAL4 plane = { normal.x, normal.y, normal.z, -det };
        plane_intersection(&plane.x, &lowboundr.x, &upboundr.x);
    }   
    lowbound[0] = getcellidx(lowboundr, M);
    upbound[0] = getcellidx(upboundr, M);
}


__device__ void plane_intersection(REAL* plane, REAL* lowboundr, REAL* upboundr) { //intersection of halfspace and unit cube
    //REAL updown[3];
    //for (int i = 0; i < 3; i++) {//choose the edges
    //    int j = (i + 1) % 3; int k = (i + 2) % 3;
    //    if (-(plane[j] * 0.5 + plane[k] * 0.5 + plane[3]) / plane[i] < 0.5)
    //        updown[i] = 0;
    //    else
    //        updown[i] = 1;
    //}
    //for (int i = 0; i < 3; i++) {//find the intersection points on the edges
    //    int j = (i + 1) % 3; int k = (i + 2) % 3;
    //    REAL bound = -(plane[j] * updown[j] + plane[k] * updown[k] + plane[3]) / plane[i];
    //    if (plane[i] < 0)
    //        upboundr[i] = bound;
    //    if (plane[i] > 0)
    //        lowboundr[i] = bound;
    //}
    for (int i = 0; i < 3; i++) {//find the intersection points on the edges
        int j = (i + 1) % 3; int k = (i + 2) % 3;
        REAL updownj = (plane[j] < 0) ? 0 : 1;
        REAL updownk = (plane[k] < 0) ? 0 : 1;
        REAL bound = -(plane[j] * updownj + plane[k] * updownk + plane[3]) / plane[i];
        if (plane[i] < 0)
            upboundr[i] = bound;
        if (plane[i] > 0)
            lowboundr[i] = bound;
    }
}


template <typename R3>
__device__ int Delaunay1Ring<R3>::ballsearch() {    //find a point in the circumsphere
    R detpre[4];
    R boundpre[4];
    int3 lowbound, upbound, ijk2;

    cell_range(&lowbound, &upbound, &ijk2);
    insphereprepare(pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, detpre, boundpre);

    int3 ijk3 = ijk2;
    int3 ijk0{ ijk2.x + 1, ijk2.y + 1, ijk2.z + 1 };
    int3 ijk1{ ijk2.x - 1, ijk2.y - 1, ijk2.z - 1 };
    bool inboundary = true;
    uchar numcell = 0;
 

    while (inboundary) {//from center to boundary
        for (int i = ijk2.x; i <= ijk3.x; i++) {
            for (int j = ijk2.y; j <= ijk3.y; j++) {
                for (int k = ijk2.z; k <= ijk3.z; k++) {
                    if (numcell > MAX_CELL) {// leave for post-processing
                        trust_tag = false;
                        return -1;
                    }
                    if (ijk0.x <= i && i <= ijk1.x && ijk0.y <= j && j <= ijk1.y && ijk0.z <= k && k <= ijk1.z)
                        continue;//searched in the previous

                    int id = i + j * M + k * M * M;
                    numcell++;
                    for (int c = cellStartIds[id]; c < cellStartIds[id + 1]; c++) {//the points in the cell
                        if (c == act_triangle.x || c == act_triangle.y || c == act_triangle.z || c == target_id)
                            continue;
                        R3 pt = pts[c];
                        if (inspherefast(predConsts, pt, target_coords, detpre, boundpre, &trust_tag))
                            return c;
                                                  
                }
            }
        }
    }
        ijk0 = ijk2;
        ijk1 = ijk3;
        ijk2 = make_int3(ijk2.x - 1, ijk2.y - 1, ijk2.z - 1);
        ijk3 = make_int3(ijk3.x + 1, ijk3.y + 1, ijk3.z + 1);
        if (ijk2.x<lowbound.x && ijk2.y<lowbound.y && ijk2.z<lowbound.z && ijk3.x>upbound.x && ijk3.y>upbound.y && ijk3.z>upbound.z)
            inboundary = false;
        ijk2 = vecmax(ijk2, lowbound);
        ijk3 = vecmin(ijk3, upbound);
}
    return -1;
}

template <typename R3>
__device__ bool Delaunay1Ring<R3>::compare_position(int v) {
    if (target_coords.x < pts[v].x)
        return true;
    if (target_coords.x == pts[v].x) {
        if (target_coords.y < pts[v].y)
            return true;
        if (target_coords.y == pts[v].y) {
            if (target_coords.z < pts[v].z)
                return true;
        }
    }
    return false;
}

template <typename R3>
__device__ int Delaunay1Ring<R3>::encode_index(int v) {
    if (!compare_position(v))
        return v;
    else if(v != 0)
            return -v;
        else
            return INT_MIN;
}

template <typename R3>
__device__ int Delaunay1Ring<R3>::decode_index(int v) {
    return ::decode_index(v);
}

template <typename R3>
__device__ void Delaunay1Ring<R3>::update() {
    if (nv >= MAX_P) {
        status = vertex_overflow;
        return;
    }

    if (vertex(nv - 1) < 0) 
        act_triangle.y = act_triangle.z;
    else 
        act_triangle.x = act_triangle.z;

    act_triangle.z = aim.z;
    if (!orient3dfast(predConsts, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, &trust_tag)) {
        act_triangle.z = aim.x;
        if (!orient3dfast(predConsts, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, &trust_tag)) {
            act_triangle.z = aim.y;
        }
        aim.z = act_triangle.z;
    }

   /* insphereprepare0(pts[act_triangle.x], pts[act_triangle.y], target_coords, detpre0, boundpre0);*/
    if (!trust_tag)
        status = need_exact_predict;
}

template <typename R3>
__device__ bool Delaunay1Ring<R3>::intriangle(int3 tri) {

    R t1, t2, t3;
    t1 = orient2dfast(predConsts, &pts[tri.x].x, &pts[tri.y].x, &target_coords.x, &trust_tag);
    t2 = orient2dfast(predConsts, &pts[tri.y].x, &pts[tri.z].x, &target_coords.x, &trust_tag);
    t3 = orient2dfast(predConsts, &pts[tri.z].x, &pts[tri.x].x, &target_coords.x, &trust_tag);

    return (t1 >= 0 && t2 >= 0 && t3 >= 0) || (t1 <= 0 && t2 <= 0 && t3 <= 0);
}

template <typename R3>
__device__ void Delaunay1Ring<R3>::shrink() {//shrink the first tri from inf
    int3 ijk0 = getcellidx(target_coords, M);
    int i0 = ijk0.x, j0 = ijk0.y, k0 = ijk0.z;

    int Idx[4];
    int j = 0;

    if (threadIdx.x % 2 == 0) {
        if (k0 - 1 >= 0)
            k0 = k0 - 1;
    }
    else {
        if (k0 + 1 < M)
            k0 = k0 + 1;
    }

    if (i0 + 1 < M)
        Idx[j++] = i0 + 1 + j0 * M + k0 * M * M;
    if (i0 - 1 >= 0)
        Idx[j++] = i0 - 1 + j0 * M + k0 * M * M;
    if (j0 + 1 < M)
        Idx[j++] = i0 + (j0 + 1) * M + k0 * M * M;
    if (j0 - 1 >= 0)
        Idx[j++] = i0 + (j0 - 1) * M + k0 * M * M;
   

    for (int i = 0; i < j; i++) {
        int id = Idx[i];
        for (int k = cellStartIds[id]; k < cellStartIds[id + 1]; k++) {
            if (threadIdx.x % 2 == 0) {
                if (pts[k].z >= target_coords.z)
                    continue;
            }
            else {
                if (pts[k].z <= target_coords.z)
                    continue;
            }
            if (intriangle({ k, act_triangle.y, act_triangle.z }))
                act_triangle.x = k;
            else if (intriangle({ act_triangle.x, k, act_triangle.z }))
                act_triangle.y = k;
            else
                act_triangle.z = k;
            break;
        }
    }
}

template <typename R3>
__device__ void Delaunay1Ring<R3>::reoder() {//adjust the order to make the vertex(0)+, vertex(1)-
    vertex(0) = encode_index(act_triangle.x);
    vertex(1) = encode_index(act_triangle.y);
    vertex(2) = encode_index(act_triangle.z);
    for (int i = 0; i < 3; i++) {
        if (vertex(i) >= 0 && vertex((i + 1) % 3) < 0) {
            if (i == 1) {
                int temp = act_triangle.x;
                act_triangle.x = act_triangle.y;
                act_triangle.y = act_triangle.z;
                act_triangle.z = temp;
            }
            if (i == 2) {
                int temp = act_triangle.y;
                act_triangle.y = act_triangle.x;
                act_triangle.x = act_triangle.z;
                act_triangle.z = temp;
            }
            break;
        }
    }
    vertex(0) = encode_index(act_triangle.x);
    vertex(1) = encode_index(act_triangle.y);
    vertex(2) = encode_index(act_triangle.z);

    if(act_triangle.y >= npts)
        if(act_triangle.x >= npts|| act_triangle.z >= npts)
            status = init_fail;
}


template <typename R3>
__device__ void Delaunay1Ring<R3>::dt_init() {//find the first tri   
    if(threadIdx.x % 2 == 0)
        act_triangle = { npts, npts + 1, npts + 2 };
    else
        act_triangle = { npts + 3, npts + 4, npts + 5 };
    shrink();

    while (trust_tag) {
        int idx = ballsearch();
        if (idx == -1)
            break;
        else {
            trust_tag = true;
            if (intriangle({ idx, act_triangle.y, act_triangle.z }))
                act_triangle.x = idx;
            else if (intriangle({ act_triangle.x, idx, act_triangle.z }))
                act_triangle.y = idx;
            else
                act_triangle.z = idx;
        }

    }
    if (!trust_tag) {
        status = init_fail;
    }
    reoder();
}

template <typename R3>
__device__ bool Delaunay1Ring<R3>::isfinished() {   
    if(status != success)
        return true;
    
    if (((aim.x == act_triangle.x)|| (aim.x >=npts && act_triangle.x >= npts)) && ((aim.y == act_triangle.y) || (aim.y >= npts && act_triangle.y >= npts)))
        return true;

    return false;
}

//template <typename R3>
//__device__ int Delaunay1Ring<R3>::ballsearch_post(int3 lowbound, int3 upbound, int i, R* detpre, R* boundpre) {// search a point in the circumsphere using one block
//    int j = (i + 1) % 3; int k = (i + 2) % 3;
//    int bounddet[3] = { upbound.x - lowbound.x + 1, upbound.y - lowbound.y + 1, upbound.z - lowbound.z + 1 };
//    int dim[2];
//    dim[0] = bounddet[i];
//    dim[1] = dim[0] * bounddet[j];
//
//    int ijk[3];
//    for (int det = threadIdx.x; det < dim[1] * bounddet[k]; det += BLOCK_SIZE_POST) {
//
//        ijk[k] = det / dim[1];
//        ijk[j] = (det % dim[1]) / dim[0];
//        ijk[i] = (det % dim[1]) % dim[0];
//        int id = lowbound.x + ijk[0] + (lowbound.y + ijk[1]) * M + (lowbound.z + ijk[2]) * M * M;
//#ifdef SORTING_POINTS
//        for (int c = cellStartIds[id]; c < cellStartIds[id + 1]; c++) {
//#else
//        for (int a = cellStartIds[id]; a < cellStartIds[id + 1]; a++) {
//            int c = permutation[a];
//#endif 
//            if (c == act_triangle.x || c == act_triangle.y || c == act_triangle.z || c == target_id) {
//                continue;
//            }
//            R3 pt = pts[c];
//            R predData[PredicateTotalSize];
//            R det = insphere(predConsts, predData, pt, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, detpre, boundpre);
//            if (det < 0)
//                return c;
//            if (det == 0) {
//                int indices[5] = { c,act_triangle.x ,act_triangle.y ,act_triangle.z ,target_id };
//                if (InSphereSoSOnly(predConsts, predData, indices, pt, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords) < 0)
//                    return c;
//            }
//
//        }
//        }
//    return -1;
//    }
//
//template <typename R3>
//__device__ void Delaunay1Ring<R3>::update_post() {
//    if (nv >= MAX_P) {
//        status = vertex_overflow;
//        return;
//    }
//
//    if (vertex(nv - 1) < 0)
//        act_triangle.y = act_triangle.z;
//    else
//        act_triangle.x = act_triangle.z;
//
//    act_triangle.z = aim.z;
//    if (!orient3dfast(predConsts, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, &trust_tag)) {
//        act_triangle.z = aim.x;
//        if (!orient3dfast(predConsts, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, &trust_tag)) {
//            act_triangle.z = aim.y;
//        }
//        aim.z = act_triangle.z;
//    }   
//}
//
//template <typename R3>
//__device__ bool Delaunay1Ring<R3>::intriangle_post(int3 tri) {
//    R t1, t2, t3;
//    t1 = orient2d(predConsts, &pts[tri.x].x, &pts[tri.y].x, &target_coords.x);
//    t2 = orient2d(predConsts, &pts[tri.y].x, &pts[tri.z].x, &target_coords.x);
//    t3 = orient2d(predConsts, &pts[tri.z].x, &pts[tri.x].x, &target_coords.x);
//
//    return (t1 >= 0 && t2 >= 0 && t3 >= 0) || (t1 <= 0 && t2 <= 0 && t3 <= 0);
//}
//
//template <typename R3>
//__device__ void Delaunay1Ring<R3>::shrink_post() {//shrink the first tri from inf
//    int3 ijk0 = getcellidx(target_coords, M);
//    int i0 = ijk0.x, j0 = ijk0.y, k0 = ijk0.z;
//
//    int Idx[4];
//    int j = 0;
//    if (k0 - 1 >= 0)
//        k0 = k0 - 1;
//    if (i0 + 1 < M)
//        Idx[j++] = i0 + 1 + j0 * M + k0 * M * M;
//    if (i0 - 1 >= 0)
//        Idx[j++] = i0 - 1 + j0 * M + k0 * M * M;
//    if (j0 + 1 < M)
//        Idx[j++] = i0 + (j0 + 1) * M + k0 * M * M;
//    if (j0 - 1 >= 0)
//        Idx[j++] = i0 + (j0 - 1) * M + k0 * M * M;
//
//    for (int i = 0; i < j; i++) {
//        int id = Idx[i];
//        for (int k = cellStartIds[id]; k < cellStartIds[id + 1]; k++) {
//            if (pts[k].z >= target_coords.z)
//                continue;
//            if (intriangle_post({ k, act_triangle.y, act_triangle.z }))
//                act_triangle.x = k;
//            else if (intriangle_post({ act_triangle.x, k, act_triangle.z }))
//                act_triangle.y = k;
//            else
//                act_triangle.z = k;
//            break;
//        }
//    }
//}


__device__ void  MainPhase::update() {//update the Delaunay one-ring
    dr.trust_tag = true;
    int idx = dr.ballsearch();
    num++;  
    if (idx == -1) {//there is no point in the sphere    
        if (!dr.trust_tag) {
            dr.status = need_exact_predict;
        }
        else {
            dr.vertex(dr.nv) = dr.encode_index(dr.act_triangle.z);
            dr.nv++;
            dr.update();
        }
    }
    else {
        dr.aim.z = dr.act_triangle.z;
        dr.act_triangle.z = idx;
    }
}



__global__ void DT_kernel(REAL3* pts, int nv, int startseed, REAL* predConsts, uchar* stat, DTStruct* output, int* cellStartIds, int M, uchar* exact, int* permutation, int* ncycles, clock_t* time) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed / 2 + startseed >= nv) return;
   
   /* seed = 2* 977 +seed;*/
    clock_t start = clock();
   
    MainPhase mp;
    mp.dr.init(seed / 2 + startseed, pts, nv, cellStartIds, M, &output[seed/2].vertex_data[(threadIdx.x % 2) * MAX_P], predConsts);

#ifndef SORTING_POINTS
    mp.dr.permutation = permutation;
#endif
   
    mp.dr.dt_init();

    mp.dr.aim.x = __shfl_xor_sync(0xffffffff, mp.dr.act_triangle.x, 1);
    mp.dr.aim.y = __shfl_xor_sync(0xffffffff, mp.dr.act_triangle.y, 1);
    int aimstatus = __shfl_xor_sync(0xffffffff, mp.dr.status, 1);  
    mp.dr.aim.z = mp.dr.aim.x;
    if (aimstatus != success)
        mp.dr.status = init_fail;

    mp.dr.update();
    while (!mp.dr.isfinished()) {
        mp.update();
        if (mp.num >= MAX_CYCLES)
            mp.dr.status = cycles_overflow;
    }

    /*if (threadIdx.x % 2 == 0) {
        printf("%d\n", mp.dr.nv);
        for (int i = 0; i < mp.dr.nv; i++)
            printf("%d\t", mp.dr.vertex(i));
        printf("\n");
    }*/
 
    output[seed/2].nv[threadIdx.x % 2] = mp.dr.nv;
    aimstatus = __shfl_xor_sync(0xffffffff, mp.dr.status, 1);
    if(threadIdx.x % 2 == 0)
    stat[seed / 2] = max(mp.dr.status, aimstatus);
    /*if(ncycles)
        ncycles[seed] = mp.num;
    if (time)
        time[seed] = clock() - start;*/

}




template <typename T4, typename R3>
__device__ void Delaunay1RingPost<T4, R3>::init(int p_seed, R3* p_pts, int nbpts, int* p_cellStartIds, int dim, T4* p_tr, int* p_idx, int p_maxp, R* p_predConsts) {

    pts = p_pts;
    npts = nbpts;
    target_id = p_seed;
    target_coords = p_pts[p_seed];
    status = success;
    cellStartIds = p_cellStartIds;
    M = dim;
    tr_data = p_tr;
    index_data = p_idx;
    maxp = p_maxp;
    predConsts = p_predConsts;
    nr = 0;
    noutput = 0;
}


template <typename T4, typename R3>
__device__ void  Delaunay1RingPost<T4, R3>::cell_range(int3* lowbound, int3* upbound, R3* start) {
    //we need to search the points in the cells within the range lowbound to upbound, here we calculate the range
    R3 realcenter = ballcenter<R, R3>(target_coords, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z]);
    R realr = norm<R, R3>(realcenter - target_coords);

    if (realcenter.x < 0. || realcenter.y < 0. || realcenter.z < 0. || realcenter.x >1. || realcenter.y >1. || realcenter.z >1.) {// the center is not in the unit cube
        double3 center = ballcenter<double, double3>(cast_to_double3(target_coords), cast_to_double3(pts[act_triangle.x]), cast_to_double3(pts[act_triangle.y]), cast_to_double3(pts[act_triangle.z]));
        double r = norm<double, double3>(center - cast_to_double3(target_coords));
        double3 lowboundr{ -1.,-1.,-1. };
        double3 upboundr{ 2.,2.,2. };

        intersection_range<double>(&(center.x), r, &(lowboundr.x), &(upboundr.x));
        start[0] = target_coords;
        lowbound[0] = getcellidx(lowboundr, M);
        upbound[0] = getcellidx(upboundr, M);
    }
    else {
        R3 lowboundr{ realcenter.x - realr, realcenter.y - realr, realcenter.z - realr };
        R3 upboundr{ realcenter.x + realr, realcenter.y + realr, realcenter.z + realr };
        start[0] = realcenter;
        lowbound[0] = getcellidx(lowboundr, M);
        upbound[0] = getcellidx(upboundr, M);
    }

    //if (exact_tag != 0) {//limit search range
    //    int det = 30;
    //   /* if (exact_tag == 1)
    //        det = 1;*/
    //    start[0] = target_coords;
    //    int3 startcell= getcellidx(target_coords, M);
    //    lowbound[0] = vecmax(lowbound[0], vecmax({ 0,0,0 }, startcell - make_int3(det, det, det)));
    //    upbound[0] = vecmin(upbound[0], vecmin({ M - 1,M - 1,M - 1 }, startcell + make_int3(det, det, det)));
    //}

}

template <typename T4, typename R3>
__device__ void Delaunay1RingPost<T4, R3>::add_triangle(T i, T j, T k) {
    uchar order = 0;
    if (index(i) < 0)
        order++;
    if (index(j) < 0)
        order++;
    if (index(k) < 0)
        order++;
    tr(nt) = { i, j, k, 0 };
    nt++;
    if (order == 0 || order == 3) {
        BitSet(tag(nt - 1), PROCESSED);
        return;
    }

    if (order == 2 && rindex(i) < npts && rindex(j) < npts && rindex(k) < npts) {
        BitSet(tag(nt - 1), OUTPUT);
    }
}

template <typename T4, typename R3>
__device__ int Delaunay1RingPost<T4, R3>::encode_index(int v) {
    if (!compare_position(v))
        return v;
    else if (v != 0)
        return -v;
    else
        return INT_MIN;
}

template <typename T4, typename R3>
__device__ int Delaunay1RingPost<T4, R3>::decode_index(int v) {
    return ::decode_index(v);
}

template <typename T4, typename R3>
__device__ int Delaunay1RingPost<T4, R3>::add_vertex(int vid) {
    if (nv >= maxp) {
        BitMap usedpoints;
        for (int i = 0; i < nt + nr; i++) {//mark the points that are still used in the triangle list
            usedpoints.set(tr(i).x);
            usedpoints.set(tr(i).y);
            usedpoints.set(tr(i).z);
        }
        int i = 0;
        while (i < nv) {
            if (!usedpoints.check(i)) {//the point is no longer used
                nv--;
                if (nv == i)
                    break;
                index(i) = index(nv);//move the last point to i
                if (usedpoints.check(nv)) {
                    usedpoints.set(i);
                    for (int j = 0; j < nt + nr; j++) {//change i in the triangle list to nv
                        if (tr(j).x == nv)
                            tr(j).x = i;
                        if (tr(j).y == nv)
                            tr(j).y = i;
                        if (tr(j).z == nv)
                            tr(j).z = i;
                    }
                }
                else
                    usedpoints.clear(i);
            }
            else
                i++;
        }
        if (nv >= maxp) {
            status = vertex_overflow;
            return -1;
        }
    }
    index(nv) = encode_index(vid);
    nv++;
    return nv - 1;
}

template <typename T4, typename R3>
__device__ Delaunay1RingPost<T4, R3>::T Delaunay1RingPost<T4, R3>::compute_boundary() {
    for (int i = 0; i < nv; i++)
        boundary_next[i] = (T)-1;
    T first_boundary = (T)-1;

    T t = nt;
    while (nr > 0) {
        bool is_in_border[3];
        bool next_is_opp[3];
        for (int i = 0; i < 3; i++)
            is_in_border[i] = (boundary_next[ith_plane(t, i)] != (T)-1);
        for (int i = 0; i < 3; i++)
            next_is_opp[i] = (boundary_next[ith_plane(t, (i + 1) % 3)] == ith_plane(t, i));

        bool new_border_is_simple = true;
        // check for non manifoldness
        for (int i = 0; i < 3; i++)
            if (!next_is_opp[i] && !next_is_opp[(i + 1) % 3] && is_in_border[(i + 1) % 3]) new_border_is_simple = false;

        // check for more than one boundary ... or first triangle
        if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
            if (first_boundary == (T)-1) {
                for (int i = 0; i < 3; i++)
                    boundary_next[ith_plane(t, i)] = ith_plane(t, (i + 1) % 3);
                first_boundary = tr(t).x;
            }
            else new_border_is_simple = false;
        }

        if (!new_border_is_simple) {
            t++;
            if (t == nt + nr) {
                status = need_exact_predict;
                return((T)-1);
            }
            continue;
        }

        // link next
        for (int i = 0; i < 3; i++)
            if (!next_is_opp[i]) boundary_next[ith_plane(t, i)] = ith_plane(t, (i + 1) % 3);

        // destroy link from removed vertices
        for (int i = 0; i < 3; i++)
            if (next_is_opp[i] && next_is_opp[(i + 1) % 3]) {
                if (first_boundary == ith_plane(t, (i + 1) % 3)) first_boundary = boundary_next[ith_plane(t, (i + 1) % 3)];
                boundary_next[ith_plane(t, (i + 1) % 3)] = (T)-1;
            }

        //remove triangle from R, and restart iterating on R
        tr(t) = tr(nt + nr - 1);
        t = nt;
        nr--;
    }
    return(first_boundary);
}
template <typename T4, typename R3>
__device__ bool Delaunay1RingPost<T4, R3>::compare_position(int v) {
    if (target_coords.x < pts[v].x)
        return true;
    if (target_coords.x == pts[v].x) {
        if (target_coords.y < pts[v].y)
            return true;
        if (target_coords.y == pts[v].y) {
            if (target_coords.z < pts[v].z)
                return true;
        }
    }
    return false;
}

template <typename T4, typename R3>
__device__ void  Delaunay1RingPost<T4, R3>::remove_triangle(int i) {
    nt--;
    swap(tr(i), tr(nt));
    nr++;
}

template <typename T4, typename R3>
__device__ bool  Delaunay1RingPost<T4, R3>::halfspace_intersection(int k) {
    int vid = index(k);
    index(k) = encode_index(vid);
 
    if (status == vertex_overflow) {
        nt = nt + nr;
        return false;
    }
    //the processing point is in the circumsphere  
    R3& position = pts[vid];
    bool exact_tag = true;
    for (int i = 0; i < nt;) {

        if (inspherefast(predConsts, position, pts[rindex(tr(i).x)], pts[rindex(tr(i).y)], pts[rindex(tr(i).z)], target_coords, &exact_tag)) {
            remove_triangle(i);
        }
        else {
            i++;
        }
    }

    if (nr == 0 || !exact_tag) {
        nt = nt + nr;
        nr = 0; 
        return false;
    }

    T first_boundary = compute_boundary();
    if (first_boundary == (T)-1) {
        return false;
    }
    //add new triangles
    T cir = first_boundary;
    do {
        add_triangle(k, cir, boundary_next[cir]);
        cir = boundary_next[cir];
    } while (cir != first_boundary);
    return true;
}

template <typename T4, typename R3>
__device__ void  Delaunay1RingPost<T4, R3>::halfspace_intersection_post(int vid, uchar * predict_tag) {

    //vertex may overflow
    int    cur_v = add_vertex(vid);
    if (status == vertex_overflow) {
        return;
    }

    nr = 0;
    for (int i = 0; i < nt;) {
        if (predict_tag[i] != 0) {
            remove_triangle(i);
            predict_tag[i] = predict_tag[nt];
        }
        else i++;
    }
    T first_boundary = compute_boundary();
    if (first_boundary == (T)-1) {
        return;
    }
    //add new triangles
    T cir = first_boundary;
    do {
        add_triangle(cur_v, cir, boundary_next[cir]);
        cir = boundary_next[cir];
    } while (cir != first_boundary);
}

template <typename T4, typename R3>
__device__ void Delaunay1RingPost<T4, R3>::dt_init(int n) {
    nv = n;
    nt = 4;
    for (int i = 4; i < nv; ) {        
        halfspace_intersection(i);
        if (status != success) {
            index(i) = index(nv - 1);
            nv--;
        }
        else
            i++;
            
    }  
    
}

template <typename T4, typename R3>
__device__ int Delaunay1RingPost<T4, R3>::ballsearch_post(int3 lowbound, int3 upbound, int3 lowbound0, int3 upbound0, R * detpre, R * boundpre, int* npredict) {// search a point in the circumsphere using one block
    int bounddet[3] = { upbound.x - lowbound.x + 1, upbound.y - lowbound.y + 1, upbound.z - lowbound.z + 1 };
    int dim = bounddet[0] * bounddet[1];

    for (int det = threadIdx.x; det < dim * bounddet[2]; det += BLOCK_SIZE_POST) {

        int k = det / dim;
        int j = (det % dim) / bounddet[0];
        int i = (det % dim) % bounddet[0];
        i = i + lowbound.x; j = j + lowbound.y; k = k + lowbound.z;
        if (lowbound0.x <= i && i <= upbound0.x && lowbound0.y <= j && j <= upbound0.y && lowbound0.z <= k && k <= upbound0.z)
            continue;//searched in the previous
        int id = i + j * M + k * M * M;
#ifdef SORTING_POINTS
        for (int c = cellStartIds[id]; c < cellStartIds[id + 1]; c++) {
#else
        for (int a = cellStartIds[id]; a < cellStartIds[id + 1]; a++) {
            int c = permutation[a];
#endif 
            if (c == act_triangle.x || c == act_triangle.y || c == act_triangle.z || c == target_id) {
                continue;
            }
            R3 pt = pts[c];
            npredict[0]++;
            R predData[PredicateTotalSize];
            R det = insphere(predConsts, predData, pt, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords, detpre, boundpre);
            if (det < 0)
                return c;
            if (det == 0) {
                int indices[5] = { c,act_triangle.x ,act_triangle.y ,act_triangle.z ,target_id };
                if (InSphereSoSOnly(predConsts, predData, indices, pt, pts[act_triangle.x], pts[act_triangle.y], pts[act_triangle.z], target_coords) < 0)
                    return c;
            }

        }
    }
    return -1;
}

template <typename T4, typename R3>
__device__ void Delaunay1RingPost<T4, R3>::output() {
    for (int i = 0; i < nt; i++) {
        if (!(tag(i) & BITOUTPUT)) {
            noutput = i;
            break;
        }
    }
    for (int i = noutput + 1; i < nt; i++) {
        if ((tag(i) & BITOUTPUT)) {
            tr(noutput) = tr(i);
            noutput++;
        }
    }
}


__global__ void PostProcessingMain_kernel(float3 * pts, int nv, int startseed, float* predConsts, uchar * stat, DTStructPost* output, int* cellStartIds, int M, int nunsuccess, int* unsuccessIds, int* permutation, int* ncycles, clock_t * time) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    /*seed = 145;*/
    if (seed  >= nunsuccess) return;
    clock_t start = clock();
    uchar boundary_next[MAX_P];
    Delaunay1RingPost<uchar4, float3> dr;
    dr.init(unsuccessIds[seed] + startseed, pts, nv, cellStartIds, M, output[seed].tr_data, output[seed].index_data, MAX_POST_P, predConsts);
    dr.boundary_next = &(boundary_next[0]);
#ifndef SORTING_POINTS
    dr.permutation = permutation;
#endif

    dr.dt_init(output[seed].nv);

    output[seed].nt = dr.nt;
    output[seed].nv = dr.nv;

}

template <typename T4>
__device__ void  PostProcessingPost<T4>::init() {
    finished = false;
    num = 0;
    idx_block = -1;
    activetr_id = -1;
}

template <typename T4>
__device__ void  PostProcessingPost<T4>::update_activetr() {
    int k = 0;
    finished = true;
    for (k = 0; k < dr.nt; k++) {
        if (!(dr.tag(k) & BITPROCESSED) || (dr.tag(k) & BITINEXACT)) {
            BitSet(dr.tag(k), PROCESSED);
            BitClear(dr.tag(k), INEXACT);
            finished = false;
            break;
        }
    }
    if (finished)
        return;

    activetr_id = k;
    dr.act_triangle = { dr.rindex(dr.tr(k).x), dr.rindex(dr.tr(k).y), dr.rindex(dr.tr(k).z) };

}

template <typename T4>
__device__ void  PostProcessingPost<T4>::search_prepare() {
    tag = false;
    dr.cell_range(&lowbound, &upbound, &center);
    insphereprepare(dr.pts[dr.act_triangle.x], dr.pts[dr.act_triangle.y], dr.pts[dr.act_triangle.z], dr.target_coords, detpre, boundpre);
}

template <typename T4>
__device__ int  PostProcessingPost<T4>::ballsearch(int* npredict) {
    int3 start = getcellidx(center, dr.M);
    int d = 2;
    int idx = -1;
    int3 lowbound0 = { start.x + 1, start.y + 1, start.z + 1 };
    int3 upbound0 = { start.x - 1, start.y - 1, start.z - 1 };
    while (!tag) {
        int3 lowbound2 = { max(lowbound.x, start.x - d),max(lowbound.y, start.y - d), max(lowbound.z, start.z - d) };
        int3 upbound2 = { min(upbound.x, start.x + d),min(upbound.y, start.y + d), min(upbound.z, start.z + d) };
        idx = dr.ballsearch_post(lowbound2, upbound2, lowbound0, upbound0, detpre, boundpre, npredict);
        if (idx != -1)
            tag = true;
        if (lowbound2.x == lowbound.x && lowbound2.y == lowbound.y && lowbound2.z == lowbound.z && upbound2.x == upbound.x && upbound2.y == upbound.y && upbound2.z == upbound.z)
            break;
        d = d + 2;
        lowbound0 = lowbound2;
        upbound0 = upbound2;
    }

    return idx;
};

template <typename T4>
__device__ void  PostProcessingPost<T4>::idx_reduction(int idx) {
    //we have already find a point in every thread, we will search for the point nearest to the target point using reduction algorithm
    REAL dist = 1000.;
    if (idx >= 0)
        dist = norm<REAL, REAL3>(center - dr.pts[idx]);
    for (int offset = 16; offset > 0; offset /= 2) {
        REAL dist2 = __shfl_down_sync(0xffffffff, dist, offset);
        int idx2 = __shfl_down_sync(0xffffffff, idx, offset);
        if (dist2 < dist) {
            idx = idx2;
            dist = dist2;
        }
    }

    if (threadIdx.x % 32 == 0) {
        idx_warp[threadIdx.x / 32] = idx;
        dist_warp[threadIdx.x / 32] = dist;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        idx_block = -1;
        REAL dist0 = 1000.;
        for (int t = 0; t < BLOCK_SIZE_POST / 32; t++) {
            if (dist_warp[t] < dist0) {
                idx_block = idx_warp[t];
                dist0 = dist_warp[t];
            }
        }
    }
    __syncthreads();
}

template <typename T4>
__device__ void  PostProcessingPost<T4>::do_predict(uchar * predict_tag) {
    for (int e = threadIdx.x; e < dr.nt; e = e + BLOCK_SIZE_POST) {
        if (e == activetr_id || idx_block == dr.rindex(dr.tr(e).x) || idx_block == dr.rindex(dr.tr(e).y) || idx_block == dr.rindex(dr.tr(e).z)) {
            predict_tag[e] = true;
            continue;
        }
        REAL predData[PredicateTotalSize];
        REAL det = insphere(dr.predConsts, predData, dr.pts[idx_block], dr.pts[dr.rindex(dr.tr(e).x)], dr.pts[dr.rindex(dr.tr(e).y)], dr.pts[dr.rindex(dr.tr(e).z)], dr.target_coords);
        if (det == 0) {
            int indices[5] = { idx_block,dr.rindex(dr.tr(e).x) ,dr.rindex(dr.tr(e).y) ,dr.rindex(dr.tr(e).z) ,dr.target_id };
            det = InSphereSoSOnly(dr.predConsts, predData, indices, dr.pts[idx_block], dr.pts[dr.rindex(dr.tr(e).x)], dr.pts[dr.rindex(dr.tr(e).y)], dr.pts[dr.rindex(dr.tr(e).z)], dr.target_coords);
        }
        predict_tag[e] = (det < 0);
    }
}

__global__ void PostProcessingPost_kernel(REAL3 * pts, int nv, int startseed, REAL * predConsts, uchar * stat, DTStructPost* output, int* cellStartIds, int M, int nunsuccess, int* unsuccessIds, uchar* exact, int* permutation, int* ncycles, clock_t * time) {
    clock_t start = clock();
    int seed = blockIdx.x;
    if (seed >= nunsuccess) return;
    if (stat[unsuccessIds[seed]] == success)
        return;
    __shared__ PostProcessingPost<uchar4> pp;

    __shared__ uchar predict_tag[MAX_POST_T];

    for (int i = threadIdx.x; i < MAX_T; i = i + BLOCK_SIZE_POST) {
        predict_tag[threadIdx.x] = 0;
    }

    int npredict = 0;
    if (threadIdx.x == 0) {
        pp.dr.init(unsuccessIds[seed] + startseed, pts, nv, cellStartIds, M, &(output[seed].tr_data[0]), &(output[seed].index_data[0]), MAX_POST_P, predConsts);
        pp.dr.nt = output[seed].nt;
        pp.dr.nv = output[seed].nv;
#ifndef SORTING_POINTS
        pp.dr.permutation = permutation;
#endif
        uchar boundary_next[MAX_P];
        pp.dr.boundary_next = &(boundary_next[0]);
        pp.init();
        if (exact)
            pp.dr.exact_tag = exact[seed];
        pp.update_activetr();
        if (!pp.finished) {
            pp.search_prepare();
        }

    }

    __syncthreads();
    while (!pp.finished && pp.num < MAX_POST_CYCLES2) {
        int idx = pp.ballsearch(&npredict);
        pp.idx_reduction(idx);
        if (pp.idx_block == -1) {//there is no point in the sphere
            if (pp.activetr_id == pp.dr.nt - 1) {
                pp.finished = true;
            }
        }
        else {// every thread process a triangle to find whether the point is in the circumsphere
            pp.do_predict(predict_tag);
        }
        __syncthreads();
        if (threadIdx.x == 0) {// update only use the first thread
            if (pp.idx_block != -1) {
                pp.dr.halfspace_intersection_post(pp.idx_block, predict_tag);
                pp.num++;
            }
            pp.update_activetr();
            if (!pp.finished) {
                pp.search_prepare();
            }
            if (pp.dr.status != success)
                pp.finished = true;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {//output
        /* printf("%d\n", pp.dr.nt);
         for (int i = 0; i < pp.dr.nt; i++)
             printf("%d\t%d\t%d\t%d\n", pp.dr.index(pp.dr.tr(i).x), pp.dr.index(pp.dr.tr(i).y), pp.dr.index(pp.dr.tr(i).z), pp.dr.tr(i).w);
         printf("\n");*/
         if (pp.num == MAX_POST_CYCLES2) {
             pp.dr.status = cycles_overflow;
         }
         /*if (pp.dr.status == success) {*/
             pp.dr.output();
             output[seed].nt = pp.dr.noutput;
             for (int i = 0; i < pp.dr.nv; i++)
                 pp.dr.index(i) = pp.dr.rindex(i);
         /*}
         else
             output[seed].nt = pp.dr.nt;*/
         stat[unsuccessIds[seed]] = pp.dr.status;
         output[seed].nv = pp.dr.nv;
    }
}

__global__ void  Voronoi_center(REAL3* pts, int nv, DTStruct* output, uchar* stat, REAL3* newpts, REAL* cellvol, uchar* exact_result, uchar* boundary) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = tid >> 1;        // tid / 2
    int side = tid & 1;         // tid % 2

    if (seed >= nv) return;
    if (stat[seed] != success) return;
    if (boundary[seed] == 0) return;

    DTStruct& dt = output[seed];

    int base = side * MAX_P;

    int last1 = dt.vertex_data[base];
    int last2 = decode_index(dt.vertex_data[base + 1]);

    REAL3 p[4];
    int   pid[4];

    int nv_local = dt.nv[side];

    for (int i = 2; i < nv_local; ++i) {

        int v = dt.vertex_data[base + i];

        if (v < 0) {  

            pid[0] = last1;
            pid[1] = last2;
            pid[2] = decode_index(v);
            pid[3] = seed;

            last2 = pid[2];

            if (pid[0] < 0 || pid[0] >= nv ||
                pid[1] < 0 || pid[1] >= nv ||
                pid[2] < 0 || pid[2] >= nv)
                continue;

            bool use_exact = (exact_result != nullptr) && (boundary[seed] == 1);

            bool in0 = false, in1 = false, in2 = false;
            if (use_exact) {
                in0 = (exact_result[pid[0]] == 0);
                in1 = (exact_result[pid[1]] == 0);
                in2 = (exact_result[pid[2]] == 0);
            }

            if (in0 || in1 || in2)
                continue;
          
            for (int j = 0; j < 4; ++j)
                p[j] = pts[pid[j]];

            REAL3 cc = ballcenter<REAL, REAL3>(p[0], p[1], p[2], p[3]);

            for (int k = 0; k < 4; ++k) {

                REAL3 bary{ 0., 0., 0. };

                REAL vol = weighted_bary(p[k], (p[k] + p[(k + 1) & 3]) / 2, (p[k] + p[(k + 2) & 3]) / 2,(p[k] + p[(k + 3) & 3]) / 2, cc,bary);

                if (k & 1) {
                    bary = -1 * bary ;
                    vol = -vol;
                }

                atomicAdd(&newpts[pid[k]].x, bary.x);
                atomicAdd(&newpts[pid[k]].y, bary.y);
                atomicAdd(&newpts[pid[k]].z, bary.z);
                atomicAdd(&cellvol[pid[k]], vol);
            }
        }
        else {
            last1 = v;
        }
    }
}

__global__ void  Voronoi_center_post(REAL3* pts, int nunsuccess, int* unsuccessIds, DTStructPost* output, REAL3* newpts, REAL* cellvol, uchar* exact_result, uchar* boundary) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nunsuccess) return;
    if (boundary[unsuccessIds[seed]] == 0) return;
    DTStructPost* dt = &(output[seed]);
    REAL3 p[4];
    int pid[4]; 
    for (int i = 0; i < dt->nt; i++) {
        uchar4 tr = dt->tr_data[i];
        pid[0] = dt->index_data[tr.x];
        pid[1] = dt->index_data[tr.y];
        pid[2] = dt->index_data[tr.z];
        pid[3] = unsuccessIds[seed];
        bool use_exact = (exact_result != nullptr) && (boundary[seed] == 1);

        bool in0 = false, in1 = false, in2 = false;
        if (use_exact) {
            in0 = (exact_result[pid[0]] == 0);
            in1 = (exact_result[pid[1]] == 0);
            in2 = (exact_result[pid[2]] == 0);
        }
        if (in0 || in1 || in2)
            continue;       
        for (int j = 0; j < 4; j++)
            p[j] = pts[pid[j]];
        REAL3 cc = ballcenter<REAL, REAL3>(p[0], p[1], p[2], p[3]);

        for (int k = 0; k < 4; k++) {
            REAL3 bary{ 0., 0., 0. };
            REAL vol = weighted_bary(p[k], (p[k] + p[(k + 1) % 4]) / 2, (p[k] + p[(k + 2) % 4]) / 2, (p[k] + p[(k + 3) % 4]) / 2, cc, bary);
            if (k % 2 == 1) {
                bary = -1 * bary;
                vol = -vol;
            }
            atomicAdd(&(newpts[pid[k]].x), bary.x);
            atomicAdd(&(newpts[pid[k]].y), bary.y);
            atomicAdd(&(newpts[pid[k]].z), bary.z);
            atomicAdd(&(cellvol[pid[k]]), vol);
        }        
    }
}


__global__ void update_pts(REAL3* pts, REAL3* sortedpts, int nv, REAL3* newpts, REAL* tetvol, uchar* boundary, REAL* distance) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= nv) return;
    REAL3 pos = sortedpts[seed];
    if (boundary[seed] != 2) {
        pts[seed] = pos;
        distance[seed] = 0;
        return;
    }
    REAL vol = tetvol[seed];
    REAL3 pos2 = newpts[seed];
    pos2 = { pos2.x / vol, pos2.y / vol, pos2.z / vol };
    if (pos2.x < 0. || pos2.y < 0. || pos2.z < 0. || pos2.x >= 1. || pos2.y >= 1. || pos2.z >= 1.) {
        pts[seed] = pos;
        distance[seed] = 0;
        return;
    }
    distance[seed] = norm<REAL, REAL3>(pos - pos2);
    pts[seed] = pos2;
}
