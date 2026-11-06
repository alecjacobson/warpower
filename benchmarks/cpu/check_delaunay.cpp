// Offline, quadratic-cost validation of an unweighted full-hull tet mesh.
// This executable is not used by warpower or by its GPU execution path.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/version.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Tet = std::array<int, 4>;
using Triple = std::array<int, 3>;
struct Face {
    Triple oriented{};
    int count = 0;
    int balance = 0;
};

int fail(const char* message) {
    std::cerr << message << '\n';
    return 3;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: check-delaunay points.txt tetrahedra.txt\n";
        return 2;
    }
    K::Orientation_3 orientation;
    K::Side_of_oriented_sphere_3 sphere;
    const K::Point_3 a(0, 0, 0), b(1, 0, 0), c(0, 1, 0), d(0, 0, 1);
    if (orientation(a, b, c, d) != CGAL::POSITIVE ||
        sphere(a, b, c, d, K::Point_3(.1, .1, .1)) != CGAL::ON_POSITIVE_SIDE ||
        sphere(a, b, c, d, K::Point_3(2, 2, 2)) != CGAL::ON_NEGATIVE_SIDE) {
        return fail("predicate convention self-check failed");
    }
    std::ifstream input(argv[1]), cells(argv[2]);
    int n = 0, m = 0;
    input >> n;
    cells >> m;
    if (!input || !cells || n < 4 || m < 1) return 2;
    std::vector<K::Point_3> points;
    std::set<std::array<double, 3>> unique_points, referenced_points;
    for (int i = 0; i < n; ++i) {
        double x, y, z;
        input >> x >> y >> z;
        if (!input || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return 2;
        points.emplace_back(x, y, z);
        unique_points.insert({x, y, z});
    }
    std::set<Tet> unique_tets;
    std::map<Triple, Face> faces;
    const int local_faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
    unsigned long long tests = 0, ties = 0, hull_tests = 0;
    for (int row = 0; row < m; ++row) {
        Tet t;
        for (int& v : t) {
            cells >> v;
            if (!cells || v < 0 || v >= n) return 2;
            referenced_points.insert({points[v].x(), points[v].y(), points[v].z()});
        }
        Tet key = t;
        std::sort(key.begin(), key.end());
        if (std::adjacent_find(key.begin(), key.end()) != key.end())
            return fail("repeated vertex in cell");
        if (!unique_tets.insert(key).second) return fail("duplicate cell");
        if (orientation(points[t[0]], points[t[1]], points[t[2]], points[t[3]]) != CGAL::POSITIVE)
            return fail("nonpositive physical cell orientation");
        for (int q = 0; q < n; ++q) {
            if (std::find(t.begin(), t.end(), q) != t.end()) continue;
            const auto side = sphere(points[t[0]], points[t[1]], points[t[2]], points[t[3]], points[q]);
            ++tests;
            if (side == CGAL::ON_POSITIVE_SIDE) return fail("site strictly inside a cell circumsphere");
            ties += side == CGAL::ON_ORIENTED_BOUNDARY;
        }
        for (const auto& local : local_faces) {
            Triple f = {t[local[0]], t[local[1]], t[local[2]]};
            int sign = 1;
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j)
                    if (f[i] > f[j]) sign = -sign;
            Triple sorted = f;
            std::sort(sorted.begin(), sorted.end());
            Face& face = faces[sorted];
            face.oriented = f;
            ++face.count;
            face.balance += sign;
            if (face.count > 2) return fail("face incidence exceeds two");
        }
    }
    if (referenced_points != unique_points) return fail("unreferenced distinct input site");
    int boundary_faces = 0;
    std::map<std::array<int, 2>, int> boundary_edges;
    for (const auto& pair : faces) {
        const Face& face = pair.second;
        if (face.count == 2) {
            if (face.balance != 0) return fail("interior faces have the same orientation");
            continue;
        }
        ++boundary_faces;
        const Triple& f = face.oriented;
        for (const auto& p : points) {
            ++hull_tests;
            if (orientation(points[f[0]], points[f[1]], points[f[2]], p) == CGAL::POSITIVE)
                return fail("exposed face does not support the input convex hull");
        }
        for (int k = 0; k < 3; ++k) {
            std::array<int, 2> e = {f[k], f[(k + 1) % 3]};
            std::sort(e.begin(), e.end());
            ++boundary_edges[e];
        }
    }
    for (const auto& pair : boundary_edges)
        if (pair.second != 2) return fail("full-hull boundary edge is not manifold");
    std::cout << "{\"scope\":\"Offline CGAL exact original-coordinate orientation, global empty-sphere, face pairing and hull-support checks; cospherical alternatives allowed. Not meshing or a speed comparison.\","
              << "\"CGAL\":\"" << CGAL_VERSION_STR << "\",\"sites\":" << n
              << ",\"tetrahedra\":" << m << ",\"sphere_tests\":" << tests
              << ",\"cospherical_ties\":" << ties << ",\"hull_faces\":" << boundary_faces
              << ",\"hull_support_tests\":" << hull_tests << ",\"valid\":true}\n";
}
