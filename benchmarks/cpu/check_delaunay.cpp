// Offline, quadratic-cost validation of an unweighted full-hull tet mesh.
// This executable is not used by warpower or by its GPU execution path.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/version.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
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
    const bool reference = argc == 4 && std::string(argv[3]) == "--cgal-connectivity";
    if (argc != 3 && !reference) {
        std::cerr << "usage: check-delaunay points.txt tetrahedra.txt [--cgal-connectivity]\n";
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
    std::set<Tet> reference_tets;
    if (reference) {
        // Strict equality to an independently built, validated CGAL triangulation.
        // This mode may reject valid cospherical alternatives; it never approves
        // a nonmatching candidate. Use exhaustive mode for those alternatives.
        using Vertex = CGAL::Triangulation_vertex_base_with_info_3<int, K>;
        using Data = CGAL::Triangulation_data_structure_3<Vertex>;
        using Delaunay = CGAL::Delaunay_triangulation_3<K, Data>;
        std::vector<std::pair<K::Point_3, int>> indexed;
        for (int i = 0; i < n; ++i) indexed.emplace_back(points[i], i);
        Delaunay expected;
        expected.insert(indexed.begin(), indexed.end());
        if (expected.dimension() != 3 || !expected.is_valid())
            return fail("invalid independent CGAL reference");
        for (auto c = expected.finite_cells_begin(); c != expected.finite_cells_end(); ++c) {
            Tet key;
            for (int j = 0; j < 4; ++j) key[j] = c->vertex(j)->info();
            std::sort(key.begin(), key.end());
            reference_tets.insert(key);
        }
    }
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
        for (int q = 0; q < (reference ? 0 : n); ++q) {
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
    if (reference && unique_tets != reference_tets)
        return fail("candidate differs from independent CGAL connectivity (cospherical alternatives require exhaustive mode)");
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
    std::cout << "{\"scope\":\""
              << (reference
                  ? "Offline validated independent CGAL connectivity equality, exact original-coordinate orientation, face pairing and hull support; nonmatching cospherical alternatives rejected. Not a speed comparison."
                  : "Offline CGAL exact original-coordinate orientation, global empty-sphere, face pairing and hull-support checks; cospherical alternatives allowed. Not meshing or a speed comparison.")
              << "\",\"cgal_connectivity_match\":" << (reference ? "true" : "false") << ","
              << "\"CGAL\":\"" << CGAL_VERSION_STR << "\",\"sites\":" << n
              << ",\"tetrahedra\":" << m << ",\"sphere_tests\":" << tests
              << ",\"cospherical_ties\":" << ties << ",\"hull_faces\":" << boundary_faces
              << ",\"hull_support_tests\":" << hull_tests << ",\"valid\":true}\n";
}
