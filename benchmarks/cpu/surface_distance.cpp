// Offline surface-distance validation only; never used by the Warp runtime.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/version.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Triangles = std::vector<K::Triangle_3>;
using Primitive = CGAL::AABB_triangle_primitive_3<K, Triangles::const_iterator>;
using Traits = CGAL::AABB_traits_3<K, Primitive>;

bool read_point(std::istream& in, K::Point_3& p) {
    double x, y, z;
    if (!(in >> x >> y >> z) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    p = K::Point_3(x, y, z);
    return true;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "CGAL " << CGAL_VERSION_STR << "\n";
        return 0;
    }
    if (argc != 3) {
        std::cerr << "usage: surface-distance target.off queries.txt\n";
        return 2;
    }
    std::ifstream surface(argv[1]), queries(argv[2]);
    std::string magic;
    int n = 0, m = 0, edges = 0;
    if (!(surface >> magic >> n >> m >> edges) || magic != "OFF" || n < 3 || m < 1) return 2;
    std::vector<K::Point_3> points(n);
    for (auto& p : points) if (!read_point(surface, p)) return 2;
    Triangles triangles;
    triangles.reserve(m);
    for (int i = 0; i < m; ++i) {
        int count, a, b, c;
        if (!(surface >> count >> a >> b >> c) || count != 3 ||
            a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) return 2;
        triangles.emplace_back(points[a], points[b], points[c]);
        if (triangles.back().is_degenerate()) return 3;
    }
    CGAL::AABB_tree<Traits> tree(triangles.cbegin(), triangles.cend());
    tree.accelerate_distance_queries();
    int count = 0;
    if (!(queries >> count) || count < 0) return 2;
    std::cout << std::setprecision(17);
    for (int i = 0; i < count; ++i) {
        K::Point_3 q;
        if (!read_point(queries, q)) return 2;
        double squared = CGAL::to_double(tree.squared_distance(q));
        if (!std::isfinite(squared) || squared < 0) return 3;
        std::cout << squared << '\n';
    }
    return 0;
}
