// Independent benchmark/reference only; the Warpower runtime uses Warp kernels.
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/version.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vertex = CGAL::Triangulation_vertex_base_with_info_3<int, Kernel>;
using Data = CGAL::Triangulation_data_structure_3<Vertex>;
using Delaunay = CGAL::Delaunay_triangulation_3<Kernel, Data>;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: cgal-delaunay points.txt tets.txt\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    int count = 0;
    input >> count;
    if (!input || count < 0) return 2;
    std::vector<std::pair<Kernel::Point_3, int>> points;
    points.reserve(count);
    for (int i = 0; i < count; ++i) {
        double x, y, z;
        input >> x >> y >> z;
        if (!input) return 2;
        points.emplace_back(Kernel::Point_3(x, y, z), i);
    }
    const auto start = std::chrono::steady_clock::now();
    Delaunay mesh;
    mesh.insert(points.begin(), points.end());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (!mesh.is_valid()) return 3;
    std::ofstream output(argv[2]);
    output << mesh.number_of_finite_cells() << '\n';
    for (auto cell = mesh.finite_cells_begin(); cell != mesh.finite_cells_end(); ++cell) {
        for (int j = 0; j < 4; ++j) output << cell->vertex(j)->info() << ' ';
        output << '\n';
    }
    std::cerr << "{\"CGAL\":\"" << CGAL_VERSION_STR
              << "\",\"vertices\":" << mesh.number_of_vertices()
              << ",\"tetrahedra\":" << mesh.number_of_finite_cells()
              << ",\"build_seconds\":" << seconds << "}\n";
    return output ? 0 : 4;
}
