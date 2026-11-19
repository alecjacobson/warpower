// Standalone CPU benchmark: exact-predicate Delaunay construction and full-hull CSR.
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/version.h>
#ifdef RF_PARALLEL
#include <tbb/global_control.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using V = CGAL::Triangulation_vertex_base_with_info_3<int, K>;
using C = CGAL::Triangulation_cell_base_3<K>;
#ifdef RF_PARALLEL
using TDS = CGAL::Triangulation_data_structure_3<V, C, CGAL::Parallel_tag>;
#else
using TDS = CGAL::Triangulation_data_structure_3<V, C>;
#endif
using DT = CGAL::Delaunay_triangulation_3<K, TDS>;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("points.f32 csr.bin warmups repeats threads");
        const int warmups = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        const int threads = std::stoi(argv[5]);
        if (warmups < 0 || repeats < 1 || threads < 1) throw std::runtime_error("bad counts");
#ifdef RF_PARALLEL
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
#else
        if (threads != 1) throw std::runtime_error("sequential executable needs one thread");
#endif
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        const auto bytes = input.tellg();
        if (!input || bytes <= 0 || bytes % 12) throw std::runtime_error("bad float32 input");
        const int n = int(bytes / 12);
        input.seekg(0);
        std::vector<float> xyz(3 * size_t(n));
        input.read(reinterpret_cast<char*>(xyz.data()), bytes);
        if (!input) throw std::runtime_error("short input");
        std::vector<std::pair<K::Point_3, int>> points;
        points.reserve(n);
        double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
                double x = xyz[3 * size_t(i) + k];
                if (!std::isfinite(x)) throw std::runtime_error("nonfinite input");
                lo[k] = std::min(lo[k], x); hi[k] = std::max(hi[k], x);
            }
            points.emplace_back(K::Point_3(xyz[3 * size_t(i)], xyz[3 * size_t(i) + 1],
                                           xyz[3 * size_t(i) + 2]), i);
        }
        xyz.clear(); xyz.shrink_to_fit();
        std::cout << std::setprecision(17);
        for (int iteration = -warmups; iteration < repeats; ++iteration) {
            auto start = Clock::now();
#ifdef RF_PARALLEL
            auto locks = std::make_unique<DT::Lock_data_structure>(
                CGAL::Bbox_3(lo[0]-1, lo[1]-1, lo[2]-1, hi[0]+1, hi[1]+1, hi[2]+1), 50);
            DT mesh(locks.get());
#else
            DT mesh;
#endif
            mesh.insert(points.begin(), points.end());
            auto built = Clock::now();
            if (mesh.dimension() != 3 || mesh.number_of_vertices() != size_t(n))
                throw std::runtime_error("this benchmark requires distinct full-dimensional sites");
            std::vector<int32_t> offsets(size_t(n) + 1, 0);
            for (auto e = mesh.finite_edges_begin(); e != mesh.finite_edges_end(); ++e) {
                ++offsets[e->first->vertex(e->second)->info() + 1];
                ++offsets[e->first->vertex(e->third)->info() + 1];
            }
            int64_t total = 0;
            for (int i = 0; i < n; ++i) {
                total += offsets[i + 1];
                if (total > std::numeric_limits<int32_t>::max()) throw std::runtime_error("CSR overflow");
                offsets[i + 1] = int32_t(total);
            }
            std::vector<int32_t> adjacency(total), cursor(offsets.begin(), offsets.end()-1);
            for (auto e = mesh.finite_edges_begin(); e != mesh.finite_edges_end(); ++e) {
                int a = e->first->vertex(e->second)->info(), b = e->first->vertex(e->third)->info();
                adjacency[cursor[a]++] = b; adjacency[cursor[b]++] = a;
            }
            auto exported = Clock::now();
            std::cout << "{\"iteration\":" << iteration << ",\"build_seconds\":"
                      << std::chrono::duration<double>(built-start).count()
                      << ",\"csr_seconds\":" << std::chrono::duration<double>(exported-built).count()
                      << ",\"total_seconds\":" << std::chrono::duration<double>(exported-start).count()
                      << ",\"vertices\":" << mesh.number_of_vertices()
                      << ",\"tetrahedra\":" << mesh.number_of_finite_cells()
                      << ",\"directed_edges\":" << total << "}" << std::endl;
            if (iteration == repeats - 1) {
                auto validation_start = Clock::now();
                if (!mesh.is_valid()) throw std::runtime_error("CGAL is_valid failed");
                const auto validation_seconds = std::chrono::duration<double>(Clock::now()-validation_start).count();
                std::ofstream output(argv[2], std::ios::binary);
                uint64_t header[2] = {uint64_t(n), uint64_t(total)};
                output.write(reinterpret_cast<char*>(header), sizeof(header));
                output.write(reinterpret_cast<char*>(offsets.data()), offsets.size()*sizeof(int32_t));
                output.write(reinterpret_cast<char*>(adjacency.data()), adjacency.size()*sizeof(int32_t));
                output.close();
                if (!output) throw std::runtime_error("CSR output failure");
                rusage usage{}; getrusage(RUSAGE_SELF, &usage);
                std::cout << "{\"final\":true,\"CGAL\":\"" << CGAL_VERSION_STR
                          << "\",\"threads\":" << threads << ",\"valid\":true,\"validation_seconds\":"
                          << validation_seconds << ",\"peak_rss_bytes\":" << uint64_t(usage.ru_maxrss)*1024
                          << "}" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl; return 1;
    }
}
