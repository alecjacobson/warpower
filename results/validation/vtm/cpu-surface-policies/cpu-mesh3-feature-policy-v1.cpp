// Native CPU Mesh_3/ODT benchmark. Never part of Warpower runtime.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/odt_optimize_mesh_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/version.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#ifdef WARPOWER_CGAL_PARALLEL
#include <tbb/global_control.h>
using Concurrency = CGAL::Parallel_tag;
#else
using Concurrency = CGAL::Sequential_tag;
#endif

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface = CGAL::Mesh_polyhedron_3<Kernel>::type;
using Domain = CGAL::Polyhedral_mesh_domain_with_features_3<Kernel, Surface>;
using Triangulation = CGAL::Mesh_triangulation_3<Domain, CGAL::Default, Concurrency>::type;
using Complex = CGAL::Mesh_complex_3_in_triangulation_3<Triangulation>;
using Criteria = CGAL::Mesh_criteria_3<Triangulation>;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

int main(int argc, char** argv) {
    try {
        if (argc != 9 && argc != 10) {
            std::cerr << "usage: cgal-mesh3 surface.off mesh.txt size facet_distance odt_iterations threads convergence freeze_bound [feature_angle]\n";
            return 2;
        }
        const int threads=std::stoi(argv[6]);
        if (threads<1) throw std::runtime_error("Thread count must be positive");
#ifdef WARPOWER_CGAL_PARALLEL
        tbb::global_control thread_limit(tbb::global_control::max_allowed_parallelism, threads);
        const char* concurrency="parallel";
#else
        if (threads!=1) throw std::runtime_error("Sequential build requires threads=1");
        const char* concurrency="sequential";
#endif
        const auto input_start = Clock::now();
        std::ifstream input(argv[1]);
        Surface surface;
        input >> surface;
        if (!input || surface.empty() || !surface.is_closed() || !surface.is_pure_triangle())
            throw std::runtime_error("Input must be a closed triangle surface");
        double h=std::stod(argv[3]), distance=std::stod(argv[4]);
        int iterations=std::stoi(argv[5]);
        double convergence=std::stod(argv[7]),freeze_bound=std::stod(argv[8]);
        if (!std::isfinite(convergence) || !std::isfinite(freeze_bound) || convergence<0 || convergence>1 || freeze_bound<0 || freeze_bound>1) throw std::runtime_error("Invalid stopping parameters");
        if (!std::isfinite(h) || h<=0 || !std::isfinite(distance) || distance<=0 || iterations<0)
            throw std::runtime_error("Invalid criteria");
        const double feature_angle = argc == 10 ? std::stod(argv[9]) : 45.0;
        if (!std::isfinite(feature_angle) || feature_angle <= 0 || feature_angle >= 180)
            throw std::runtime_error("Feature angle must be finite and strictly between 0 and 180 degrees");
        const auto start = Clock::now();
        CGAL::get_default_random() = CGAL::Random(2026);
        CGAL::Random rng(2026);
        Domain domain(surface, &rng);
        domain.detect_features(feature_angle);
        Criteria criteria(CGAL::parameters::edge_size(h).facet_angle(25.0)
            .facet_size(h).facet_distance(distance).cell_radius_edge_ratio(2.0).cell_size(h));
        const auto domain_end = Clock::now();
        Complex mesh = CGAL::make_mesh_3<Complex>(domain, criteria,
            CGAL::parameters::no_perturb(), CGAL::parameters::no_exude());
        const auto mesh_end = Clock::now();
        int result=-1;
        if (iterations>0)
            result=int(CGAL::odt_optimize_mesh_3(mesh,domain,
                CGAL::parameters::max_iteration_number(iterations).convergence(convergence).freeze_bound(freeze_bound)));
        const auto odt_end = Clock::now();
        if (!mesh.triangulation().is_valid()) throw std::runtime_error("Invalid triangulation");
        const auto checked = Clock::now();
        std::ofstream output(argv[2]);
        output << std::setprecision(17);
        std::map<Triangulation::Vertex_handle,int> ids;
        for (auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c)
            for (int j=0;j<4;++j) ids.emplace(c->vertex(j),-1);
        output << ids.size() << ' ' << mesh.number_of_cells_in_complex() << '\n';
        int id=0;
        for (auto v=mesh.triangulation().finite_vertices_begin();v!=mesh.triangulation().finite_vertices_end();++v) {
            if (ids.find(v)==ids.end()) continue;
            ids[v]=id++;
            const auto& p=v->point().point();
            output << CGAL::to_double(p.x()) << ' ' << CGAL::to_double(p.y()) << ' ' << CGAL::to_double(p.z()) << '\n';
        }
        for (auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c) {
            for (int j=0;j<4;++j) output << ids.at(c->vertex(j)) << ' ';
            output << '\n';
        }
        if (!output) throw std::runtime_error("Output failed");
        output.close();
        const auto done=Clock::now();
        std::cout << std::setprecision(17)
            << "{\"cgal\":\"" << CGAL_VERSION_STR << "\",\"concurrency\":\"" << concurrency
            << "\",\"threads\":" << threads << ",\"seed\":2026"
            << ",\"size\":" << h << ",\"facet_distance\":" << distance
            << ",\"facet_angle\":25,\"feature_angle\":" << feature_angle << ",\"cell_radius_edge_ratio\":2"
            << ",\"odt_convergence\":" << convergence << ",\"odt_freeze_bound\":" << freeze_bound
            << ",\"requested_odt_iterations\":" << iterations << ",\"odt_return_code\":" << result
            << ",\"vertices\":" << id
            << ",\"triangulation_vertices\":" << mesh.triangulation().number_of_vertices()
            << ",\"tetrahedra\":" << mesh.number_of_cells_in_complex()
            << ",\"input_seconds\":" << elapsed(input_start,start)
            << ",\"domain_seconds\":" << elapsed(start,domain_end)
            << ",\"mesh_seconds\":" << elapsed(domain_end,mesh_end)
            << ",\"odt_seconds\":" << elapsed(mesh_end,odt_end)
            << ",\"complete_seconds\":" << elapsed(start,odt_end)
            << ",\"validation_seconds\":" << elapsed(odt_end,checked)
            << ",\"output_seconds\":" << elapsed(checked,done) << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
