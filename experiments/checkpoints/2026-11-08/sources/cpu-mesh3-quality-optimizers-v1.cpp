// Native CPU Mesh_3/ODT benchmark. Never part of Warpower runtime.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/odt_optimize_mesh_3.h>
#include <CGAL/perturb_mesh_3.h>
#include <CGAL/exude_mesh_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/version.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
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
        if (argc != 9 && argc != 10 && argc != 11 && argc != 17) {
            std::cerr << "usage: cgal-mesh3 surface.off mesh.txt size facet_distance odt_iterations threads convergence freeze_bound [feature_angle [non_manifold|manifold [perturb_enabled perturb_seconds perturb_bound exude_enabled exude_seconds exude_bound]]]\n";
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
        const double feature_angle = argc >= 10 ? std::stod(argv[9]) : 45.0;
        if (!std::isfinite(feature_angle) || feature_angle <= 0 || feature_angle >= 180)
            throw std::runtime_error("Feature angle must be finite and strictly between 0 and 180 degrees");
        const std::string topology = argc >= 11 ? argv[10] : "non_manifold";
        if (topology != "non_manifold" && topology != "manifold")
            throw std::runtime_error("Topology must be non_manifold or manifold");
        const bool optimizer_options = argc == 17;
        const std::string perturb_flag = optimizer_options ? argv[11] : "0";
        const std::string exude_flag = optimizer_options ? argv[14] : "0";
        if ((perturb_flag != "0" && perturb_flag != "1") || (exude_flag != "0" && exude_flag != "1"))
            throw std::runtime_error("Optimizer flags must be 0 or 1");
        const bool perturb = perturb_flag == "1", exude = exude_flag == "1";
        const double perturb_seconds = optimizer_options ? std::stod(argv[12]) : 0.0;
        const double perturb_bound = optimizer_options ? std::stod(argv[13]) : 0.0;
        const double exude_seconds = optimizer_options ? std::stod(argv[15]) : 0.0;
        const double exude_bound = optimizer_options ? std::stod(argv[16]) : 0.0;
        if (!std::isfinite(perturb_seconds) || perturb_seconds < 0 ||
            !std::isfinite(exude_seconds) || exude_seconds < 0 ||
            !std::isfinite(perturb_bound) || perturb_bound < 0 || perturb_bound > 180 ||
            !std::isfinite(exude_bound) || exude_bound < 0 || exude_bound > 180)
            throw std::runtime_error("Invalid optimizer time limit or sliver bound");
        const auto topology_options = topology == "manifold"
            ? CGAL::parameters::manifold() : CGAL::parameters::non_manifold();
        const auto start = Clock::now();
        CGAL::get_default_random() = CGAL::Random(2026);
        CGAL::Random rng(2026);
        Domain domain(surface, &rng);
        domain.detect_features(feature_angle);
        Criteria criteria(CGAL::parameters::edge_size(h).facet_angle(25.0)
            .facet_size(h).facet_distance(distance).cell_radius_edge_ratio(2.0).cell_size(h));
        const auto domain_end = Clock::now();
        Complex mesh = CGAL::make_mesh_3<Complex>(domain, criteria,
            CGAL::parameters::no_perturb(), CGAL::parameters::no_exude(), topology_options);
        const auto mesh_end = Clock::now();
        int result=-1;
        if (iterations>0)
            result=int(CGAL::odt_optimize_mesh_3(mesh,domain,
                CGAL::parameters::max_iteration_number(iterations).convergence(convergence).freeze_bound(freeze_bound)));
        const auto odt_end = Clock::now();
        int perturb_result=-1, exude_result=-1;
        if (perturb)
            perturb_result=int(CGAL::perturb_mesh_3(mesh,domain,
                CGAL::parameters::time_limit(perturb_seconds).sliver_bound(perturb_bound)));
        const auto perturb_end = Clock::now();
        // Exudation modifies weights and must follow all vertex-position optimizers.
        if (exude)
            exude_result=int(CGAL::exude_mesh_3(mesh,
                CGAL::parameters::time_limit(exude_seconds).sliver_bound(exude_bound)));
        const auto optimization_end = Clock::now();
        if (!mesh.triangulation().is_valid()) throw std::runtime_error("Invalid triangulation");
        const auto checked = Clock::now();
        std::ofstream output(argv[2]);
        output << std::setprecision(17);
        std::map<Triangulation::Vertex_handle,int> ids;
        for (auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c)
            for (int j=0;j<4;++j) ids.emplace(c->vertex(j),-1);
        output << ids.size() << ' ' << mesh.number_of_cells_in_complex() << '\n';
        int id=0, nonzero_weights=0;
        double minimum_weight=0.0, maximum_weight=0.0;
        for (auto v=mesh.triangulation().finite_vertices_begin();v!=mesh.triangulation().finite_vertices_end();++v) {
            if (ids.find(v)==ids.end()) continue;
            const double weight=CGAL::to_double(v->point().weight());
            if (id==0 || weight<minimum_weight) minimum_weight=weight;
            if (id==0 || weight>maximum_weight) maximum_weight=weight;
            nonzero_weights+=int(weight!=0.0);
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
            << ",\"topology\":\"" << topology << "\""
            << ",\"odt_convergence\":" << convergence << ",\"odt_freeze_bound\":" << freeze_bound
            << ",\"requested_odt_iterations\":" << iterations << ",\"odt_return_code\":" << result
            << ",\"perturb_enabled\":" << int(perturb)
            << ",\"perturb_time_limit\":" << perturb_seconds << ",\"perturb_sliver_bound\":" << perturb_bound
            << ",\"perturb_return_code\":" << perturb_result
            << ",\"exude_enabled\":" << int(exude)
            << ",\"exude_time_limit\":" << exude_seconds << ",\"exude_sliver_bound\":" << exude_bound
            << ",\"exude_return_code\":" << exude_result
            << ",\"material_vertices_with_nonzero_weight\":" << nonzero_weights
            << ",\"minimum_material_weight\":" << minimum_weight << ",\"maximum_material_weight\":" << maximum_weight
            << ",\"vertices\":" << id
            << ",\"triangulation_vertices\":" << mesh.triangulation().number_of_vertices()
            << ",\"tetrahedra\":" << mesh.number_of_cells_in_complex()
            << ",\"input_seconds\":" << elapsed(input_start,start)
            << ",\"domain_seconds\":" << elapsed(start,domain_end)
            << ",\"mesh_seconds\":" << elapsed(domain_end,mesh_end)
            << ",\"odt_seconds\":" << elapsed(mesh_end,odt_end)
            << ",\"perturb_seconds\":" << elapsed(odt_end,perturb_end)
            << ",\"exude_seconds\":" << elapsed(perturb_end,optimization_end)
            << ",\"mesh_and_optimization_seconds\":" << elapsed(domain_end,optimization_end)
            << ",\"complete_seconds\":" << elapsed(start,optimization_end)
            << ",\"validation_seconds\":" << elapsed(optimization_end,checked)
            << ",\"output_seconds\":" << elapsed(checked,done) << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
