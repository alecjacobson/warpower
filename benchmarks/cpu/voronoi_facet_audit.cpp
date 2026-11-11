// Offline exact rational LP: does a candidate pair share a positive-area face
// inside the given box? Constraints are all neighbors of the owner in a valid
// CGAL full-hull Delaunay triangulation (or all other sites in small controls).
#include <CGAL/Gmpq.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Q = CGAL::Gmpq;
using Point = std::array<Q, 3>;
using Program = CGAL::Quadratic_program<Q>;

Point read_point(std::istream& input) {
    Point p;
    for (auto& x : p) {
        double value;
        if (!(input >> value)) throw std::runtime_error("incomplete input");
        x = Q(value); // Exact rational representation of the stored binary value.
    }
    return p;
}

void plane(Program& lp, int row, const Point& owner, const Point& neighbor, bool equality) {
    Q b(0);
    for (int k = 0; k < 3; ++k) {
        Q d = neighbor[k] - owner[k];
        lp.set_a(k, row, 2*d);
        b += d*d;
    }
    lp.set_b(row, b);
    if (equality) lp.set_r(row, CGAL::EQUAL);
    else lp.set_a(3, row, Q(1));
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: voronoi-facet-audit queries.txt");
        std::ifstream input(argv[1]);
        int count;
        if (!(input >> count) || count < 0) throw std::runtime_error("bad query count");
        for (int id = 0; id < count; ++id) {
            auto owner = read_point(input), candidate = read_point(input);
            auto lower = read_point(input), upper = read_point(input);
            int neighbors;
            if (!(input >> neighbors) || neighbors < 0) throw std::runtime_error("bad neighbor count");
            if (owner == candidate) throw std::runtime_error("duplicate candidate");
            Program lp(CGAL::SMALLER, false, Q(0), false, Q(0));
            lp.set_l(3, true, Q(0));
            lp.set_c(3, Q(-1));
            plane(lp, 0, owner, candidate, true);
            int row = 1;
            for (int i = 0; i < neighbors; ++i) {
                auto neighbor = read_point(input);
                if (neighbor != candidate) plane(lp, row++, owner, neighbor, false);
            }
            for (int k = 0; k < 3; ++k) {
                lp.set_a(k, row, Q(1)); lp.set_a(3, row, Q(1));
                lp.set_b(row++, upper[k]-owner[k]);
                lp.set_a(k, row, Q(-1)); lp.set_a(3, row, Q(1));
                lp.set_b(row++, owner[k]-lower[k]);
            }
            auto solution = CGAL::solve_linear_program(lp, Q());
            if (!solution.solves_linear_program(lp)) throw std::runtime_error("LP certificate invalid");
            if (solution.is_unbounded()) throw std::runtime_error("box-constrained LP is unbounded");
            bool face = solution.is_optimal() && solution.objective_value() < 0;
            std::cout << "{\"query\":" << id << ",\"positive_area_face\":"
                      << (face ? "true" : "false") << ",\"feasible\":"
                      << (solution.is_optimal() ? "true" : "false") << "}" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl; return 1;
    }
}
