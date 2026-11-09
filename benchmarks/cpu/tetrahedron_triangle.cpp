// Offline exact reference for strict-interior versus boundary-only contact.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Intersections_3/Tetrahedron_3_Triangle_3.h>
#include <iostream>
#include <variant>
#include <vector>

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using P = K::Point_3;

int main() {
  int cell, face;
  while (std::cin >> cell >> face) {
    P p[7];
    for (auto& q : p) {
      double x, y, z;
      if (!(std::cin >> x >> y >> z)) return 2;
      q = P(x, y, z);
    }
    K::Tetrahedron_3 tet(p[0], p[1], p[2], p[3]);
    K::Triangle_3 tri(p[4], p[5], p[6]);
    if (tet.is_degenerate() || tri.is_degenerate()) return 3;
    auto result = CGAL::intersection(tet, tri);
    std::vector<P> vertices;
    if (result) {
      if (auto q = std::get_if<P>(&*result)) vertices.push_back(*q);
      else if (auto s = std::get_if<K::Segment_3>(&*result))
        vertices = {s->source(), s->target()};
      else if (auto t = std::get_if<K::Triangle_3>(&*result))
        vertices = {t->vertex(0), t->vertex(1), t->vertex(2)};
      else if (auto v = std::get_if<std::vector<P>>(&*result)) vertices = *v;
      else return 4;
    }
    bool strict = false;
    if (!vertices.empty()) {
      K::Vector_3 sum(0, 0, 0);
      for (const auto& v : vertices) sum = sum + (v - CGAL::ORIGIN);
      auto mean = CGAL::ORIGIN + sum / K::FT(vertices.size());
      strict = tet.has_on_bounded_side(mean);
    }
    std::cout << cell << ' ' << face << ' ' << bool(result) << ' ' << strict << '\n';
  }
  return std::cin.eof() ? 0 : 5;
}
