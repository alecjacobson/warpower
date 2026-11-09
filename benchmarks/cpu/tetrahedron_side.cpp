// Offline exact material-side reference for proved noncrossing tetrahedra.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <fstream>
#include <iostream>
#include <vector>

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using M = CGAL::Surface_mesh<K::Point_3>;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream in(argv[1]);
  std::string magic;
  int n, m, e;
  if (!(in >> magic >> n >> m >> e) || magic != "OFF" || n < 4 || m < 4) return 3;
  M mesh;
  std::vector<M::Vertex_index> ids;
  for (int i = 0; i < n; ++i) {
    double x, y, z;
    if (!(in >> x >> y >> z)) return 4;
    ids.push_back(mesh.add_vertex(K::Point_3(x, y, z)));
  }
  for (int i = 0; i < m; ++i) {
    int k, a, b, c;
    if (!(in >> k >> a >> b >> c) || k != 3 ||
        a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) return 5;
    if (mesh.add_face(ids[a], ids[b], ids[c]) == M::null_face()) return 6;
  }
  if (!CGAL::is_closed(mesh)) return 7;
  CGAL::Side_of_triangle_mesh<M, K> side(mesh);
  int cell;
  while (std::cin >> cell) {
    K::Point_3 p[4];
    for (auto& q : p) {
      double x, y, z;
      if (!(std::cin >> x >> y >> z)) return 8;
      q = K::Point_3(x, y, z);
    }
    K::Tetrahedron_3 tet(p[0], p[1], p[2], p[3]);
    if (tet.is_degenerate()) return 9;
    K::Vector_3 sum(0, 0, 0);
    for (const auto& q : p) sum = sum + (q - CGAL::ORIGIN);
    auto mean = CGAL::ORIGIN + sum / K::FT(4);
    if (!tet.has_on_bounded_side(mean)) return 10;
    std::cout << cell << ' ' << int(side(mean)) << '\n';
  }
  return std::cin.eof() ? 0 : 11;
}
