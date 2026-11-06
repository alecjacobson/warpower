#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <fstream>
#include <iostream>
#include <vector>
using K=CGAL::Exact_predicates_exact_constructions_kernel;
using M=CGAL::Surface_mesh<K::Point_3>;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);std::string magic;int n,m,e;in>>magic>>n>>m>>e;if(magic!="OFF")return 3;
 M mesh;std::vector<M::Vertex_index> ids;
 for(int i=0;i<n;++i){double x,y,z;in>>x>>y>>z;ids.push_back(mesh.add_vertex(K::Point_3(x,y,z)));}
 for(int i=0;i<m;++i){int k,a,b,c;in>>k>>a>>b>>c;if(k!=3)return 4;if(mesh.add_face(ids[a],ids[b],ids[c])==M::null_face())return 5;}
 std::vector<std::pair<M::Face_index,M::Face_index>> pairs;
 CGAL::Polygon_mesh_processing::self_intersections(mesh,std::back_inserter(pairs));
 std::cout<<n<<" "<<m<<" "<<pairs.size()<<"\n";
 for(size_t i=0;i<std::min(pairs.size(),size_t(256));++i)std::cout<<pairs[i].first.idx()<<" "<<pairs[i].second.idx()<<"\n";
}
