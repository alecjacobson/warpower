#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Intersections_3/Triangle_3_Triangle_3.h>
#include <iostream>
#include <variant>
#include <vector>
using K=CGAL::Exact_predicates_exact_constructions_kernel;
using P=K::Point_3;
int main(){
  int n;std::cin>>n;
  for(int row=0;row<n;++row){
    int ids[6];P p[6];for(int i=0;i<6;++i)std::cin>>ids[i];
    for(int i=0;i<6;++i){double x,y,z;std::cin>>x>>y>>z;p[i]=P(x,y,z);}
    std::vector<P> shared;
    for(int i=0;i<3;++i)for(int j=3;j<6;++j)if(ids[i]==ids[j])shared.push_back(p[i]);
    auto result=CGAL::intersection(K::Triangle_3(p[0],p[1],p[2]),K::Triangle_3(p[3],p[4],p[5]));
    bool excess=bool(result);int type=-1;
    if(result){
      type=int(result->index());
      if(auto point=std::get_if<P>(&*result)){
        if(shared.size()==1 && *point==shared[0])excess=false;
      }else if(auto edge=std::get_if<K::Segment_3>(&*result)){
        if(shared.size()==2 && ((edge->source()==shared[0] && edge->target()==shared[1]) || (edge->source()==shared[1] && edge->target()==shared[0])))excess=false;
      }
    }
    std::cout<<row<<" "<<shared.size()<<" "<<int(bool(result))<<" "<<int(excess)<<" "<<type<<"\n";
  }
}
