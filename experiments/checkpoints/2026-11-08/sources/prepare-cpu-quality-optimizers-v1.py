"""Prepare optional CGAL quality optimizer controls privately; production sources unchanged."""
import ast,hashlib,json
from pathlib import Path
w=Path('.work/vtm');cpp=Path('benchmarks/cpu/mesh3.cpp');runner=Path('benchmarks/cpu_mesher.py')
text=cpp.read_text().replace('#include <CGAL/odt_optimize_mesh_3.h>','#include <CGAL/odt_optimize_mesh_3.h>\n#include <CGAL/perturb_mesh_3.h>\n#include <CGAL/exude_mesh_3.h>')
text=text.replace('argc != 9 && argc != 10 && argc != 11','argc != 9 && argc != 10 && argc != 11 && argc != 17')
text=text.replace('[feature_angle [non_manifold|manifold]]','[feature_angle [non_manifold|manifold [perturb_enabled perturb_seconds perturb_bound exude_enabled exude_seconds exude_bound]]]')
text=text.replace('argc == 11 ? argv[10]','argc >= 11 ? argv[10]')
needle='        const auto topology_options = topology == "manifold"';assert text.count(needle)==1
text=text.replace(needle,'''        const bool optimizer_options = argc == 17;
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
'''+needle)
needle='        const auto odt_end = Clock::now();';assert text.count(needle)==1
text=text.replace(needle,needle+'''
        int perturb_result=-1, exude_result=-1;
        if (perturb)
            perturb_result=int(CGAL::perturb_mesh_3(mesh,domain,
                CGAL::parameters::time_limit(perturb_seconds).sliver_bound(perturb_bound)));
        const auto perturb_end = Clock::now();
        // Exudation modifies weights and must follow all vertex-position optimizers.
        if (exude)
            exude_result=int(CGAL::exude_mesh_3(mesh,
                CGAL::parameters::time_limit(exude_seconds).sliver_bound(exude_bound)));
        const auto optimization_end = Clock::now();''')
text=text.replace('        int id=0;','        int id=0, nonzero_weights=0;\n        double minimum_weight=0.0, maximum_weight=0.0;')
needle='            ids[v]=id++;';assert text.count(needle)==1
text=text.replace(needle,'''            const double weight=CGAL::to_double(v->point().weight());
            if (id==0 || weight<minimum_weight) minimum_weight=weight;
            if (id==0 || weight>maximum_weight) maximum_weight=weight;
            nonzero_weights+=int(weight!=0.0);
'''+needle)
needle='            << ",\\"vertices\\":" << id';assert text.count(needle)==1
text=text.replace(needle,'''            << ",\\"perturb_enabled\\":" << int(perturb)
            << ",\\"perturb_time_limit\\":" << perturb_seconds << ",\\"perturb_sliver_bound\\":" << perturb_bound
            << ",\\"perturb_return_code\\":" << perturb_result
            << ",\\"exude_enabled\\":" << int(exude)
            << ",\\"exude_time_limit\\":" << exude_seconds << ",\\"exude_sliver_bound\\":" << exude_bound
            << ",\\"exude_return_code\\":" << exude_result
            << ",\\"material_vertices_with_nonzero_weight\\":" << nonzero_weights
            << ",\\"minimum_material_weight\\":" << minimum_weight << ",\\"maximum_material_weight\\":" << maximum_weight
'''+needle)
text=text.replace('            << ",\\"complete_seconds\\":" << elapsed(start,odt_end)','''            << ",\\"perturb_seconds\\":" << elapsed(odt_end,perturb_end)
            << ",\\"exude_seconds\\":" << elapsed(perturb_end,optimization_end)
            << ",\\"mesh_and_optimization_seconds\\":" << elapsed(domain_end,optimization_end)
            << ",\\"complete_seconds\\":" << elapsed(start,optimization_end)''')
text=text.replace('elapsed(odt_end,checked)','elapsed(optimization_end,checked)')
p=w/'cpu-mesh3-quality-optimizers-v1.cpp';assert not p.exists();p.write_text(text)
s=runner.read_text().replace('optional ODT. Complete time','optional ODT, perturbation and exudation. Complete time')
needle='    parser.add_argument("--threads",';i=s.index(needle)
s=s[:i]+'''    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--exude", action="store_true")
    parser.add_argument("--perturb-time-limit", type=float, default=0.0)
    parser.add_argument("--perturb-sliver-bound", type=float, default=0.0)
    parser.add_argument("--exude-time-limit", type=float, default=0.0)
    parser.add_argument("--exude-sliver-bound", type=float, default=0.0)
'''+s[i:]
needle='    points, faces = read_surface(args.surface)';assert s.count(needle)==1
s=s.replace(needle,'''    for name in ["perturb", "exude"]:
        limit, bound = getattr(args, name+"_time_limit"), getattr(args, name+"_sliver_bound")
        if not np.isfinite(limit) or limit < 0 or not np.isfinite(bound) or not 0 <= bound <= 180:
            parser.error("Optimizer time limits must be finite/nonnegative and sliver bounds in [0,180]")
        if not getattr(args,name) and (limit != 0 or bound != 0):
            parser.error("Optimizer limits require the corresponding --perturb or --exude flag")
'''+needle)
s=s.replace('if args.feature_angle != 45.0 or args.topology != "non_manifold":','if args.feature_angle != 45.0 or args.topology != "non_manifold" or args.perturb or args.exude:')
s=s.replace('        if args.topology != "non_manifold":','        if args.topology != "non_manifold" or args.perturb or args.exude:')
needle='            command.append(args.topology)';assert s.count(needle)==1
s=s.replace(needle,needle+'''
        if args.perturb or args.exude:
            command.extend(map(str, [int(args.perturb), args.perturb_time_limit, args.perturb_sliver_bound,
                                     int(args.exude), args.exude_time_limit, args.exude_sliver_bound]))''')
needle='            if "mesh_sha256" not in record:';assert s.count(needle)==1
s=s.replace(needle,'''            for name in ["perturb", "exude"]:
                enabled = int(getattr(args,name))
                if native.get(name+"_enabled",0) != enabled:
                    raise ValueError("Native optimizer enablement does not match requested policy")
                if args.perturb or args.exude:
                    for option in ["time_limit","sliver_bound"]:
                        if native.get(name+"_"+option) != getattr(args,name+"_"+option):
                            raise ValueError("Native optimizer limit does not match requested policy")
'''+needle)
needle='        runs=records,';assert s.count(needle)==1
s=s.replace(needle,'''        median_unthrottled_mesh_and_optimization_seconds=float(np.median([
            x.get("mesh_and_optimization_seconds", x["mesh_seconds"] + x["odt_seconds"])
            for x in valid if not x["cpu_throttled"]])) if clean else None,
'''+needle)
p2=w/'cpu_mesher_quality_optimizers_v1.py';assert not p2.exists();ast.parse(s);p2.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(scope=__doc__,production={str(x):sha(x) for x in [cpp,runner]},private={str(x):sha(x) for x in [p,p2]},cgalsource='44ab808c25399fc6dbd89a6e114f3fd6b5ae8142',documentation=['https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Functions.html','https://doc.cgal.org/latest/Mesh_3/index.html'],limits='Not built or measured. Both scopes measured in one fresh-domain run; subtracting preparation is not evidence of separately measured repeated domain reuse.')
p=w/'cpu-quality-optimizers-source-v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')
