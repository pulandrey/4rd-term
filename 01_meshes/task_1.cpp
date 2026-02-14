#include <set>
#include <gmsh.h>

int main(int argc, char **argv)
{
  gmsh::initialize();

  gmsh::model::add("torus");

  double lc = 1e-2;
  
  int torus1 = gmsh::model::occ::addTorus(0, 0, 0, 1.0, 0.5);
  int torus2 = gmsh::model::occ::addTorus(0, 0, 0, 1.0, 0.4);

  std::vector<std::pair<int,int>> out;
  std::vector<std::vector<std::pair<int,int>>> outTool;

  gmsh::model::occ::cut(
      {{3, torus1}},
      {{3, torus2}},
      out,
      outTool
  );

  int f = gmsh::model::mesh::field::add("MathEval");

  gmsh::model::mesh::field::setString(f, "F", "0.02*sin((x+y)/5) + 0.03");

  gmsh::model::mesh::field::setAsBackgroundMesh(f);

  gmsh::model::occ::synchronize();

  gmsh::model::mesh::generate(3);

  gmsh::write("t33.msh");

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) gmsh::fltk::run();

  gmsh::finalize();

  return 0;
}
