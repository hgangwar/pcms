#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <pcms/transfer/omega_h_form_integrator_utils.hpp>

#include <array>
#include <cmath>
#include <utility>

namespace
{

r3d::Polytope<2> MakePoly(int nverts,
                          std::initializer_list<std::pair<double, double>> pts)
{
  r3d::Polytope<2> poly;
  poly.nverts = nverts;
  int i = 0;
  for (auto [x, y] : pts) {
    poly.verts[i].pos[0] = x;
    poly.verts[i].pos[1] = y;
    poly.verts[i].pnbrs[0] = -1;
    poly.verts[i].pnbrs[1] = -1;
    ++i;
  }
  return poly;
}

} // namespace

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: no duplicates leaves triangle intact",
  "[form_integrator_utils]")
{
  // A clean CCW triangle — nothing to remove, area should equal 0.5.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
  // The function sets correct pnbrs, so r3d::measure is valid after the call.
  const double area = r3d::measure(poly);
  REQUIRE(area == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("RemoveDuplicateVerticesAndFixLinks: one duplicate vertex is removed",
          "[form_integrator_utils]")
{
  // 4-vertex polygon; last vertex duplicates the first within the default tol.
  // After dedup: 3 unique vertices, area of the underlying triangle preserved.
  auto poly = MakePoly(4, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {5e-13, 0.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
  const double area = r3d::measure(poly);
  REQUIRE(area == Catch::Approx(0.5).epsilon(1e-10));
}

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: polygon becomes degenerate after dedup",
  "[form_integrator_utils]")
{
  // 3 vertices where two positions are identical — only 2 unique remain.
  // The function must mark the polygon degenerate: all pnbrs set to -1.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 2);
  REQUIRE(poly.nverts == 2);
  for (int i = 0; i < 2; ++i) {
    CAPTURE(i);
    CHECK(poly.verts[i].pnbrs[0] == -1);
    CHECK(poly.verts[i].pnbrs[1] == -1);
  }
}

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: near-tolerance vertex is retained",
  "[form_integrator_utils]")
{
  // Vertex at (0, 2e-12) differs from (0, 0) by 2e-12 in y, which exceeds the
  // default tol=1e-12, so it must NOT be treated as a duplicate.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 2e-12}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
}

// ---------------------------------------------------------------------------
// ForEachPolytopeFaceTriangle: the 3D face-walk that ForEachIntersectionSubtet
// relies on to star-decompose a clipped intersection polyhedron.
// ---------------------------------------------------------------------------

namespace
{

double TetVolume(const r3d::Vector<3>& a, const r3d::Vector<3>& b,
                 const r3d::Vector<3>& c, const r3d::Vector<3>& d)
{
  const double bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const double cx = c[0] - a[0], cy = c[1] - a[1], cz = c[2] - a[2];
  const double dx = d[0] - a[0], dy = d[1] - a[1], dz = d[2] - a[2];
  const double triple = bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) +
                        bz * (cx * dy - cy * dx);
  return std::abs(triple) / 6.0;
}

// Star-decompose `poly` from its centroid exactly as ForEachIntersectionSubtet
// does, summing the sub-tet volumes and counting the emitted face triangles.
std::pair<double, int> DecomposeFromCentroid(const r3d::Polytope<3>& poly)
{
  r3d::Vector<3> apex;
  apex[0] = apex[1] = apex[2] = 0.0;
  for (int v = 0; v < poly.nverts; ++v) {
    apex[0] += poly.verts[v].pos[0];
    apex[1] += poly.verts[v].pos[1];
    apex[2] += poly.verts[v].pos[2];
  }
  apex[0] /= poly.nverts;
  apex[1] /= poly.nverts;
  apex[2] /= poly.nverts;

  double total_vol = 0.0;
  int ntri = 0;
  pcms::detail::ForEachPolytopeFaceTriangle(poly, [&](const r3d::Vector<3>& a,
                                                      const r3d::Vector<3>& b,
                                                      const r3d::Vector<3>& c) {
    total_vol += TetVolume(apex, a, b, c);
    ++ntri;
  });
  return {total_vol, ntri};
}

r3d::Few<r3d::Vector<3>, 4> MakeTetVerts(
  std::initializer_list<std::array<double, 3>> pts)
{
  r3d::Few<r3d::Vector<3>, 4> verts;
  int i = 0;
  for (const auto& p : pts) {
    verts[i][0] = p[0];
    verts[i][1] = p[1];
    verts[i][2] = p[2];
    ++i;
  }
  return verts;
}

} // namespace

TEST_CASE("ForEachPolytopeFaceTriangle: a tetrahedron yields 4 face triangles "
          "that tile its volume",
          "[form_integrator_utils]")
{
  // A tetrahedron has 4 triangular faces, so the walk must emit exactly 4
  // triangles (= 2*(nverts-2) for nverts==4), and the centroid star-tiling must
  // reproduce the tet's volume (1/6 for the reference tet).
  r3d::Polytope<3> poly;
  r3d::init(poly, MakeTetVerts({{{0.0, 0.0, 0.0}},
                                {{1.0, 0.0, 0.0}},
                                {{0.0, 1.0, 0.0}},
                                {{0.0, 0.0, 1.0}}}));

  const auto [vol, ntri] = DecomposeFromCentroid(poly);
  REQUIRE(ntri == 4);
  REQUIRE(ntri == 2 * (poly.nverts - 2));
  REQUIRE(vol == Catch::Approx(1.0 / 6.0).epsilon(1e-12));
  REQUIRE(vol == Catch::Approx(std::abs(r3d::measure(poly))).epsilon(1e-12));
}

TEST_CASE(
  "ForEachPolytopeFaceTriangle: a clipped polyhedron with quad faces is "
  "tiled exactly",
  "[form_integrator_utils]")
{
  // Clip the reference tet with the plane x <= 0.5. This truncates the corner
  // at (1,0,0), producing a polyhedron with a quadrilateral face, so at least
  // one face must fan into more than one triangle. The removed piece is a tet
  // similar to the original at scale 0.5 (volume (1/6)*0.5^3 = 1/48), leaving
  // 1/6 - 1/48 = 7/48.
  r3d::Polytope<3> poly;
  r3d::init(poly, MakeTetVerts({{{0.0, 0.0, 0.0}},
                                {{1.0, 0.0, 0.0}},
                                {{0.0, 1.0, 0.0}},
                                {{0.0, 0.0, 1.0}}}));
  r3d::Few<r3d::Plane<3>, 1> planes;
  planes[0].n[0] = -1.0; // keep -x + 0.5 >= 0, i.e. x <= 0.5
  planes[0].n[1] = 0.0;
  planes[0].n[2] = 0.0;
  planes[0].d = 0.5;
  r3d::clip(poly, planes);

  REQUIRE(poly.nverts > 4); // truncation added vertices / a quad face
  const double measure = std::abs(r3d::measure(poly));
  REQUIRE(measure == Catch::Approx(7.0 / 48.0).epsilon(1e-12));

  const auto [vol, ntri] = DecomposeFromCentroid(poly);
  // Every face walked exactly once <=> the genus-0 fan invariant holds.
  REQUIRE(ntri == 2 * (poly.nverts - 2));
  REQUIRE(ntri > 4); // a quad face fans into 2+ triangles
  REQUIRE(vol == Catch::Approx(measure).epsilon(1e-12));
}
