#include "coupling_support.h"

#include <pcms/coupler/coupler.hpp>
#include <pcms/coupler/overlap_mask.h>
#include <pcms/field/function_space/mfem.h>
#include <pcms/field/layout/mfem.h>
#include <pcms/transfer/interpolator.h>

#include <Omega_h_file.hpp>
#include <mfem.hpp>
#include <redev.h>



using dtype = pcms::Real;
namespace
{

struct CoordKey
{
  long long ix;
  long long iy;

  bool operator<(const CoordKey& other) const
  {
    if (ix != other.ix)
      return ix < other.ix;
    return iy < other.iy;
  }
};

static CoordKey MakeCoordKey(double x, double y, double tol)
{
  return {static_cast<long long>(std::llround(x / tol)),
          static_cast<long long>(std::llround(y / tol))};
}

void CheckCoordinateOverlap(const Omega_h::Mesh& mesh_A,
                            const Omega_h::Mesh& mesh_B,
                            double xmin,
                            double xmax,
                            double ymin,
                            double ymax,
                            double tol)
{
  auto coords_A = Omega_h::HostRead<double>(mesh_A.coords());
  auto coords_B = Omega_h::HostRead<double>(mesh_B.coords());

  std::map<CoordKey, int> A_pts;
  std::map<CoordKey, int> B_pts;

  for (int v = 0; v < mesh_A.nverts(); ++v) {
    const double x = coords_A[2 * v + 0];
    const double y = coords_A[2 * v + 1];

    if (x >= xmin - tol && x <= xmax + tol && y >= ymin - tol &&
        y <= ymax + tol) {
      A_pts[MakeCoordKey(x, y, tol)] = v;
    }
  }

  for (int v = 0; v < mesh_B.nverts(); ++v) {
    const double x = coords_B[2 * v + 0];
    const double y = coords_B[2 * v + 1];

    if (x >= xmin - tol && x <= xmax + tol && y >= ymin - tol &&
        y <= ymax + tol) {
      B_pts[MakeCoordKey(x, y, tol)] = v;
    }
  }

  int common = 0;
  int missing_in_B = 0;
  int missing_in_A = 0;

  std::cout << "\n[COORD_OVERLAP_CHECK]\n";
  std::cout << "  A overlap vertices = " << A_pts.size() << "\n";
  std::cout << "  B overlap vertices = " << B_pts.size() << "\n";

  int shown = 0;

  for (const auto& [key, a_v] : A_pts) {
    auto it = B_pts.find(key);

    if (it != B_pts.end()) {
      ++common;
    } else {
      ++missing_in_B;

      if (shown < 20) {
        std::cout << "  missing in B: A local_v=" << a_v
                  << " x=" << coords_A[2 * a_v + 0]
                  << " y=" << coords_A[2 * a_v + 1] << "\n";
        ++shown;
      }
    }
  }

  shown = 0;

  for (const auto& [key, b_v] : B_pts) {
    auto it = A_pts.find(key);

    if (it == A_pts.end()) {
      ++missing_in_A;

      if (shown < 20) {
        std::cout << "  missing in A: B local_v=" << b_v
                  << " x=" << coords_B[2 * b_v + 0]
                  << " y=" << coords_B[2 * b_v + 1] << "\n";
        ++shown;
      }
    }
  }

  std::cout << "  common coordinates = " << common << "\n";
  std::cout << "  missing in B = " << missing_in_B << "\n";
  std::cout << "  missing in A = " << missing_in_A << "\n";
}

static redev::Partition MakeRCBPartition(int dim)
{
  redev::LOs ranks(1);
  std::iota(ranks.begin(), ranks.end(), 0);
  redev::Reals cuts = {0};
  return redev::Partition{redev::RCBPtn{dim, ranks, cuts}};
}

static void app_A(MPI_Comm comm,
                  const std::string mesh_file,
                  support::ThermalParams params,
                  string solver_type,
                  string prec_type)
{
  constexpr int order = 1;

  mfem::Mesh mesh(mesh_file, 1, 1);
  mfem::ParMesh pmesh(comm, mesh);

  const double T_left = 270.0;

  support::FEMSystem fem = support::Init_FEMSystem(&pmesh, order, params.kappa);

  const int left_bdr_attr = 1;
  const int right_bdr_attr = 3;
  const pcms::LO overlap_attr = 1;

  mfem::Array<int> ess_bdrA(pmesh.bdr_attributes.Max());
  ess_bdrA = 0;
  ess_bdrA[left_bdr_attr - 1] = 1;
  ess_bdrA[right_bdr_attr - 1] = 1;

  fem.fes->GetEssentialTrueDofs(ess_bdrA, fem.ess_tdofs);

  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, left_bdr_attr, T_left);
  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, 280.0);

  support::OutputPack outA("schwarz_A", pmesh, *fem.fes);
  outA.pvd.RegisterField("T", fem.x);
  outA.pvd.RegisterField("T_exact", &outA.exact);
  outA.pvd.RegisterField("error", &outA.err);

  const std::string coupler_name = "mfem_coupler";
  const std::string app_name = "client_A";
  const std::string field_name = "temp";

  pcms::Coupler cpl(coupler_name, comm, false, redev::Partition{});
  auto* app = cpl.AddApplication(app_name);

  auto fs = pcms::MFEMFieldFactory(
    *fem.pmesh, *fem.fes, *fem.x, pcms::CoordinateSystem::Cartesian);

  auto layout = fs.GetLayout();

  auto overlap_view =
    pcms::MFEMLayout::OverlapMaskFromAttribute(*fem.pmesh, overlap_attr);

  app->SetLayoutOverlapMask(
    field_name,
    std::make_unique<pcms::OverlapMask>(
      static_cast<size_t>(layout->GetNumOwnedDofHolder()), overlap_view));

  app->AddLayout(field_name, layout);

  auto handle = app->AddField(field_name, fs.CreateField<Real>());

  auto gdi = app->Add_GDI<pcms::GO>("global_comm", comm);

  GO flag = 1;
  int itr = 1;

  do {
    support::PrintTempStats(pmesh, *fem.x, "App A before solve", itr);

    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::SaveFields(outA, fem, itr);
    support::PrintTempStats(pmesh, *fem.x, "App A after solve", itr);

    double err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("A abs error=%e\n", err);

    app->SendPhase([&]() {
      handle.Send();
      gdi->Send(&residual, "residual", 1);
    });

    app->ReceivePhase([&]() { handle.Receive(); });

    app->ReceivePhase([&]() {
      auto done = gdi->Receive("done", 1)[0];

      while (!done) {
        sleep(1);
        done = gdi->Receive("done", 1)[0];
      }

      flag = gdi->Receive("flag", 1)[0];
    });

    ++itr;
  } while (flag);
}

static void app_B(MPI_Comm comm,
                  const std::string mesh_file,
                  support::ThermalParams params,
                  string solver_type,
                  string prec_type)
{
  constexpr int order = 1;

  mfem::Mesh mesh(mesh_file, 1, 1);
  mfem::ParMesh pmesh(comm, mesh);

  const double T_right = 300.0;

  support::FEMSystem fem = support::Init_FEMSystem(&pmesh, order, params.kappa);

  const int left_bdr_attr = 1;
  const int right_bdr_attr = 3;
  const pcms::LO overlap_attr = 1;

  mfem::Array<int> ess_bdrB(pmesh.bdr_attributes.Max());
  ess_bdrB = 0;
  ess_bdrB[left_bdr_attr - 1] = 1;
  ess_bdrB[right_bdr_attr - 1] = 1;

  fem.fes->GetEssentialTrueDofs(ess_bdrB, fem.ess_tdofs);

  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, T_right);

  support::OutputPack outB("schwarz_B", pmesh, *fem.fes);
  outB.pvd.RegisterField("T", fem.x);
  outB.pvd.RegisterField("T_exact", &outB.exact);
  outB.pvd.RegisterField("error", &outB.err);

  const std::string coupler_name = "mfem_coupler";
  const std::string app_name = "client_B";
  const std::string field_name = "temp";

  pcms::Coupler cpl(coupler_name, comm, false, redev::Partition{});
  auto* app = cpl.AddApplication(app_name);

  auto fs = pcms::MFEMFieldFactory(
    *fem.pmesh, *fem.fes, *fem.x, pcms::CoordinateSystem::Cartesian);

  auto layout = fs.GetLayout();

  auto overlap_view =
    pcms::MFEMLayout::OverlapMaskFromAttribute(*fem.pmesh, overlap_attr);

  app->SetLayoutOverlapMask(
    field_name,
    std::make_unique<pcms::OverlapMask>(
      static_cast<size_t>(layout->GetNumOwnedDofHolder()), overlap_view));

  app->AddLayout(field_name, layout);

  auto handle = app->AddField(field_name, fs.CreateField<Real>());

  auto gdi = app->Add_GDI<pcms::GO>("global_comm", comm);

  GO flag = 1;
  int itr = 1;

  do {
    support::PrintTempStats(pmesh, *fem.x, "App B before receive", itr);

    app->ReceivePhase([&]() { handle.Receive(); });

    support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, T_right);

    if (itr > 1 && flag == 0) {
      break;
    }

    support::PrintTempStats(
      pmesh, *fem.x, "App B after receive and before solve", itr);

    double err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("B before solve abs error=%e\n", err);

    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::PrintTempStats(pmesh, *fem.x, "App B after solve", itr);

    err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("B abs error=%e\n", err);

    support::SaveFields(outB, fem, itr);

    app->SendPhase([&]() {
      handle.Send();
      gdi->Send(&residual, "residual", 1);
    });

    app->ReceivePhase([&]() {
      auto done = gdi->Receive("done", 1)[0];

      while (!done) {
        sleep(1);
        done = gdi->Receive("done", 1)[0];
      }

      flag = gdi->Receive("flag", 1)[0];
      std::printf("received flag at B=%ld\n", static_cast<long>(flag));
    });

    ++itr;
  } while (flag);
}

void coupler(MPI_Comm comm,
             const std::string mesh_A_file,
             const std::string mesh_B_file)
{
  Omega_h::Library lib(nullptr, nullptr, comm);
  auto world = lib.world();

  Omega_h::Mesh mesh_A(&lib);
  Omega_h::binary::read(mesh_A_file, world, &mesh_A);

  Omega_h::Mesh mesh_B(&lib);
  Omega_h::binary::read(mesh_B_file, world, &mesh_B);

  const auto dim = mesh_A.dim();
  const auto nverts = mesh_A.nverts();

  const std::string coupler_name = "mfem_coupler";
  const std::string app_A_name = "client_A";
  const std::string app_B_name = "client_B";
  const std::string field_name = "temp";

  dtype initial_temp = 280.0;
  Omega_h::Read<dtype> init(nverts, initial_temp);

  mesh_A.add_tag<dtype>(Omega_h::VERT, field_name, 1, init);
  mesh_B.add_tag<dtype>(Omega_h::VERT, field_name, 1, init);

  pcms::LO tag = 0;
  auto is_overlap_A = support::create_mask(mesh_A, "domain", tag);
  auto is_overlap_B = support::create_mask(mesh_B, "domain", tag);

  auto partition = MakeRCBPartition(dim);

  pcms::Coupler cpl(coupler_name, comm, true, partition);

  auto* app_A = cpl.AddApplication(app_A_name);
  auto* app_B = cpl.AddApplication(app_B_name);

  pcms::Real fill_value = 0.0;

  auto layout_A = pcms::CreateLagrangeLayout(
    mesh_A, 1, 1, pcms::CoordinateSystem::Cartesian, "global");
  auto field_A = layout_A->CreateFieldReal();
  field_A->SetOutOfBoundsMode(pcms::OutOfBoundsMode::FILL, fill_value);

  auto layout_B = pcms::CreateLagrangeLayout(
    mesh_B, 1, 1, pcms::CoordinateSystem::Cartesian, "global");
  auto field_B = layout_B->CreateFieldReal();
  field_B->SetOutOfBoundsMode(pcms::OutOfBoundsMode::FILL, fill_value);

  app_A->SetLayoutOverlapMask(
    field_name,
    std::make_unique<pcms::OverlapMask>(
      static_cast<size_t>(layout_A->GetNumOwnedDofHolder()), is_overlap_A));

  app_B->SetLayoutOverlapMask(
    field_name,
    std::make_unique<pcms::OverlapMask>(
      static_cast<size_t>(layout_B->GetNumOwnedDofHolder()), is_overlap_B));

  app_A->AddLayout(field_name, layout_A);
  app_B->AddLayout(field_name, layout_B);

  auto handle_A = app_A->AddField(field_name, field_A);
  auto handle_B = app_B->AddField(field_name, field_B);

  auto gdi_A = app_A->Add_GDI<pcms::GO>("global_comm", comm);
  auto gdi_B = app_B->Add_GDI<pcms::GO>("global_comm", comm);

  GO flag = 1;
  GO done = 0;
  int itr = 1;

  const float tol = 1e-3;

  do {
    done = 0;

    auto dof_C =
      Omega_h::deep_copy(mesh_A.get_array<dtype>(0, field_name));

    pcms::GO residual_A = 0;
    pcms::GO residual_B = 0;

    app_A->ReceivePhase([&]() {
      handle_A.Receive();
      residual_A = gdi_A->Receive("residual", 1)[0];
    });

    std::printf("received residual at coupler from A = %ld\n",
                static_cast<long>(residual_A));

    auto dof_A = mesh_A.get_array<dtype>(0, field_name);

    double errA = support::ComputeAbsoluteError(mesh_A);
    support::PrintTempStats(mesh_A, "Coupler mesh A", itr);
    std::printf("iteration=%d A abs error=%e\n", itr, errA);

    auto rms = support::ComputeRMS(Omega_h::Read<dtype>(dof_C), dof_A);
    std::printf("rms received at coupler from A: %f\n", rms);

    flag = (rms > tol);

    CheckCoordinateOverlap(mesh_A, mesh_B, 0.4, 0.6, 0.0, 1.0, 1e-10);

    support::PrintTempStats(mesh_B, "Coupler mesh_B before interpolation", itr);

    auto before =
      Omega_h::deep_copy(mesh_B.get_array<dtype>(0, field_name));

    pcms::interpolate_field2(*field_A, *field_B);

    auto after = mesh_B.get_array<dtype>(0, field_name);

    std::cout << "interp rms B = " << support::ComputeRMS(before, after)
              << "\n";

    support::PrintTempStats(mesh_B, "Coupler mesh_B after interpolation", itr);

    app_B->SendPhase([&]() {
      handle_B.Send();
      gdi_B->Send(&flag, "flag", 1);
      gdi_B->Send(&done, "done", 1);
    });

    app_B->ReceivePhase([&]() {
      handle_B.Receive();
      residual_B = gdi_B->Receive("residual", 1)[0];
    });

    std::printf("received residual at coupler from B = %ld\n",
                static_cast<long>(residual_B));

    support::PrintTempStats(
      mesh_B, "Coupler mesh_B after receive back from app B", itr);

    support::PrintTempStats(mesh_A, "Coupler mesh_A before interpolation", itr);

    pcms::interpolate_field2(*field_B, *field_A);

    support::PrintTempStats(mesh_A, "Coupler mesh_A after interpolation", itr);

    dof_A = mesh_A.get_array<dtype>(0, field_name);
    rms = support::ComputeRMS(Omega_h::Read<dtype>(dof_C), dof_A);

    flag = (rms > tol);

    app_A->SendPhase([&]() {
      handle_A.Send();
      gdi_A->Send(&flag, "flag", 1);
      gdi_A->Send(&done, "done", 1);
    });

    std::printf("rms received at coupler after B->A interpolation: %f\n", rms);

    done = 1;

    app_A->SendPhase([&]() {
      gdi_A->Send(&flag, "flag", 1);
      gdi_A->Send(&done, "done", 1);
    });

    app_B->SendPhase([&]() {
      gdi_B->Send(&flag, "flag", 1);
      gdi_B->Send(&done, "done", 1);
    });

    std::printf("sent flag %ld, with rms %f coupler to apps after itr = %d\n",
                static_cast<long>(flag), rms, itr);

    double errB = support::ComputeAbsoluteError(mesh_B);
    std::printf("iteration=%d B abs error=%e\n", itr, errB);
    support::PrintTempStats(mesh_B, "Coupler mesh B", itr);

    ++itr;
  } while (flag);

  std::cout << "The system converged\n";
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rc = 0;

  {
    Kokkos::ScopeGuard kokkos(argc, argv);

    if (argc < 3) {
      std::cerr << "Usage:\n"
                << "  " << argv[0]
                << " -1 <omega_h_mesh_A> <omega_h_mesh_B>\n"
                << "  " << argv[0]
                << "  0 <mfem_mesh_A> <solver_type> <prec_type>\n"
                << "  " << argv[0]
                << "  1 <mfem_mesh_B> <solver_type> <prec_type>\n";
      MPI_Finalize();
      return EXIT_FAILURE;
    }

    const int clientId = std::atoi(argv[1]);
    REDEV_ALWAYS_ASSERT(clientId >= -1 && clientId <= 1);

    support::ThermalParams params;
    params.size = {0.6, 1.0};
    params.ne = {30, 30};
    params.kappa = 1.0;

    MPI_Comm comm = MPI_COMM_WORLD;

    try {
      switch (clientId) {
        case -1:
          REDEV_ALWAYS_ASSERT(argc >= 4);
          coupler(comm, argv[2], argv[3]);
          break;

        case 0:
          REDEV_ALWAYS_ASSERT(argc >= 5);
          app_A(comm, argv[2], params, argv[3], argv[4]);
          break;

        case 1:
          REDEV_ALWAYS_ASSERT(argc >= 5);
          app_B(comm, argv[2], params, argv[3], argv[4]);
          break;
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << "\n";
      rc = 1;
    }
  }

  MPI_Finalize();
  return rc;
}