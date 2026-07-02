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
                  std::string solver_type,
                  std::string prec_type)
{
  constexpr int order = 1;

  mfem::Mesh mesh(mesh_file, 1, 1);
  mfem::ParMesh pmesh(comm, mesh);

  const double T_left = 270.0;

  support::FEMSystem fem = support::Init_FEMSystem(&pmesh, order, params.kappa);

  const int bottom_bdr_attr = 1;
  const int left_bdr_attr   = 3;  // x = 0.4 physical left side
  const int right_bdr_attr  = 5;  // x = 1 overlap/interface side
  const int top_bdr_attr    = 7;
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

  auto handle = app->AddField(field_name, fs.CreateField<dtype>());

  auto gdi = app->Add_GDI<pcms::GO>("global_comm", comm);

  pcms::GO flag = 1;
  int itr = 1;
  pcms::GO done = 0;
  do {
    support::PrintTempStats(pmesh, *fem.x,overlap_view, "App A:: before solve", itr);

    support::SaveFields(outA, fem, itr);

    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::SaveFields(outA, fem, itr+1);
    support::PrintTempStats(pmesh, *fem.x, overlap_view,"App A:: after solve", itr);

    double err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("App A:: abs error=%e\n", err);

    app->SendPhase([&]() {
      handle.Send();
      gdi->Send(&residual, "residual", 1);
    });

    app->ReceivePhase([&]() { handle.Receive(); });

    app->ReceivePhase([&]() {
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
                  std::string solver_type,
                  std::string prec_type)
{
  constexpr int order = 1;

  mfem::Mesh mesh(mesh_file, 1, 1);
  mfem::ParMesh pmesh(comm, mesh);

  const double T_right = 300.0;

  support::FEMSystem fem = support::Init_FEMSystem(&pmesh, order, params.kappa);

  const int bottom_bdr_attr = 1;
  const int left_bdr_attr   = 3;  // x = 0.4 overlap/interface side
  const int right_bdr_attr  = 5;  // x = 1 physical right side
  const int top_bdr_attr    = 7;
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

  auto handle = app->AddField(field_name, fs.CreateField<dtype>());

  auto gdi = app->Add_GDI<pcms::GO>("global_comm", comm);

  pcms::GO flag = 1;
  int itr = 1;

  do {
    support::PrintTempStats(pmesh, *fem.x, overlap_view, "App B:: before receive", itr);
    support::SaveFields(outB, fem, itr);
    app->ReceivePhase([&]() { handle.Receive(); });
    support::PrintTempStats(pmesh, *fem.x, overlap_view, "App B:: after receive and before solve", itr);

    support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, T_right);

    support::SaveFields(outB, fem, itr+1);

    if (itr > 1 && flag == 0) {
      break;
    }


    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::PrintTempStats(pmesh, *fem.x, overlap_view, "App B:: after solve", itr);

    auto err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("App B:: abs error=%e\n", err);

    support::SaveFields(outB, fem, itr + 2);

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
      std::printf("App B:: received flag at B=%ld\n", static_cast<long>(flag));
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

  pcms::Real fill_value = 3;

  auto fs_A = pcms::LagrangeFunctionSpace::FromMesh(
    mesh_A, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto layout_A = fs_A.GetLayout();
  auto field_A = fs_A.CreateField<dtype>(pcms::FieldMetadata{});


  auto fs_B = pcms::LagrangeFunctionSpace::FromMesh(
    mesh_B, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto field_B = fs_B.CreateField<dtype>(pcms::FieldMetadata{});
  auto layout_B = fs_B.GetLayout();

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

  pcms::OutOfBoundsPolicy policy{pcms::OutOfBoundsMode::FILL, fill_value};
  pcms::Interpolator<pcms::Real> interpolator_A2B(fs_A, fs_B, policy);

  pcms::Interpolator<pcms::Real> interpolator_B2A(fs_B, fs_A, policy);

  auto handle_A = app_A->AddField(field_name, std::move(field_A));
  auto handle_B = app_B->AddField(field_name, std::move(field_B));


  auto gdi_A = app_A->Add_GDI<pcms::GO>("global_comm", comm);
  auto gdi_B = app_B->Add_GDI<pcms::GO>("global_comm", comm);

  pcms::GO flag = 1;
  pcms::GO done = 0;
  int itr = 1;

  const float tol = 1e-3;

  do {
    done = 0;

    auto c1_view = handle_A.GetField().GetDOFHolderDataHost();
    Kokkos::View<double*, pcms::HostMemorySpace> data_C(
        "data_C",
        c1_view.size());

    pcms::Rank1View<double, pcms::HostMemorySpace> stored_C{
      data_C.data(),
      data_C.extent(0)};

    Kokkos::parallel_for(
        "copy_before",
        c1_view.size(),
        KOKKOS_LAMBDA(const int i) {
          stored_C(i) = c1_view(i);
        });
    Kokkos::fence();
    pcms::GO residual_A = 0;
    pcms::GO residual_B = 0;

    app_A->ReceivePhase([&]() {
      handle_A.Receive();
      residual_A = gdi_A->Receive("residual", 1)[0];
    });
    c1_view = handle_A.GetField().GetDOFHolderDataHost();
    int counter=0;
    for (int i=0; i<c1_view.size(); i++) {
      //x`x`printf(" Index: %d, field value %f.\n",i, c1_view[i]);
      if (c1_view(i)==0) {counter++;}
    }
    printf("Number of zeroes in the receive from A to C: %d\n", counter);

    std::printf("Itr : %d , coupler :: received residual at coupler from A = %ld\n", itr,
                static_cast<long>(residual_A));

    c1_view = handle_A.GetField().GetDOFHolderDataHost();

    double errA = support::ComputeAbsoluteError(mesh_A);

    std::printf("Itr : %d , coupler :: After receive from A abs error=%e\n", itr, errA);

    auto rms = support::ComputeRMS(stored_C, c1_view);
    std::printf("Itr : %d , coupler :: rms between A & AC: %f\n", itr, rms);

    flag = (rms > tol);

    //CheckCoordinateOverlap(mesh_A, mesh_B, 0.4, 0.6, 0.0, 1.0, 1e-10);
    auto c2_view = handle_B.GetField().GetDOFHolderDataHost();

    Kokkos::View<double*, pcms::HostMemorySpace> data_CB(
        "data_CB",
        c2_view.size());

    pcms::Rank1View<double, pcms::HostMemorySpace> c2_pview{
      data_CB.data(),
      data_CB.extent(0)};

    Kokkos::parallel_for(
        "stored_CB",
        c1_view.size(),
        KOKKOS_LAMBDA(const int i) {
          c2_pview(i) = c2_view(i);
        });
    Kokkos::fence();

    interpolator_A2B.Apply(handle_A.GetField(), handle_B.GetField());

    c2_view = handle_B.GetField().GetDOFHolderDataHost();

    std::cout << "Itr : "<<itr<<" , coupler :: interp rms C1 to C2 on C2 = " << support::ComputeRMS(c2_pview, c2_view)
              << "\n";

    app_B->SendPhase([&]() {
      handle_B.Send();
      gdi_B->Send(&flag, "flag", 1);
      gdi_B->Send(&done, "done", 1);
    });
    counter =0;
    for (int i=0; i<c2_view.size(); i++) {
      if (c2_view(i)==0) {counter++;}
    }

    printf("Number of zeroes in the send from C to B: %d\n", counter);

    app_B->ReceivePhase([&]() {
      handle_B.Receive();
      residual_B = gdi_B->Receive("residual", 1)[0];
    });

    std::printf("Itr : %d , coupler :: received residual at coupler from B = %ld\n", itr,
                static_cast<long>(residual_B));


    Kokkos::View<double*, pcms::HostMemorySpace> data_BC(
        "data_BC",
        c1_view.size());

    pcms::Rank1View<double, pcms::HostMemorySpace> c1_pview{
      data_BC.data(),
      data_BC.extent(0)};

    Kokkos::parallel_for(
        "copy",
        data_C.size(),
        KOKKOS_LAMBDA(const int i) {
          c1_pview(i) = c1_view(i);
        });
    Kokkos::fence();

    interpolator_B2A.Apply(handle_B.GetField(), handle_A.GetField());

    c1_view = handle_A.GetField().GetDOFHolderDataHost();

    rms = support::ComputeRMS(c1_view,c1_pview);
    std::cout << "Itr : "<<itr<<" , coupler :: interp r2s C1 1o C2 1n C2 = " << support::ComputeRMS(c2_pview, c2_view)
              << "\n";

    flag = (rms > tol);

    app_A->SendPhase([&]() {
      handle_A.Send();
      gdi_A->Send(&flag, "flag", 1);
      gdi_A->Send(&done, "done", 1);
    });

    done = 1;

    app_A->SendPhase([&]() {
      gdi_A->Send(&flag, "flag", 1);
      gdi_A->Send(&done, "done", 1);
    });

    app_B->SendPhase([&]() {
      gdi_B->Send(&flag, "flag", 1);
      gdi_B->Send(&done, "done", 1);
    });

    std::printf("Itr : %d , coupler :: sent flag %ld, with rms %f coupler to both clients.\n",
                itr, static_cast<long>(flag), rms);

    double errB = support::ComputeAbsoluteError(mesh_B);
    std::printf("Itr : %d , coupler :: abs error=%e\n", itr, errB);

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