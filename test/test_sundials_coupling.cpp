#include "coupling_support.h"

#include <pcms/coupler/coupler.hpp>
#include <pcms/coupler/overlap_mask.h>
#include <pcms/field/function_space/mfem.h>
#include <pcms/field/layout/mfem.h>
#include <pcms/transfer/interpolator.h>

#include <Omega_h_file.hpp>
#include <mfem.hpp>
#include <redev.h>

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_stepper.h>
#include <sundials/sundials_types.h>



using dtype = pcms::Real;

static redev::Partition MakeRCBPartition(int dim)
{
  redev::LOs ranks(1);
  std::iota(ranks.begin(), ranks.end(), 0);
  redev::Reals cuts = {0};
  return redev::Partition{redev::RCBPtn{dim, ranks, cuts}};
}

struct CouplerStepperContent
{
  sunrealtype tcur = 0.0;
  sunrealtype dt = 1.0;
  sunrealtype tstop = 1e300;
  suncountertype nsteps = 0;

  double tol = 1e-8;
  bool converged = false;

  int nverts = 0;

  double rmsA = 0.0;
  double rmsB = 0.0;
  double rms = 0.0;

  pcms::GO flag = 1;
  pcms::GO done = 0;
  pcms::GO residual_A = 0;
  pcms::GO residual_B = 0;

  std::function<void(N_Vector)> pack_A_block;
  std::function<void(N_Vector)> pack_B_block;

  std::function<void()> copy_A_to_stored_A;
  std::function<void()> copy_B_to_stored_B;
  std::function<void()> copy_A_to_stored_A_B2A;

  std::function<double()> rms_stored_A_to_A;
  std::function<double()> rms_stored_B_to_B;
  std::function<double()> rms_stored_A_B2A_to_A;

  std::function<void()> recv_A_field;
  std::function<void()> interp_A_to_B;
  std::function<void()> send_B_field;
  std::function<void()> recv_B_field;
  std::function<void()> interp_B_to_A;
  std::function<void()> send_A_field;
  std::function<void(pcms::GO, pcms::GO)> send_status_to_A;
  std::function<void(pcms::GO, pcms::GO)> send_status_to_B;
};

static SUNErrCode CouplerStepper_Evolve(SUNStepper stepper,
                                        sunrealtype tout,
                                        N_Vector y,
                                        sunrealtype* tret)
{
  void* content_void = nullptr;
  if (SUNStepper_GetContent(stepper, &content_void) != SUN_SUCCESS) {
    return SUN_ERR_EXT_FAIL;
  }

  auto* C = static_cast<CouplerStepperContent*>(content_void);

  try {
    while (C->tcur < tout && C->tcur < C->tstop && !C->converged) {
      C->done = 0;

      C->copy_A_to_stored_A();

      std::printf("A -> receive -> C\n");
      C->recv_A_field();

      C->rmsA = C->rms_stored_A_to_A();
      C->flag = (C->rmsA > C->tol) ? 1 : 0;

      std::printf(
        "Itr : %ld , coupler :: received residual from A = %ld\n",
        static_cast<long>(C->nsteps + 1),
        static_cast<long>(C->residual_A));
      std::printf("Itr : %ld , coupler :: rms between A & AC: %f\n",
                  static_cast<long>(C->nsteps + 1), C->rmsA);

      C->pack_A_block(y);

      C->copy_B_to_stored_B();

      std::printf("C -> interpolate A -> B\n");
      C->interp_A_to_B();

      C->rmsB = C->rms_stored_B_to_B();
      std::printf("Itr : %ld , coupler :: interp rms C1 to C2 on C2 = %f\n",
                  static_cast<long>(C->nsteps + 1), C->rmsB);

      std::printf("C -> send -> B\n");
      C->send_B_field();


      std::printf("B -> receive -> C\n");
      C->recv_B_field();

      std::printf(
        "Itr : %ld , coupler :: received residual from B = %ld\n",
        static_cast<long>(C->nsteps + 1),
        static_cast<long>(C->residual_B));

      C->copy_A_to_stored_A_B2A();
      C->pack_B_block(y);

      std::printf("C -> interpolate B -> A\n");
      C->interp_B_to_A();

      C->rms = C->rms_stored_A_B2A_to_A();
      C->flag = (C->rms > C->tol) ? 1 : 0;
      C->converged = (C->flag == 0);

      C->pack_A_block(y);

      std::printf("Itr : %ld , coupler :: interp rms B to A on A = %f\n",
                  static_cast<long>(C->nsteps + 1), C->rms);

      std::printf("C -> send -> A\n");
      C->send_A_field();

      C->done = 1;
      C->send_status_to_A(C->flag, C->done);
      C->send_status_to_B(C->flag, C->done);

      std::printf(
        "Itr : %ld , coupler :: sent flag %ld, with rms %f coupler to both clients.\n",
        static_cast<long>(C->nsteps + 1), static_cast<long>(C->flag), C->rms);

      C->tcur += C->dt;
      C->nsteps++;
    }

    if (tret) {
      *tret = C->tcur;
    }

    SUNStepper_SetLastFlag(stepper, SUN_SUCCESS);
    return SUN_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "[coupler] CouplerStepper_Evolve failed: " << e.what()
              << std::endl;
    SUNStepper_SetLastFlag(stepper, SUN_ERR_EXT_FAIL);
    return SUN_ERR_EXT_FAIL;
  } catch (...) {
    std::cerr << "[coupler] CouplerStepper_Evolve failed with unknown exception."
              << std::endl;
    SUNStepper_SetLastFlag(stepper, SUN_ERR_EXT_FAIL);
    return SUN_ERR_EXT_FAIL;
  }
}

static SUNErrCode CouplerStepper_Reset(SUNStepper stepper,
                                       sunrealtype tR,
                                       N_Vector)
{
  void* content_void = nullptr;
  if (SUNStepper_GetContent(stepper, &content_void) != SUN_SUCCESS) {
    return SUN_ERR_EXT_FAIL;
  }

  auto* C = static_cast<CouplerStepperContent*>(content_void);
  C->tcur = tR;
  C->nsteps = 0;
  C->converged = false;
  C->rmsA = 0.0;
  C->rmsB = 0.0;
  C->rms = 0.0;
  C->flag = 1;
  C->done = 0;

  return SUN_SUCCESS;
}

static SUNErrCode CouplerStepper_GetNumSteps(SUNStepper stepper,
                                             suncountertype* nsteps)
{
  void* content_void = nullptr;
  if (SUNStepper_GetContent(stepper, &content_void) != SUN_SUCCESS) {
    return SUN_ERR_EXT_FAIL;
  }

  auto* C = static_cast<CouplerStepperContent*>(content_void);
  *nsteps = C->nsteps;
  return SUN_SUCCESS;
}

static SUNErrCode CouplerStepper_SetStopTime(SUNStepper stepper,
                                             sunrealtype tstop)
{
  void* content_void = nullptr;
  if (SUNStepper_GetContent(stepper, &content_void) != SUN_SUCCESS) {
    return SUN_ERR_EXT_FAIL;
  }

  auto* C = static_cast<CouplerStepperContent*>(content_void);
  C->tstop = tstop;
  return SUN_SUCCESS;
}

static SUNErrCode CouplerStepper_Destroy(SUNStepper stepper)
{
  void* content_void = nullptr;
  if (SUNStepper_GetContent(stepper, &content_void) != SUN_SUCCESS) {
    return SUN_ERR_EXT_FAIL;
  }

  delete static_cast<CouplerStepperContent*>(content_void);
  SUNStepper_SetContent(stepper, nullptr);
  return SUN_SUCCESS;
}

static SUNErrCode CreateCouplerSUNStepper(SUNContext sunctx,
                                          CouplerStepperContent* content,
                                          SUNStepper* stepper)
{
  SUNErrCode err = SUNStepper_Create(sunctx, stepper);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetContent(*stepper, content);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetEvolveFn(*stepper, CouplerStepper_Evolve);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetResetFn(*stepper, CouplerStepper_Reset);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetReInitFn(*stepper, CouplerStepper_Reset);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetStopTimeFn(*stepper, CouplerStepper_SetStopTime);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetGetNumStepsFn(*stepper, CouplerStepper_GetNumSteps);
  if (err != SUN_SUCCESS) { return err; }

  err = SUNStepper_SetDestroyFn(*stepper, CouplerStepper_Destroy);
  if (err != SUN_SUCCESS) { return err; }

  return SUN_SUCCESS;
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

  const int left_bdr_attr = 3;
  const int right_bdr_attr = 5;
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

  do {
    support::PrintTempStats(pmesh, *fem.x, overlap_view,
                            "App A:: before solve", itr);
    support::SaveFields(outA, fem, itr);

    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::SaveFields(outA, fem, itr + 1);
    support::PrintTempStats(pmesh, *fem.x, overlap_view,
                            "App A:: after solve", itr);

    const double err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("App A:: abs error=%e\n", err);

    app->SendPhase([&]() {
      handle.Send();
      gdi->Send(&residual, "residual", 1);
    });

    app->ReceivePhase([&]() { handle.Receive(); });

    app->ReceivePhase([&]() {
      pcms::GO done = 0;
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

  const int left_bdr_attr = 3;
  const int right_bdr_attr = 5;
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
    support::PrintTempStats(pmesh, *fem.x, overlap_view,
                            "App B:: before receive", itr);
    support::SaveFields(outB, fem, itr);

    app->ReceivePhase([&]() { handle.Receive(); });

    support::PrintTempStats(pmesh, *fem.x, overlap_view,
                            "App B:: after receive and before solve", itr);

    support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr,
                                         T_right);

    support::SaveFields(outB, fem, itr + 1);

    if (itr > 1 && flag == 0) {
      break;
    }

    auto residual = support::SolveSystem(fem, solver_type, prec_type, 1e-8, 500);

    support::PrintTempStats(pmesh, *fem.x, overlap_view,
                            "App B:: after solve", itr);

    const double err = support::ComputeAbsoluteError(pmesh, *fem.x);
    std::printf("App B:: abs error=%e\n", err);

    support::SaveFields(outB, fem, itr + 2);

    app->SendPhase([&]() {
      handle.Send();
      gdi->Send(&residual, "residual", 1);
    });

    app->ReceivePhase([&]() {
      pcms::GO done = 0;
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

  const pcms::LO tag = 0;
  auto is_overlap_A = support::create_mask(mesh_A, "domain", tag);
  auto is_overlap_B = support::create_mask(mesh_B, "domain", tag);

  auto partition = MakeRCBPartition(dim);

  pcms::Coupler cpl(coupler_name, comm, true, partition);

  auto* app_A = cpl.AddApplication(app_A_name);
  auto* app_B = cpl.AddApplication(app_B_name);

  const pcms::Real fill_value = 3.0;

  auto fs_A = pcms::LagrangeFunctionSpace::FromMesh(
    mesh_A, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto layout_A = fs_A.GetLayout();
  auto field_A = fs_A.CreateField<dtype>(pcms::FieldMetadata{});

  auto fs_B = pcms::LagrangeFunctionSpace::FromMesh(
    mesh_B, 1, 1, pcms::CoordinateSystem::Cartesian);
  auto layout_B = fs_B.GetLayout();
  auto field_B = fs_B.CreateField<dtype>(pcms::FieldMetadata{});

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
  pcms::Interpolator<dtype> interpolator_A2B(fs_A, fs_B, policy);
  pcms::Interpolator<dtype> interpolator_B2A(fs_B, fs_A, policy);

  auto handle_A = app_A->AddField(field_name, std::move(field_A));
  auto handle_B = app_B->AddField(field_name, std::move(field_B));

  auto gdi_A = app_A->Add_GDI<pcms::GO>("global_comm", comm);
  auto gdi_B = app_B->Add_GDI<pcms::GO>("global_comm", comm);

  const double tol = 1e-3;
  const int max_iters = 20;

  SUNContext sunctx = nullptr;
  if (SUNContext_Create(SUN_COMM_NULL, &sunctx) != SUN_SUCCESS) {
    throw std::runtime_error("SUNContext_Create failed in coupler.");
  }

  N_Vector y = N_VNew_Serial(2 * nverts, sunctx);
  if (!y) {
    SUNContext_Free(&sunctx);
    throw std::runtime_error("Failed to allocate coupler N_Vector state.");
  }
  N_VConst(SUN_RCONST(0.0), y);

  Kokkos::View<double*, pcms::HostMemorySpace> data_A_before(
    "data_A_before", layout_A->GetNumOwnedDofHolder());
  Kokkos::View<double*, pcms::HostMemorySpace> data_B_before(
    "data_B_before", layout_B->GetNumOwnedDofHolder());
  Kokkos::View<double*, pcms::HostMemorySpace> data_A_before_B2A(
    "data_A_before_B2A", layout_A->GetNumOwnedDofHolder());

  pcms::Rank1View<double, pcms::HostMemorySpace> stored_A{
    data_A_before.data(), data_A_before.extent(0)};
  pcms::Rank1View<double, pcms::HostMemorySpace> stored_B{
    data_B_before.data(), data_B_before.extent(0)};
  pcms::Rank1View<double, pcms::HostMemorySpace> stored_A_B2A{
    data_A_before_B2A.data(), data_A_before_B2A.extent(0)};

  auto* stepper_content = new CouplerStepperContent;
  stepper_content->dt = 1.0;
  stepper_content->tcur = 0.0;
  stepper_content->tstop = static_cast<sunrealtype>(max_iters);
  stepper_content->tol = tol;
  stepper_content->nverts = nverts;

  stepper_content->copy_A_to_stored_A = [&]() {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    Kokkos::parallel_for(
      "copy_A_to_stored_A", view.size(),
      KOKKOS_LAMBDA(const int i) { stored_A(i) = view(i); });
    Kokkos::fence();
  };

  stepper_content->copy_B_to_stored_B = [&]() {
    auto view = handle_B.GetField().GetDOFHolderDataHost();
    Kokkos::parallel_for(
      "copy_B_to_stored_B", view.size(),
      KOKKOS_LAMBDA(const int i) { stored_B(i) = view(i); });
    Kokkos::fence();
  };

  stepper_content->copy_A_to_stored_A_B2A = [&]() {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    Kokkos::parallel_for(
      "copy_A_to_stored_A_B2A", view.size(),
      KOKKOS_LAMBDA(const int i) { stored_A_B2A(i) = view(i); });
    Kokkos::fence();
  };

  stepper_content->rms_stored_A_to_A = [&]() {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    return support::ComputeRMS(stored_A, view);
  };

  stepper_content->rms_stored_B_to_B = [&]() {
    auto view = handle_B.GetField().GetDOFHolderDataHost();
    return support::ComputeRMS(stored_B, view);
  };

  stepper_content->rms_stored_A_B2A_to_A = [&]() {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    return support::ComputeRMS(view, stored_A_B2A);
  };

  stepper_content->recv_A_field = [&]() {
    app_A->ReceivePhase([&]() {
      handle_A.Receive();
      stepper_content->residual_A = gdi_A->Receive("residual", 1)[0];
    });
  };

  stepper_content->interp_A_to_B = [&]() {
    interpolator_A2B.Apply(handle_A.GetField(), handle_B.GetField());
  };

  stepper_content->send_B_field = [&]() {
    app_B->SendPhase([&]() {
      handle_B.Send();
      gdi_B->Send(&stepper_content->flag, "flag", 1);
      gdi_B->Send(&stepper_content->done, "done", 1);
    });
  };

  stepper_content->recv_B_field = [&]() {
    app_B->ReceivePhase([&]() {
      handle_B.Receive();
      stepper_content->residual_B = gdi_B->Receive("residual", 1)[0];
    });
  };

  stepper_content->interp_B_to_A = [&]() {
    interpolator_B2A.Apply(handle_B.GetField(), handle_A.GetField());
  };

  stepper_content->send_A_field = [&]() {
    app_A->SendPhase([&]() {
      handle_A.Send();
      gdi_A->Send(&stepper_content->flag, "flag", 1);
      gdi_A->Send(&stepper_content->done, "done", 1);
    });
  };

  stepper_content->send_status_to_A = [&](pcms::GO flag, pcms::GO done) {
    app_A->SendPhase([&]() {
      gdi_A->Send(&flag, "flag", 1);
      gdi_A->Send(&done, "done", 1);
    });
  };

  stepper_content->send_status_to_B = [&](pcms::GO flag, pcms::GO done) {
    app_B->SendPhase([&]() {
      gdi_B->Send(&flag, "flag", 1);
      gdi_B->Send(&done, "done", 1);
    });
  };

  stepper_content->pack_A_block = [&](N_Vector yy) {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    auto* ydata = N_VGetArrayPointer(yy);
    if (!ydata) {
      throw std::runtime_error("pack_A_block: N_Vector data pointer is null.");
    }
    for (int i = 0; i < view.size(); ++i) {
      ydata[i] = view(i);
    }
  };

  stepper_content->pack_B_block = [&](N_Vector yy) {
    auto view = handle_B.GetField().GetDOFHolderDataHost();
    auto* ydata = N_VGetArrayPointer(yy);
    if (!ydata) {
      throw std::runtime_error("pack_B_block: N_Vector data pointer is null.");
    }
    for (int i = 0; i < view.size(); ++i) {
      ydata[nverts + i] = view(i);
    }
  };

  SUNStepper stepper = nullptr;
  if (CreateCouplerSUNStepper(sunctx, stepper_content, &stepper) != SUN_SUCCESS) {
    N_VDestroy(y);
    SUNContext_Free(&sunctx);
    throw std::runtime_error("CreateCouplerSUNStepper failed.");
  }

  int itr = 1;
  sunrealtype tret = 0.0;

  while (!stepper_content->converged && itr <= max_iters) {
    const sunrealtype tout = tret + stepper_content->dt;

    const SUNErrCode err = SUNStepper_Evolve(stepper, tout, y, &tret);
    if (err != SUN_SUCCESS) {
      SUNStepper_Destroy(&stepper);
      N_VDestroy(y);
      SUNContext_Free(&sunctx);
      throw std::runtime_error("Coupler SUNStepper_Evolve failed.");
    }

    std::cout << "[coupler] itr=" << itr
              << " tret=" << tret
              << " rms_A=" << stepper_content->rmsA
              << " rms_B=" << stepper_content->rmsB
              << " rms_final=" << stepper_content->rms
              << " converged=" << stepper_content->converged << "\n";

    ++itr;
  }

  if (!stepper_content->converged) {
    std::cout << "[coupler] stopped at max_iters=" << max_iters << "\n";
  } else {
    std::cout << "[coupler] converged based on SUNDIALS coupler step\n";
  }

  SUNStepper_Destroy(&stepper);
  N_VDestroy(y);
  SUNContext_Free(&sunctx);
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rc = 0;

  {
    Kokkos::ScopeGuard kokkos(argc, argv);

    if (argc < 3) {
      std::cerr << "Usage:\n"
                << "  " << argv[0] << " -1 <omega_h_mesh_A> <omega_h_mesh_B>\n"
                << "  " << argv[0] << "  0 <mfem_mesh_A> <solver_type> <prec_type>\n"
                << "  " << argv[0] << "  1 <mfem_mesh_B> <solver_type> <prec_type>\n";
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