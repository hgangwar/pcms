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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

using dtype = pcms::Real;

static redev::Partition MakeRCBPartition(int dim)
{
  redev::LOs ranks(1);
  std::iota(ranks.begin(), ranks.end(), 0);
  redev::Reals cuts = {0};
  return redev::Partition{redev::RCBPtn{dim, ranks, cuts}};
}

static void wait_briefly()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

struct CouplerStepperContent
{
  sunrealtype tcur = 0.0;
  sunrealtype dt = 0.01;
  sunrealtype tstop = 1e300;
  suncountertype nsteps = 0;

  double tol = 1e-8;
  bool converged = false;

  int nverts = 0;

  double rmsA = 0.0;
  double rmsB = 0.0;
  double rms = 0.0;
  double rms_norm = 0.0;
  double rel_rms = 0.0;
  double rel_norm_floor = 100.0;

  pcms::GO schwarz_flag = 1;
  pcms::GO advance_flag = 0;
  pcms::GO abort_flag = 0;
  pcms::GO done = 0;

  double omega = 0.5;

  pcms::GO residual_A = 0;
  pcms::GO residual_B = 0;

  int schwarz_iter = 0;
  int max_schwarz_iters = 20;

  int time_step = 0;

  std::function<void(N_Vector)> pack_A_block;
  std::function<void(N_Vector)> pack_B_block;

  std::function<void()> copy_A_to_stored_A;
  std::function<void()> copy_B_to_stored_B;
  std::function<void()> copy_A_to_stored_A_B2A;

  std::function<double()> rms_stored_A_to_A;
  std::function<double()> rms_stored_B_to_B;
  std::function<double()> rms_stored_A_B2A_to_A;
  std::function<double()> rms_current_A_norm;

  std::function<void(dtype)> send_target_time_to_A;
  std::function<void(dtype)> send_target_time_to_B;

  std::function<void()> recv_A_field;
  std::function<void()> interp_A_to_B;
  std::function<void()> relax_B_field;
  std::function<void()> send_B_field;
  std::function<void()> recv_B_field;
  std::function<void()> interp_B_to_A;
  std::function<void()> relax_A_field;
  std::function<void()> send_A_field;

  std::function<void(pcms::GO, pcms::GO, pcms::GO, pcms::GO)> send_status_to_A;
  std::function<void(pcms::GO, pcms::GO, pcms::GO, pcms::GO)> send_status_to_B;
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
    C->schwarz_iter = 0;
    C->schwarz_flag = 1;
    C->advance_flag = 0;
    C->abort_flag = 0;
    C->done = 0;
    C->converged = false;

    while (!C->converged && C->schwarz_iter < C->max_schwarz_iters) {
      C->done = 0;
      C->advance_flag = 0;

      const int k = C->schwarz_iter + 1;

      const dtype target_time = static_cast<dtype>(tout);
      C->send_target_time_to_A(target_time);
      C->send_target_time_to_B(target_time);

      C->copy_A_to_stored_A();

      std::printf("[coupler] step=%d t_next=%g k=%d receive A\n",
                  C->time_step + 1, tout, k);
      C->recv_A_field();

      C->rmsA = C->rms_stored_A_to_A();
      C->pack_A_block(y);

      std::printf("[coupler] step=%d t_next=%g k=%d rms A recv=%e\n",
                  C->time_step + 1, tout, k, C->rmsA);

      C->copy_B_to_stored_B();

      std::printf("[coupler] step=%d t_next=%g k=%d interp A->B\n",
                  C->time_step + 1, tout, k);
      C->interp_A_to_B();
      C->relax_B_field();

      C->rmsB = C->rms_stored_B_to_B();

      std::printf("[coupler] step=%d t_next=%g k=%d rms A2B=%e\n",
                  C->time_step + 1, tout, k, C->rmsB);

      std::printf("[coupler] step=%d t_next=%g k=%d send B\n",
                  C->time_step + 1, tout, k);
      C->send_B_field();

      std::printf("[coupler] step=%d t_next=%g k=%d receive B\n",
                  C->time_step + 1, tout, k);
      C->recv_B_field();

      C->copy_A_to_stored_A_B2A();
      C->pack_B_block(y);

      std::printf("[coupler] step=%d t_next=%g k=%d interp B->A\n",
                  C->time_step + 1, tout, k);
      C->interp_B_to_A();
      C->relax_A_field();

      C->rms = C->rms_stored_A_B2A_to_A();
      C->rms_norm = std::max(C->rms_current_A_norm(), C->rel_norm_floor);
      C->rel_rms = C->rms / std::max(C->rms_norm, 1.0e-14);

      C->converged = (C->rel_rms <= C->tol);
      const bool out_of_schwarz_iters = (!C->converged && k >= C->max_schwarz_iters);

      C->abort_flag = out_of_schwarz_iters ? 1 : 0;
      C->schwarz_flag = (C->converged || out_of_schwarz_iters) ? 0 : 1;
      C->advance_flag = C->converged ? 1 : 0;

      C->pack_A_block(y);

      std::printf("[coupler] step=%d t_next=%g k=%d rms B2A=%e rel_rms=%e norm=%e converged=%d\n",
                  C->time_step + 1,
                  tout,
                  k,
                  C->rms,
                  C->rel_rms,
                  C->rms_norm,
                  static_cast<int>(C->converged));

      std::printf("[coupler] step=%d t_next=%g k=%d send A\n",
                  C->time_step + 1, tout, k);
      C->send_A_field();

      C->done = 1;

      C->send_status_to_A(C->schwarz_flag, C->advance_flag, C->abort_flag, C->done);
      C->send_status_to_B(C->schwarz_flag, C->advance_flag, C->abort_flag, C->done);

      std::printf(
        "[coupler] step=%d t_next=%g k=%d sent schwarz_flag=%ld advance_flag=%ld abort_flag=%ld done=%ld\n",
        C->time_step + 1,
        tout,
        k,
        static_cast<long>(C->schwarz_flag),
        static_cast<long>(C->advance_flag),
        static_cast<long>(C->abort_flag),
        static_cast<long>(C->done));

      C->schwarz_iter++;
    }

    if (!C->converged) {
      std::cerr << "[coupler] Schwarz failed at t_next=" << tout
                << " after " << C->max_schwarz_iters << " iterations.\n";
      SUNStepper_SetLastFlag(stepper, SUN_ERR_EXT_FAIL);
      return SUN_ERR_EXT_FAIL;
    }

    C->tcur = tout;
    C->time_step++;
    C->nsteps++;

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
  C->time_step = 0;
  C->schwarz_iter = 0;
  C->converged = false;
  C->rmsA = 0.0;
  C->rmsB = 0.0;
  C->rms = 0.0;
  C->rms_norm = 0.0;
  C->rel_rms = 0.0;
  C->schwarz_flag = 1;
  C->advance_flag = 0;
  C->abort_flag = 0;
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
  *fem.x = 280.0;
  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, left_bdr_attr, T_left);
  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, 280.0);

  mfem::ParGridFunction T_old(fem.fes);
  T_old = *fem.x;

  support::OutputPack outA("transient_schwarz_A", pmesh, *fem.fes);
  outA.pvd.RegisterField("T_A", fem.x);
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
  auto time_gdi = app->Add_GDI<dtype>("time_comm", comm);

  const double t_final = params.t_final;

  support::GaussianPulseCoefficient source(
    5000.0, 0.5, 0.5, 0.08, 0.20, 0.50);

  double time = 0.0;
  int time_step = 0;

  while (time < t_final - 1.0e-14) {
    double target_time = time;

    pcms::GO schwarz_flag = 1;
    pcms::GO advance_flag = 0;
    pcms::GO abort_flag = 0;

    int schwarz_iter = 1;

    while (!advance_flag) {
      app->ReceivePhase([&]() {
        target_time = time_gdi->Receive("target_time", 1)[0];
      });

      source.SetTime(target_time);

      support::PrintTempStats(pmesh, *fem.x, overlap_view,
                              "App A:: before transient solve", schwarz_iter);

      support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, left_bdr_attr, T_left);

      mfem::ParGridFunction T_stage_old(fem.fes);
      T_stage_old = T_old;

      double residual = 0.0;
      double local_time = time;
      while (local_time < target_time - 1.0e-14) {
        const double local_dt =
          std::min(params.dt, target_time - local_time);
        source.SetTime(local_time + local_dt);

        residual = support::SolveTransientHeatBE(fem,
                                                 T_stage_old,
                                                 source,
                                                 params.kappa,
                                                 local_dt,
                                                 solver_type,
                                                 prec_type,
                                                 1e-8,
                                                 500);

        T_stage_old = *fem.x;
        local_time += local_dt;
      }

      support::PrintTempStats(pmesh, *fem.x, overlap_view,
                              "App A:: after transient solve", schwarz_iter);



      app->SendPhase([&]() {
        handle.Send();
        pcms::GO residual_go = static_cast<pcms::GO>(residual);
        gdi->Send(&residual_go, "residual", 1);
      });

      app->ReceivePhase([&]() { handle.Receive(); });

      app->ReceivePhase([&]() {
        pcms::GO done = 0;
        while (!done) {
          wait_briefly();
          done = gdi->Receive("done", 1)[0];
        }

        schwarz_flag = gdi->Receive("schwarz_flag", 1)[0];
        advance_flag = gdi->Receive("advance_flag", 1)[0];
        abort_flag = gdi->Receive("abort_flag", 1)[0];
      });

      std::printf(
        "App A:: target_time=%f schwarz_iter=%d schwarz_flag=%ld advance_flag=%ld abort_flag=%ld\n",
        target_time,
        schwarz_iter,
        static_cast<long>(schwarz_flag),
        static_cast<long>(advance_flag),
        static_cast<long>(abort_flag));

      if (abort_flag) {
        throw std::runtime_error("App A received coupler abort flag.");
      }

      if (!schwarz_flag && !advance_flag) {
        throw std::runtime_error("App A received inconsistent status flags.");
      }

      ++schwarz_iter;
    }

    T_old = *fem.x;
    time = target_time;
    ++time_step;
    support::SaveFields(outA, fem, time_step);
    std::printf("App A:: accepted time step %d at time %f\n", time_step, time);
  }
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
  *fem.x = 280.0;
  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, left_bdr_attr, 280.0);
  support::ApplyBoundaryConstantByAttr(pmesh, *fem.x, right_bdr_attr, T_right);

  mfem::ParGridFunction T_old(fem.fes);
  T_old = *fem.x;

  support::OutputPack outB("transient_schwarz_B", pmesh, *fem.fes);
  outB.pvd.RegisterField("T_B", fem.x);
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
  auto time_gdi = app->Add_GDI<dtype>("time_comm", comm);

  const double t_final = params.t_final;

  support::GaussianPulseCoefficient source(
    5000.0, 0.5, 0.5, 0.08, 0.20, 0.50);

  double time = 0.0;
  int time_step = 0;

  while (time < t_final - 1.0e-14) {
    double target_time = time;

    pcms::GO schwarz_flag = 1;
    pcms::GO advance_flag = 0;
    pcms::GO abort_flag = 0;

    int schwarz_iter = 1;

    while (!advance_flag) {
      app->ReceivePhase([&]() {
        target_time = time_gdi->Receive("target_time", 1)[0];
      });

      source.SetTime(target_time);

      support::PrintTempStats(pmesh, *fem.x, overlap_view,
                              "App B:: before receive", schwarz_iter);

      app->ReceivePhase([&]() { handle.Receive(); });

      support::PrintTempStats(pmesh, *fem.x, overlap_view,
                              "App B:: after receive before transient solve",
                              schwarz_iter);

      support::ApplyBoundaryConstantByAttr(pmesh, *fem.x,
                                           right_bdr_attr, T_right);

      mfem::ParGridFunction T_stage_old(fem.fes);
      T_stage_old = T_old;

      double residual = 0.0;
      double local_time = time;
      while (local_time < target_time - 1.0e-14) {
        const double local_dt =
          std::min(params.dt, target_time - local_time);
        source.SetTime(local_time + local_dt);

        residual = support::SolveTransientHeatBE(fem,
                                                 T_stage_old,
                                                 source,
                                                 params.kappa,
                                                 local_dt,
                                                 solver_type,
                                                 prec_type,
                                                 1e-8,
                                                 500);

        T_stage_old = *fem.x;
        local_time += local_dt;
      }

      support::PrintTempStats(pmesh, *fem.x, overlap_view,
                              "App B:: after transient solve", schwarz_iter);



      app->SendPhase([&]() {
        handle.Send();
        pcms::GO residual_go = static_cast<pcms::GO>(residual);
        gdi->Send(&residual_go, "residual", 1);
      });

      app->ReceivePhase([&]() {
        pcms::GO done = 0;
        while (!done) {
          wait_briefly();
          done = gdi->Receive("done", 1)[0];
        }

        schwarz_flag = gdi->Receive("schwarz_flag", 1)[0];
        advance_flag = gdi->Receive("advance_flag", 1)[0];
        abort_flag = gdi->Receive("abort_flag", 1)[0];
      });

      std::printf(
        "App B:: target_time=%f schwarz_iter=%d schwarz_flag=%ld advance_flag=%ld abort_flag=%ld\n",
        target_time,
        schwarz_iter,
        static_cast<long>(schwarz_flag),
        static_cast<long>(advance_flag),
        static_cast<long>(abort_flag));

      if (abort_flag) {
        throw std::runtime_error("App B received coupler abort flag.");
      }

      if (!schwarz_flag && !advance_flag) {
        throw std::runtime_error("App B received inconsistent status flags.");
      }

      ++schwarz_iter;
    }

    T_old = *fem.x;
    time = target_time;
    ++time_step;
    support::SaveFields(outB, fem, time_step);
    std::printf("App B:: accepted time step %d at time %f\n", time_step, time);
  }
}

void coupler(MPI_Comm comm,
             const std::string mesh_A_file,
             const std::string mesh_B_file,
             const support::ThermalParams& params)
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
  Omega_h::Read<dtype> init_A(mesh_A.nverts(), initial_temp);
  Omega_h::Read<dtype> init_B(mesh_B.nverts(), initial_temp);

  mesh_A.add_tag<dtype>(Omega_h::VERT, field_name, 1, init_A);
  mesh_B.add_tag<dtype>(Omega_h::VERT, field_name, 1, init_B);

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
  auto time_gdi_A = app_A->Add_GDI<dtype>("time_comm", comm);
  auto time_gdi_B = app_B->Add_GDI<dtype>("time_comm", comm);

  const double tol = 1e-3;
  const int max_schwarz_iters = 20;

  SUNContext sunctx = nullptr;
  if (SUNContext_Create(SUN_COMM_NULL, &sunctx) != SUN_SUCCESS) {
    throw std::runtime_error("SUNContext_Create failed in coupler.");
  }

  N_Vector y = N_VNew_Serial(mesh_A.nverts() + mesh_B.nverts(), sunctx);
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
  stepper_content->dt = params.coupling_dt;
  stepper_content->tcur = 0.0;
  stepper_content->tstop = params.t_final;
  stepper_content->tol = tol;
  stepper_content->rel_norm_floor = 100.0;
  stepper_content->omega = 1;
  stepper_content->nverts = nverts;
  stepper_content->max_schwarz_iters = max_schwarz_iters;

  stepper_content->send_target_time_to_A = [&](dtype target_time) {
    app_A->SendPhase([&]() {
      time_gdi_A->Send(&target_time, "target_time", 1);
    });
  };

  stepper_content->send_target_time_to_B = [&](dtype target_time) {
    app_B->SendPhase([&]() {
      time_gdi_B->Send(&target_time, "target_time", 1);
    });
  };

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

  stepper_content->rms_current_A_norm = [&]() {
    auto view = handle_A.GetField().GetDOFHolderDataHost();
    double sum_sq = 0.0;
    for (int i = 0; i < view.size(); ++i) {
      const double v = view(i);
      sum_sq += v * v;
    }
    return std::sqrt(sum_sq / static_cast<double>(view.size()));
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

  stepper_content->relax_B_field = [&]() {
    auto& field = handle_B.GetField();
    auto view = field.GetDOFHolderDataHost();

    Kokkos::View<double*, pcms::HostMemorySpace> relaxed_data(
      "relaxed_B", view.size());

    pcms::Rank1View<double, pcms::HostMemorySpace> relaxed{
      relaxed_data.data(), relaxed_data.extent(0)};

    const double omega = stepper_content->omega;

    Kokkos::parallel_for(
      "relax_B_field", view.size(),
      KOKKOS_LAMBDA(const int i) {
        relaxed(i) = omega * view(i) + (1.0 - omega) * stored_B(i);
      });
    Kokkos::fence();

    field.SetDOFHolderDataHost(relaxed);
  };

  stepper_content->send_B_field = [&]() {
    app_B->SendPhase([&]() { handle_B.Send(); });
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

  stepper_content->relax_A_field = [&]() {
    auto& field = handle_A.GetField();
    auto view = field.GetDOFHolderDataHost();

    Kokkos::View<double*, pcms::HostMemorySpace> relaxed_data(
      "relaxed_A", view.size());

    pcms::Rank1View<double, pcms::HostMemorySpace> relaxed{
      relaxed_data.data(), relaxed_data.extent(0)};

    const double omega = stepper_content->omega;

    Kokkos::parallel_for(
      "relax_A_field", view.size(),
      KOKKOS_LAMBDA(const int i) {
        relaxed(i) = omega * view(i) + (1.0 - omega) * stored_A_B2A(i);
      });
    Kokkos::fence();

    field.SetDOFHolderDataHost(relaxed);
  };

  stepper_content->send_A_field = [&]() {
    app_A->SendPhase([&]() { handle_A.Send(); });
  };

  stepper_content->send_status_to_A =
    [&](pcms::GO schwarz_flag, pcms::GO advance_flag, pcms::GO abort_flag, pcms::GO done) {
      app_A->SendPhase([&]() {
        gdi_A->Send(&done, "done", 1);
        gdi_A->Send(&schwarz_flag, "schwarz_flag", 1);
        gdi_A->Send(&advance_flag, "advance_flag", 1);
        gdi_A->Send(&abort_flag, "abort_flag", 1);
      });
    };

  stepper_content->send_status_to_B =
    [&](pcms::GO schwarz_flag, pcms::GO advance_flag, pcms::GO abort_flag, pcms::GO done) {
      app_B->SendPhase([&]() {
        gdi_B->Send(&done, "done", 1);
        gdi_B->Send(&schwarz_flag, "schwarz_flag", 1);
        gdi_B->Send(&advance_flag, "advance_flag", 1);
        gdi_B->Send(&abort_flag, "abort_flag", 1);
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

    const int offset = static_cast<int>(mesh_A.nverts());
    for (int i = 0; i < view.size(); ++i) {
      ydata[offset + i] = view(i);
    }
  };

  SUNStepper stepper = nullptr;
  if (CreateCouplerSUNStepper(sunctx, stepper_content, &stepper) !=
      SUN_SUCCESS) {
    N_VDestroy(y);
    SUNContext_Free(&sunctx);
    throw std::runtime_error("CreateCouplerSUNStepper failed.");
  }

  sunrealtype tret = 0.0;
  const double dt = stepper_content->dt;
  const double t_final = params.t_final;

  while (tret < t_final - 1.0e-14) {
    const sunrealtype tout = std::min<sunrealtype>(tret + dt, t_final);

    const SUNErrCode err = SUNStepper_Evolve(stepper, tout, y, &tret);
    if (err != SUN_SUCCESS) {
      SUNStepper_Destroy(&stepper);
      N_VDestroy(y);
      SUNContext_Free(&sunctx);
      throw std::runtime_error("Coupler SUNStepper_Evolve failed.");
    }

    std::cout << "[coupler] accepted physical step="
              << stepper_content->time_step
              << " time=" << tret
              << " schwarz_iters=" << stepper_content->schwarz_iter
              << " rms_A=" << stepper_content->rmsA
              << " rms_B=" << stepper_content->rmsB
              << " rms_final=" << stepper_content->rms
              << " rel_rms_final=" << stepper_content->rel_rms
              << "\n";
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
    params.dt = 0.05;          // app/local transient time step
    params.coupling_dt = 0.01;  // coupler target-time stride
    params.t_final = 1.0;

    MPI_Comm comm = MPI_COMM_WORLD;

    try {
      switch (clientId) {
        case -1:
          REDEV_ALWAYS_ASSERT(argc >= 4);
          coupler(comm, argv[2], argv[3], params);
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