#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>
#include "test_support.h"
#include "pcms/coupler/coupler.hpp"
#include <pcms/utility/types.h>

static constexpr bool done = true;
static constexpr int COMM_ROUNDS = 1;

void xgc_delta_f(MPI_Comm comm)
{
  pcms::Coupler coupler("proxy_couple", comm, false, {});
  pcms::Application* app = coupler.AddApplication("proxy_couple_xgc_delta_f");

  auto gdi = app->AddData<pcms::GO>("global_comm", comm);

  Kokkos::View<long*, pcms::HostMemorySpace> mean_buffer("mean_buffer", 1);

  pcms::Rank1View<long, pcms::HostMemorySpace> mean{mean_buffer.data(),
                                                    mean_buffer.extent(0)};

  mean[0] = 16;

  do {
    for (int i = 0; i < COMM_ROUNDS; ++i) {
      app->BeginSendPhase();
      gdi.Send(mean, "mean");
      app->EndSendPhase();
      printf("delta Sent mean:%ld\n", mean[0]);
      app->BeginReceivePhase();
      gdi.Receive(mean, "mean");
      app->EndReceivePhase();
      mean[0] = mean[0] / 2;
    }
  } while (!done);
  printf("final Mean = %ld\n", mean[0]);
  assert(std::fabs(mean[0] - 1.0) < 1e-12);
  printf("GDI test successful.\n");
}
void xgc_total_f(MPI_Comm comm)
{
  pcms::Coupler coupler("proxy_couple", comm, false, {});
  pcms::Application* app = coupler.AddApplication("proxy_couple_xgc_total_f");

  auto GDI = app->AddData<pcms::GO>("global_comm", comm);
  Kokkos::View<long*, pcms::HostMemorySpace> mean_buffer("mean_buffer", 1);

  pcms::Rank1View<long, pcms::HostMemorySpace> mean{mean_buffer.data(),
                                                    mean_buffer.extent(0)};

  do {
    for (int i = 0; i < COMM_ROUNDS; ++i) {
      app->BeginReceivePhase();
      GDI.Receive(mean, "mean");
      app->EndReceivePhase();
      printf("total Recieved mean:%ld\n", mean[0]);
      mean[0] = mean[0] / 2;
      app->BeginSendPhase();
      GDI.Send(mean, "mean");
      app->EndSendPhase();
      printf("total Sent mean:%ld\n", mean[0]);
    }
  } while (!done);
}
void xgc_coupler(MPI_Comm comm)
{
  // Define Partition
  redev::LO dim = 3;
  redev::LOs ranks(1);
  std::iota(ranks.begin(), ranks.end(), 0);
  redev::Reals cuts = {0};
  auto partition = redev::Partition{redev::RCBPtn{dim, ranks, cuts}};

  pcms::Coupler cpl("proxy_couple", comm, true, partition);
  auto* total_f = cpl.AddApplication("proxy_couple_xgc_total_f");
  auto* delta_f = cpl.AddApplication("proxy_couple_xgc_delta_f");

  auto GDI_total = total_f->AddData<pcms::GO>("global_comm", comm);
  auto GDI_delta = delta_f->AddData<pcms::GO>("global_comm", comm);

  Kokkos::View<long*, pcms::HostMemorySpace> mean_buffer("mean_buffer", 1);

  pcms::Rank1View<long, pcms::HostMemorySpace> mean{mean_buffer.data(),
                                                    mean_buffer.extent(0)};

  do {
    for (int i = 0; i < COMM_ROUNDS; ++i) {
      delta_f->BeginReceivePhase();
      GDI_delta.Receive(mean, "mean");
      delta_f->EndReceivePhase();
      printf("delta Received mean:%ld\n", mean[0]);
      mean[0] = mean[0] / 2;
      const auto msg_size = mean.size();
      total_f->BeginSendPhase();
      GDI_total.Send(mean, "mean");
      total_f->EndSendPhase();
      printf("total sent mean:%ld\n", mean[0]);
      total_f->BeginReceivePhase();
      GDI_total.Receive(mean, "mean");
      total_f->EndReceivePhase();
      printf("delta Received mean:%ld\n", mean[0]);
      mean[0] = mean[0] / 2;
      delta_f->BeginSendPhase();
      GDI_delta.Send(mean, "mean");
      delta_f->EndSendPhase();
      printf("delta sent mean:%ld\n", mean[0]);
    }
  } while (!done);
}

int main(int argc, char** argv)
{
  try {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    OMEGA_H_CHECK(argc == 2);

    const auto clientId = std::atoi(argv[1]);
    REDEV_ALWAYS_ASSERT(clientId >= -1 && clientId <= 1);

    MPI_Comm comm = MPI_COMM_WORLD;

    switch (clientId) {
      case -1: xgc_coupler(comm); break;

      case 0: xgc_delta_f(comm); break;

      case 1: xgc_total_f(comm); break;
      default:
        std::cerr << "Unhandled client id; expected -1, 0, or 1\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught in main: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught in main\n";
    return 1;
  }
}
