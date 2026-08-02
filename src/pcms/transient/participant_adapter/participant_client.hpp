#pragma once

#include "pcms/coupler/coupler.hpp"
#include "pcms/transient/participant.hpp"
#include "pcms/transient/participant_adapter/participant_protocol.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace pcms::transient
{

// Client-side runtime that exposes any Participant implementation to a remote
// transient driver through the participant protocol.
class ParticipantClient
{
public:
  struct Configuration
  {
    std::string produced_interface;
    std::string consumed_interface;
    std::size_t consumed_interface_size = 0;
    std::function<Real()> scalar_diagnostic;
  };

  struct FieldExchange
  {
    std::function<void()> send_produced_field;
    std::function<void()> receive_consumed_field;
  };

  ParticipantClient(Application& application, MPI_Comm mpi_comm,
                    Participant& participant, Configuration configuration);

  void ConfigureFieldExchange(FieldExchange exchange);
  void Run();

private:
  Participant* participant_;
  Configuration configuration_;
  protocol::ControlChannel control_;
  FieldExchange field_exchange_;
};

} // namespace pcms::transient
