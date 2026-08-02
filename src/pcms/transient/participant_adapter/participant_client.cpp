#include "pcms/transient/participant_adapter/participant_client.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

namespace pcms::transient
{
using protocol::Command;

ParticipantClient::ParticipantClient(Application& application,
                                     MPI_Comm mpi_comm,
                                     Participant& participant,
                                     Configuration configuration)
  : participant_(&participant),
    configuration_(std::move(configuration)),
    control_(application, mpi_comm)
{
  if (configuration_.produced_interface.empty() ||
      configuration_.consumed_interface.empty() ||
      configuration_.consumed_interface_size == 0) {
    throw std::invalid_argument("ParticipantClient: incomplete configuration");
  }
}

void ParticipantClient::ConfigureFieldExchange(FieldExchange exchange)
{
  if (!exchange.send_produced_field || !exchange.receive_consumed_field) {
    throw std::invalid_argument(
      "ParticipantClient: incomplete PCMS field exchange");
  }
  field_exchange_ = std::move(exchange);
}

void ParticipantClient::Run()
{
  if (!field_exchange_.send_produced_field ||
      !field_exchange_.receive_consumed_field) {
    throw std::runtime_error(
      "ParticipantClient: PCMS field exchange is not configured");
  }
  const Capabilities capabilities = participant_->GetCapabilities();
  auto hello = protocol::Message::For(Command::hello);
  hello.integers[1] =
    static_cast<std::int64_t>(configuration_.consumed_interface_size);
  hello.integers[2] = capabilities.can_restart ? 1 : 0;
  hello.integers[3] = capabilities.has_dense_output ? 1 : 0;
  hello.integers[4] = capabilities.reports_qoi ? 1 : 0;
  control_.Send(hello);
  const auto hello_response = control_.Receive();
  if (hello_response.GetCommand() != Command::ok) {
    throw std::runtime_error("ParticipantClient: server rejected handshake");
  }

  std::map<std::int64_t, Checkpoint> checkpoints;
  bool running = true;
  while (running) {
    const auto request = control_.Receive();

    switch (request.GetCommand()) {
      case Command::save: {
        const auto token = request.integers[1];
        checkpoints.insert_or_assign(token, participant_->Save());
        control_.Send(protocol::Message::For(Command::ok));
        break;
      }
      case Command::restore: {
        const auto token = request.integers[1];
        const auto checkpoint = checkpoints.find(token);
        if (checkpoint == checkpoints.end())
          throw std::runtime_error(
            "ParticipantClient: unknown checkpoint token");
        participant_->Restore(checkpoint->second);
        control_.Send(protocol::Message::For(Command::ok));
        break;
      }
      case Command::advance:
        participant_->AdvanceTo(request.scalars[0]);
        control_.Send(protocol::Message::For(Command::ok));
        break;
      case Command::get_field: {
        field_exchange_.send_produced_field();
        control_.Send(protocol::Message::For(Command::ok));
        break;
      }
      case Command::set_boundary: {
        if (request.integers[1] !=
            static_cast<std::int64_t>(configuration_.consumed_interface_size)) {
          throw std::runtime_error(
            "ParticipantClient: interface size mismatch");
        }
        field_exchange_.receive_consumed_field();
        participant_->SetInterface(configuration_.consumed_interface,
                                   InterfaceState{});
        control_.Send(protocol::Message::For(Command::ok));
        break;
      }
      case Command::get_scalar: {
        if (!configuration_.scalar_diagnostic)
          throw std::runtime_error(
            "ParticipantClient: no scalar diagnostic configured");
        auto response = protocol::Message::For(Command::ok);
        response.scalars[0] = configuration_.scalar_diagnostic();
        control_.Send(response);
        break;
      }
      case Command::shutdown:
        control_.Send(protocol::Message::For(Command::ok));
        running = false;
        break;
      default: throw std::runtime_error("ParticipantClient: unknown command");
    }
  }
}

} // namespace pcms::transient
