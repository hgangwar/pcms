#include "pcms/transient/participant_adapter/coupler_participant.hpp"

#include <stdexcept>
#include <utility>

namespace pcms::transient
{
using protocol::Command;

CouplerParticipant::CouplerParticipant(Application& application,
                                       MPI_Comm mpi_comm,
                                       Configuration configuration)
  : participant_name_(std::move(configuration.participant)),
    produced_interface_(std::move(configuration.produced_interface)),
    consumed_interface_(std::move(configuration.consumed_interface)),
    control_(application, mpi_comm)
{
  if (!configuration.defer_connect)
    Connect();
}

void CouplerParticipant::ConfigureFieldExchange(FieldExchange exchange)
{
  if (!exchange.receive_produced_field || !exchange.send_consumed_field) {
    throw std::invalid_argument(
      "CouplerParticipant: incomplete PCMS field exchange");
  }
  field_exchange_ = std::move(exchange);
}

void CouplerParticipant::Connect()
{
  if (connected_)
    return;

  const auto hello = control_.Receive();
  if (hello.GetCommand() != Command::hello || hello.integers[1] <= 0) {
    throw std::runtime_error(
      "CouplerParticipant: invalid participant handshake");
  }
  interface_size_ = static_cast<std::size_t>(hello.integers[1]);
  capabilities_ = Capabilities{/*can_restart=*/hello.integers[2] != 0,
                               /*has_dense_output=*/hello.integers[3] != 0,
                               /*reports_qoi=*/hello.integers[4] != 0};
  control_.Send(protocol::Message::For(Command::ok));
  connected_ = true;
}

std::string_view CouplerParticipant::Name() const
{
  return participant_name_;
}

void CouplerParticipant::AdvanceTo(Real target_time)
{
  auto request = protocol::Message::For(Command::advance);
  request.scalars[0] = target_time;
  ExpectOk(Request(request));
  current_time_ = target_time;
}

Checkpoint CouplerParticipant::Save() const
{
  const std::int64_t token = ++next_checkpoint_token_;
  auto request = protocol::Message::For(Command::save);
  request.integers[1] = token;
  ExpectOk(Request(request));
  return Checkpoint{current_time_, CheckpointToken{token}};
}

void CouplerParticipant::Restore(const Checkpoint& checkpoint)
{
  const auto& token = std::any_cast<const CheckpointToken&>(checkpoint.state);
  auto request = protocol::Message::For(Command::restore);
  request.integers[1] = token.value;
  ExpectOk(Request(request));
  current_time_ = checkpoint.time;
}

InterfaceState CouplerParticipant::GetInterface(std::string_view name) const
{
  if (name != produced_interface_)
    throw std::invalid_argument(
      "CouplerParticipant: unknown produced interface");

  if (!field_exchange_.receive_produced_field)
    throw std::runtime_error(
      "CouplerParticipant: PCMS field receive is not configured");
  control_.Send(protocol::Message::For(Command::get_field));
  field_exchange_.receive_produced_field();
  ExpectOk(control_.Receive());
  // The distributed field now resides in the PCMS coupler-side Field bound
  // by field_exchange_. The transient vector is produced by the coupling
  // backend only after PCMS interpolation.
  return InterfaceState{};
}

void CouplerParticipant::SetInterface(std::string_view name,
                                      const InterfaceState& state)
{
  if (name != consumed_interface_)
    throw std::invalid_argument(
      "CouplerParticipant: unknown consumed interface");
  if (state.Size() != interface_size_)
    throw std::invalid_argument("CouplerParticipant: interface size mismatch");

  if (!field_exchange_.send_consumed_field)
    throw std::runtime_error(
      "CouplerParticipant: PCMS field send is not configured");
  auto request = protocol::Message::For(Command::set_boundary);
  request.integers[1] = static_cast<std::int64_t>(state.Size());
  control_.Send(request);
  field_exchange_.send_consumed_field(state);
  ExpectOk(control_.Receive());
}

Capabilities CouplerParticipant::GetCapabilities() const
{
  return capabilities_;
}

std::size_t CouplerParticipant::InterfaceSize() const noexcept
{
  return interface_size_;
}

double CouplerParticipant::QueryScalar() const
{
  const auto response = Request(protocol::Message::For(Command::get_scalar));
  ExpectOk(response);
  return response.scalars[0];
}

void CouplerParticipant::Shutdown()
{
  if (shutdown_)
    return;
  ExpectOk(Request(protocol::Message::For(Command::shutdown)));
  shutdown_ = true;
}

protocol::Message CouplerParticipant::Request(protocol::Message message) const
{
  control_.Send(message);
  return control_.Receive();
}

void CouplerParticipant::ExpectOk(const protocol::Message& response)
{
  if (response.GetCommand() != Command::ok)
    throw std::runtime_error(
      "CouplerParticipant: participant operation failed");
}

} // namespace pcms::transient
