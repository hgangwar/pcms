#include "pcms/transient/participant_adapter/participant_protocol.hpp"

namespace pcms::transient::protocol
{

Message Message::For(Command command)
{
  Message message;
  message.integers[0] = static_cast<std::int64_t>(command);
  return message;
}

Command Message::GetCommand() const
{
  return static_cast<Command>(integers[0]);
}

ControlChannel::ControlChannel(Application& application, MPI_Comm mpi_comm)
  : application_(&application),
    integer_data_(application.AddData<std::int64_t>(
      "transient_protocol_integer", mpi_comm)),
    scalar_data_(
      application.AddData<double>("transient_protocol_scalar", mpi_comm))
{
}

void ControlChannel::Send(Message message) const
{
  auto integers = Rank1View<std::int64_t, HostMemorySpace>(
    message.integers.data(), message.integers.size());
  auto scalars = Rank1View<double, HostMemorySpace>(message.scalars.data(),
                                                    message.scalars.size());
  application_->SendPhase([&] {
    integer_data_.Send(integers, "transient_protocol_header");
    scalar_data_.Send(scalars, "transient_protocol_value");
  });
}

Message ControlChannel::Receive() const
{
  Message message;
  auto integers = Rank1View<std::int64_t, HostMemorySpace>(
    message.integers.data(), message.integers.size());
  auto scalars = Rank1View<double, HostMemorySpace>(message.scalars.data(),
                                                    message.scalars.size());
  application_->ReceivePhase([&] {
    integer_data_.Receive(integers, "transient_protocol_header");
    scalar_data_.Receive(scalars, "transient_protocol_value");
  });
  return message;
}

} // namespace pcms::transient::protocol
