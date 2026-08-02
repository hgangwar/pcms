#pragma once

#include "pcms/transient/participant.hpp"
#include "pcms/transient/participant_adapter/participant_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace pcms::transient
{

// Coupler-side Participant implementation. Operations are forwarded through
// the control protocol to the actual participant process.
class CouplerParticipant final : public Participant
{
public:
  struct Configuration
  {
    std::string participant;
    std::string produced_interface;
    std::string consumed_interface;
    // Applications sharing a coupler must all be created before communication
    // begins. Set this when Connect() must be delayed until that setup is done.
    bool defer_connect = false;
  };

  struct FieldExchange
  {
    std::function<void()> receive_produced_field;
    std::function<void(const InterfaceState&)> send_consumed_field;
  };

  CouplerParticipant(Application& application, MPI_Comm mpi_comm,
                     Configuration configuration);
  void ConfigureFieldExchange(FieldExchange exchange);
  void Connect();

  [[nodiscard]] std::string_view Name() const override;
  void AdvanceTo(Real target_time) override;
  [[nodiscard]] Checkpoint Save() const override;
  void Restore(const Checkpoint& checkpoint) override;
  [[nodiscard]] InterfaceState GetInterface(
    std::string_view name) const override;
  void SetInterface(std::string_view name,
                    const InterfaceState& state) override;
  [[nodiscard]] Capabilities GetCapabilities() const override;

  [[nodiscard]] std::size_t InterfaceSize() const noexcept;
  // Optional application-defined scalar diagnostic. Its meaning is agreed by
  // the participant-adapter configuration rather than prescribed by the
  // protocol.
  [[nodiscard]] double QueryScalar() const;
  void Shutdown();

private:
  struct CheckpointToken
  {
    std::int64_t value;
  };

  [[nodiscard]] protocol::Message Request(protocol::Message message) const;
  static void ExpectOk(const protocol::Message& response);

  std::string participant_name_;
  std::string produced_interface_;
  std::string consumed_interface_;
  mutable protocol::ControlChannel control_;
  std::size_t interface_size_ = 0;
  Capabilities capabilities_;
  Real current_time_ = 0.0;
  mutable std::int64_t next_checkpoint_token_ = 0;
  bool shutdown_ = false;
  bool connected_ = false;
  FieldExchange field_exchange_;
};

} // namespace pcms::transient
