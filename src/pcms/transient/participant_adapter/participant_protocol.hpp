#pragma once

#include "pcms/coupler/coupler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace pcms::transient::protocol
{

enum class Command
{
  hello = 1,
  save,
  restore,
  advance,
  get_field,
  set_boundary,
  get_scalar,
  shutdown,
  ok
};

inline constexpr std::size_t IntegerCount = 5;
inline constexpr std::size_t ScalarCount = 1;

// Fixed control envelope carried by two typed PCMS global-data interfaces.
// integers[0] is Command; remaining integers carry tokens, sizes, and flags.
// scalars[0] carries time or an application-defined scalar diagnostic.
struct Message
{
  std::array<std::int64_t, IntegerCount> integers{};
  std::array<double, ScalarCount> scalars{};

  [[nodiscard]] static Message For(Command command);
  [[nodiscard]] Command GetCommand() const;
};

// Shared transport for the two ends of the transient participant protocol.
// Commands and integral metadata use one typed GDI; times and diagnostics use
// another so integer values are never encoded as floating point values.
class ControlChannel
{
public:
  ControlChannel(Application& application, MPI_Comm mpi_comm);

  void Send(Message message) const;
  [[nodiscard]] Message Receive() const;

private:
  Application* application_;
  DataHandle<std::int64_t> integer_data_;
  DataHandle<double> scalar_data_;
};

} // namespace pcms::transient::protocol
