#ifndef PCMS_TRANSIENT_APPLICATION_HPP
#define PCMS_TRANSIENT_APPLICATION_HPP

#include "pcms/utility/types.h"

#include <any>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace pcms::transient
{

// Owns interface values that the coupling algorithm must retain between
// fixed-point iterations.
class InterfaceState
{
public:
  InterfaceState();
  explicit InterfaceState(std::size_t size, Real value = 0.0);
  explicit InterfaceState(std::vector<Real> values);

  [[nodiscard]] std::size_t Size() const noexcept;
  [[nodiscard]] Real& operator[](std::size_t index) noexcept;
  [[nodiscard]] Real operator[](std::size_t index) const noexcept;
  [[nodiscard]] std::span<const Real> View() const noexcept;

private:
  std::vector<Real> values_;
};

// Stores the application state required to repeat a coupling window.
struct Checkpoint
{
  Real time = 0.0;
  std::any state;
};

// Describes optional transient features provided by an application.
struct Capabilities
{
  bool can_restart = false;
  bool has_dense_output = false;
  bool reports_qoi = false;
};

// Interface implemented by each solver in a transient coupled simulation.
class Application
{
public:
  // Return the stable name used in the coupling configuration.
  [[nodiscard]] virtual std::string_view Name() const = 0;

  // Advance from the current state to the requested physical time.
  virtual void AdvanceTo(Real target_time) = 0;

  // Save and restore all state required to repeat the current time window.
  [[nodiscard]] virtual Checkpoint Save() const = 0;
  virtual void Restore(const Checkpoint& checkpoint) = 0;

  // Borrow the values produced on a named coupling interface.
  [[nodiscard]] virtual std::span<const Real> GetInterface(
    std::string_view name) const = 0;

  // Apply values to a named coupling interface. Implementations must consume
  // or copy the borrowed values before returning.
  virtual void SetInterface(std::string_view name,
                            std::span<const Real> values) = 0;

  // Return optional quantities used for error and conservation checks.
  [[nodiscard]] virtual std::span<const Real> ReportQoI() const;

  [[nodiscard]] virtual Capabilities GetCapabilities() const = 0;

  virtual ~Application() = default;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_APPLICATION_HPP
