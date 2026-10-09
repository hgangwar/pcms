#ifndef PCMS_TRANSIENT_TIMESTEPPER_HPP
#define PCMS_TRANSIENT_TIMESTEPPER_HPP

#include "pcms/utility/types.h"

namespace pcms::transient
{

// Result of Timestepper::Update.
struct StepDecision
{
  // Whether the completed time window is accepted.
  bool accepted;
  // Positive step size to try next.
  Real next_step;
};

// Decides whether to accept a completed time window and selects the next step.
class Timestepper
{
public:
  // Return the positive step size used for the first coupling window.
  [[nodiscard]] virtual Real InitialStep() const = 0;

  // Given the current time step that was actually completed and its normalized
  // error, return whether to accept it and the positive step size to try next.
  [[nodiscard]] virtual StepDecision Update(Real dt, Real err) = 0;

  virtual ~Timestepper() = default;
};

// Always accepts and keeps a constant time step.
class FixedTimestepper : public Timestepper
{
public:
  // Store the positive step size used for every window.
  explicit FixedTimestepper(Real dt);

  // Return the configured fixed step.
  Real InitialStep() const override;

  // Keep the configured fixed time step.
  StepDecision Update(Real dt, Real err) override;

private:
  Real dt_;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_TIMESTEPPER_HPP
