#include "pcms/transient/application.hpp"

#include <utility>

namespace pcms::transient
{

InterfaceState::InterfaceState() = default;

InterfaceState::InterfaceState(std::size_t size, Real value)
  : values_(size, value)
{
}

InterfaceState::InterfaceState(std::vector<Real> values)
  : values_(std::move(values))
{
}

std::size_t InterfaceState::Size() const noexcept
{
  return values_.size();
}

Real& InterfaceState::operator[](std::size_t index) noexcept
{
  return values_[index];
}

Real InterfaceState::operator[](std::size_t index) const noexcept
{
  return values_[index];
}

std::span<const Real> InterfaceState::View() const noexcept
{
  return values_;
}

std::span<const Real> Application::ReportQoI() const
{
  return {};
}

} // namespace pcms::transient
