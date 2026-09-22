#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pcms/transient/application.hpp"

#include <any>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tr = pcms::transient;
using pcms::Real;

namespace
{

class TestApplication final : public tr::Application
{
public:
  std::string_view Name() const override { return "test-application"; }

  void AdvanceTo(Real target_time) override { time_ = target_time; }

  tr::Checkpoint Save() const override
  {
    const auto values = interface_.View();
    return {time_, std::vector<Real>(values.begin(), values.end())};
  }

  void Restore(const tr::Checkpoint& checkpoint) override
  {
    time_ = checkpoint.time;
    interface_ =
      tr::InterfaceState(std::any_cast<std::vector<Real>>(checkpoint.state));
  }

  std::span<const Real> GetInterface(std::string_view) const override
  {
    return interface_.View();
  }

  void SetInterface(std::string_view, std::span<const Real> values) override
  {
    interface_ =
      tr::InterfaceState(std::vector<Real>(values.begin(), values.end()));
  }

  tr::Capabilities GetCapabilities() const override
  {
    return {/*can_restart=*/true, /*has_dense_output=*/false,
            /*reports_qoi=*/false};
  }

  Real Time() const noexcept { return time_; }

private:
  Real time_ = 0.0;
  tr::InterfaceState interface_{2, 0.0};
};

} // namespace

TEST_CASE("Application state can be saved and restored", "[transient]")
{
  TestApplication application;

  // These values represent user-owned interface DOFs in their coupling order.
  const std::vector<Real> initial_values{1.0, 2.0};
  application.SetInterface("interface", initial_values);
  application.AdvanceTo(0.5);
  const tr::Checkpoint checkpoint = application.Save();

  // Change both parts of the state so restore cannot pass accidentally.
  const std::vector<Real> replacement_values{3.0, 4.0};
  application.SetInterface("interface", replacement_values);
  application.AdvanceTo(1.0);
  application.Restore(checkpoint);

  const auto restored = application.GetInterface("interface");

  REQUIRE(std::string(application.Name()) == "test-application");
  REQUIRE(application.Time() == Catch::Approx(0.5));
  REQUIRE(restored.size() == 2);
  REQUIRE(restored[0] == Catch::Approx(1.0));
  REQUIRE(restored[1] == Catch::Approx(2.0));
  REQUIRE(application.GetCapabilities().can_restart);
  REQUIRE(application.ReportQoI().empty());
}
