#include "pcms/coupler/coupler.hpp"

namespace pcms
{

FieldLayoutCommunicator& Application::GetLayoutCommunicator(
  const FieldLayout& layout)
{
  PCMS_FUNCTION_TIMER;
  auto it = field_layout_communicators_.find(&layout);
  if (it != field_layout_communicators_.end()) {
    return *it->second;
  } else {
    throw pcms_error("Field added with unregistered layout. Call AddLayout() "
                     "before AddField().");
  }
}

const FieldLayout& Application::AddLayout(
  std::string name, std::shared_ptr<const FieldLayout> layout,
  bool participates)
{
  return AddLayout(std::move(name), std::move(layout),
                   std::make_unique<GenericFieldExchangePlanner>(),
                   participates);
}

const FieldLayout& Application::AddLayout(
  std::string name, std::shared_ptr<const FieldLayout> layout,
  std::unique_ptr<FieldExchangePlanner> planner, bool participates)
{
  MPI_Comm mpi_comm_subset = MPI_COMM_NULL;
  bool own_mpi_comm = false;
  PCMS_ALWAYS_ASSERT((mpi_comm_ == MPI_COMM_NULL) ? (!participates) : true);
  if (mpi_comm_ != MPI_COMM_NULL) {
    int rank = -1;
    MPI_Comm_rank(mpi_comm_, &rank);
    MPI_Comm_split(mpi_comm_, participates ? 0 : MPI_UNDEFINED, rank,
                   &mpi_comm_subset);
    own_mpi_comm = true;
  }
  layouts_.push_back(std::move(layout));
  const FieldLayout& layout_ref = *layouts_.back();

  // Check if there's an overlap mask for this layout
  const OverlapMask* overlap_mask = nullptr;
  auto mask_it = layout_overlap_masks_.find(name);
  if (mask_it != layout_overlap_masks_.end()) {
    overlap_mask = mask_it->second.get();
  }

  field_layout_communicators_.emplace(
    &layout_ref, std::make_unique<FieldLayoutCommunicator>(
                   name, mpi_comm_subset, redev_, channel_, layout_ref,
                   std::move(planner), own_mpi_comm, overlap_mask));
  return layout_ref;
}

void Application::SetLayoutOverlapMask(
  const std::string& layout_name, std::unique_ptr<OverlapMask> overlap_mask)
{
  layout_overlap_masks_[layout_name] = std::move(overlap_mask);
}

} // namespace pcms
