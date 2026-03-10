#include "ra/timestepper.cuh"

namespace ra {

TimeStepperExplicitRK1D::~TimeStepperExplicitRK1D() {}

TimeStepperExplicitRK1D::TimeStepperExplicitRK1D() : TimeStepper() {}

TimeStepperExplicitRK1D::TimeStepperExplicitRK1D(
  const TimeStepperConfig& config)
    : TimeStepper(config) {}

} // namespace ra
