#include "ra/timestepper.cuh"

namespace ra {

__host__ TimeStepperExplicitRK1D::~TimeStepperExplicitRK1D() {}

__host__
TimeStepperExplicitRK1D::TimeStepperExplicitRK1D()
    : TimeStepper() {}

__host__
TimeStepperExplicitRK1D::TimeStepperExplicitRK1D(
  const TimeStepperConfig& config)
    : TimeStepper(config) {}

} // namespace ra
