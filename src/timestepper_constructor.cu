#include "ra/timestepper.cuh"

namespace ra {

__host__ TimeStepper::~TimeStepper() {}

__host__
TimeStepper::TimeStepper() {}

__host__
TimeStepper::TimeStepper(const TimeStepperConfig& config)
    : config(config) {}

} // namespace ra
