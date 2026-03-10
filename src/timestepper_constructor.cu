#include "ra/timestepper.cuh"

namespace ra {

TimeStepper::~TimeStepper() {}

TimeStepper::TimeStepper() {}

TimeStepper::TimeStepper(const TimeStepperConfig& config) : config(config) {}

} // namespace ra
