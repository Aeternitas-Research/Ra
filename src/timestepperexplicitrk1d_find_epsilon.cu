#include "ra/error.cuh"
#include "ra/timestepper.cuh"
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <thrust/copy.h>

namespace ra {

__host__ Error
TimeStepperExplicitRK1D::find_epsilon(double& epsilon) {
  const auto h       = this->config.time.delta;
  const auto& b      = config_extra.b;
  const auto& b_star = config_extra.b_star;

  error.assign(OperationSpace::Device, 0.0);
  for (int stage = 0; stage < config_extra.stage; ++stage) {
    error.add(
      OperationSpace::Device, h * (b[stage] - b_star[stage]), k[stage]);
  }

  const auto tolerance_r = this->config.parameter.tolerance.time.relative;
  const auto tolerance_a = this->config.parameter.tolerance.time.absolute;

  auto op = [=] __host__ __device__(double e, double y) {
    return e / (tolerance_r * cuda::std::abs(y) + tolerance_a);
  };

  cuda::zip_transform_iterator kernel{
    op, error.local.device.f.begin(), backup.local.device.f.begin()};
  const auto n = error.local.device.f.size();
  thrust::copy(kernel, kernel + n, error.local.device.f.begin());

  ra_invoke(error.norm(OperationSpace::Device, epsilon, "l^2"));
  epsilon /= cuda::std::sqrt(static_cast<double>(n));

  return cudaSuccess;
}

} // namespace ra
