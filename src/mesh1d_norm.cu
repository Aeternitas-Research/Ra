#include "ra/mesh.cuh"
#include <cuda/std/cmath>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace ra {

__host__ Error
Mesh1D::norm(OperationSpace space, double& r, const std::string type) {
  if ((type == "1") || (type == "l1") || (type == "l^1")) {
    return norm_1(space, r);
  } else if ((type == "2") || (type == "l2") || (type == "l^2")) {
    return norm_2(space, r);
  } else if ((type == "infinity") || (type == "inf") || (type == "max")) {
    return norm_infinity(space, r);
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::norm_1(OperationSpace space, double& r) {
  auto op = [] __host__ __device__(double v1) { return cuda::std::abs(v1); };

  if (space == OperationSpace::Host) {
    r = thrust::transform_reduce(
      host.f.begin(), host.f.end(), op, 0.0, cuda::std::plus<double>());
  } else if (space == OperationSpace::Device) {
    r = thrust::transform_reduce(
      device.f.begin(), device.f.end(), op, 0.0, cuda::std::plus<double>());
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::norm_2(OperationSpace space, double& r) {
  auto op = [] __host__ __device__(double v1) { return v1 * v1; };

  if (space == OperationSpace::Host) {
    r = thrust::transform_reduce(
      host.f.begin(), host.f.end(), op, 0.0, cuda::std::plus<double>());
  } else if (space == OperationSpace::Device) {
    r = thrust::transform_reduce(
      device.f.begin(), device.f.end(), op, 0.0, cuda::std::plus<double>());
  } else {
    return cudaErrorInvalidValue;
  }

  r = cuda::std::sqrt(r);

  return cudaSuccess;
}

__host__ Error
Mesh1D::norm_infinity(OperationSpace space, double& r) {
  auto op = [] __host__ __device__(double v1) { return cuda::std::abs(v1); };

  if (space == OperationSpace::Host) {
    r = thrust::transform_reduce(
      host.f.begin(), host.f.end(), op, 0.0, cuda::maximum<double>());
  } else if (space == OperationSpace::Device) {
    r = thrust::transform_reduce(
      device.f.begin(), device.f.end(), op, 0.0, cuda::maximum<double>());
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
