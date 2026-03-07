#include "ra/mesh.cuh"
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/transform.h>

namespace ra {

__host__ Error
Mesh1D::multiply(OperationSpace space, const double c) {
  auto op = [=] __host__ __device__(double v1) { return v1 * c; };

  if (space == OperationSpace::Host) {
    thrust::transform(host.f.begin(), host.f.end(), host.f.begin(), op);
  } else if (space == OperationSpace::Device) {
    thrust::transform(device.f.begin(), device.f.end(), device.f.begin(), op);
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::multiply(OperationSpace space, Mesh1D& x) {
  auto op = [] __host__ __device__(double v1, double v2) { return v1 * v2; };

  if (space == OperationSpace::Host) {
    cuda::zip_transform_iterator kernel{op, host.f.begin(), x.host.f.begin()};
    const auto n = host.f.size();
    thrust::copy(kernel, kernel + n, host.f.begin());
  } else if (space == OperationSpace::Device) {
    cuda::zip_transform_iterator kernel{
      op, device.f.begin(), x.device.f.begin()};
    const auto n = device.f.size();
    thrust::copy(kernel, kernel + n, device.f.begin());
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
