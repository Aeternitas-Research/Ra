#include "ra/mesh.cuh"
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace ra {

__host__ Error
Mesh1D::assign(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    thrust::fill(host.f.begin(), host.f.end(), c);
  } else if (space == OperationSpace::Device) {
    thrust::fill(device.f.begin(), device.f.end(), c);
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::assign(const OperationSpace space, Mesh1D& x) {
  if (space == OperationSpace::Host) {
    thrust::copy(x.host.f.begin(), x.host.f.end(), host.f.begin());
  } else if (space == OperationSpace::Device) {
    thrust::copy(x.device.f.begin(), x.device.f.end(), device.f.begin());
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
