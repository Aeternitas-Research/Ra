#include "ra/mesh.cuh"

namespace ra {

__host__ Error
Mesh1D::divide(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.divide(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.divide(device.f, c));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::divide(const OperationSpace space, Mesh1D& mesh_x) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.divide(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.divide(device.f, mesh_x, device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
