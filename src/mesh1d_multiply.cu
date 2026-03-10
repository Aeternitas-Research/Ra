#include "ra/mesh.cuh"

namespace ra {

Error
Mesh1D::multiply(const OperationSpace space, const double c) {

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.multiply(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.multiply(device.f, c));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::multiply(const OperationSpace space, Mesh1D& mesh_x) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.multiply(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.multiply(device.f, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
