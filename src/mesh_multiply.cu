#include "ra/mesh.cuh"

namespace ra {

Error
Mesh::multiply(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.multiply(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.multiply(device.f, c));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
