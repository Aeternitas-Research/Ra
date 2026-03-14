#include "ra/mesh.cuh"

namespace ra {

Error
Mesh::subtract(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, c));
  } else {
    return RA_ERROR(ErrorValue::InvalidParameter);
  }

  return cudaSuccess;
}

} // namespace ra
