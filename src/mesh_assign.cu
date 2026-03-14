#include "ra/mesh.cuh"

namespace ra {

Error
Mesh::assign(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.assign(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.assign(device.f, c));
  } else {
    return RA_ERROR(ErrorValue::InvalidParameter);
  }

  return RA_SUCCESS;
}

} // namespace ra
