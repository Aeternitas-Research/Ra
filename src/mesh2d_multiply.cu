#include "ra/mesh.cuh"

namespace ra {

Error
Mesh2D::multiply(const OperationSpace space, Mesh2D& mesh_x) {
  auto& host = this->host;
  auto& device = this->device;

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.multiply(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.multiply(device.f, mesh_x.device.f));
  } else {
    return RA_ERROR(ErrorValue::InvalidParameter);
  }

  return RA_ERROR(ErrorValue::Success);
}

} // namespace ra
