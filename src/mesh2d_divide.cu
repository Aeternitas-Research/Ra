#include "ra/mesh.cuh"

namespace ra {

Error
Mesh2D::divide(const OperationSpace space, Mesh2D& mesh_x) {
  auto& host = this->host;
  auto& device = this->device;

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.divide(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.divide(device.f, mesh_x.device.f));
  } else {
    return RA_ERROR_HOST(ErrorValue::InvalidParameter);
  }

  return RA_ERROR_HOST(ErrorValue::Success);
}

} // namespace ra
