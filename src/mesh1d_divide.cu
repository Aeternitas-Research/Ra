#include "ra/mesh.cuh"

namespace ra {

Error
Mesh1D::divide(const OperationSpace space, Mesh1D& mesh_x) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.divide(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.divide(device.f, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
