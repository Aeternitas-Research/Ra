#include "ra/mesh.cuh"

namespace ra {

Error
Mesh1D::subtract(const OperationSpace space, Mesh1D& mesh_x) {
  auto& host = this->host;
  auto& device = this->device;

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::subtract(const OperationSpace space, const double c, Mesh1D& mesh_x) {
  auto& host = this->host;
  auto& device = this->device;

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, c, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, c, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::subtract(const OperationSpace space, Mesh1D& mesh_c, Mesh1D& mesh_x) {
  auto& host = this->host;
  auto& device = this->device;

  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, mesh_c.host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, mesh_c.device.f, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
