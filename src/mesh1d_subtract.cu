#include "ra/mesh.cuh"

namespace ra {

__host__ Error
Mesh1D::subtract(const OperationSpace space, const double c) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, c));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, c));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::subtract(const OperationSpace space, Mesh1D& mesh_x) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::subtract(const OperationSpace space, const double c, Mesh1D& mesh_x) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.subtract(host.f, c, mesh_x.host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.subtract(device.f, c, mesh_x.device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

__host__ Error
Mesh1D::subtract(const OperationSpace space, Mesh1D& mesh_c, Mesh1D& mesh_x) {
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
