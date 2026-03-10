#include "ra/mesh.cuh"

namespace ra {

Error
Mesh1D::norm(const OperationSpace space, double& r, const std::string type) {
  if ((type == "1") || (type == "l1") || (type == "l^1")) {
    return norm_1(space, r);
  } else if ((type == "2") || (type == "l2") || (type == "l^2")) {
    return norm_2(space, r);
  } else if ((type == "infinity") || (type == "inf") || (type == "max")) {
    return norm_infinity(space, r);
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::norm_1(const OperationSpace space, double& r) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.norm_1(r, host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.norm_1(r, device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::norm_2(const OperationSpace space, double& r) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.norm_2(r, host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.norm_2(r, device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

Error
Mesh1D::norm_infinity(const OperationSpace space, double& r) {
  if (space == OperationSpace::Host) {
    ra_invoke(host.op.norm_infinity(r, host.f));
  } else if (space == OperationSpace::Device) {
    ra_invoke(device.op.norm_infinity(r, device.f));
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
