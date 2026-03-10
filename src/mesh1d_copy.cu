#include "ra/mesh.cuh"

namespace ra {

Error
Mesh1D::copy(const Mesh1D& other) {
  this->config = other.config;
  this->host = other.host;
  this->device = other.device;

  return cudaSuccess;
}

} // namespace ra
