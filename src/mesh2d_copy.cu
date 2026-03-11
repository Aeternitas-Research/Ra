#include "ra/mesh.cuh"

namespace ra {

Error
Mesh2D::copy(const Mesh2D& other) {
  this->config = other.config;
  this->host = other.host;
  this->device = other.device;

  return cudaSuccess;
}

} // namespace ra
