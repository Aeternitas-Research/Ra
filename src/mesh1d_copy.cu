#include "ra/mesh.cuh"

namespace ra {

__host__ Error
Mesh1D::copy(const Mesh1D& other) {
  config = other.config;
  host = other.host;
  device = other.device;

  return cudaSuccess;
}

} // namespace ra
