#include "ra/snapshot.cuh"

namespace ra {

__host__ Error
Snapshot::copy(const Snapshot& other) {
  config = other.config;

  return cudaSuccess;
}

} // namespace ra
