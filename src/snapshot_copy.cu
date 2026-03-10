#include "ra/snapshot.cuh"

namespace ra {

Error
Snapshot::copy(const Snapshot& other) {
  config = other.config;

  return cudaSuccess;
}

} // namespace ra
