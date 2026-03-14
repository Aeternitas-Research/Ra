#include "ra/snapshot.cuh"

namespace ra {

Error
Snapshot::copy(const Snapshot& other) {
  config = other.config;

  return RA_SUCCESS;
}

} // namespace ra
