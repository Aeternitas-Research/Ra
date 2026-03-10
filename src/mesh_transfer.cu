#include "ra/mesh.cuh"
#include <cuda_runtime.h>

namespace ra {

Error
Mesh::transfer(const cudaMemcpyKind kind, const bool x, const bool f) {
  if (kind == cudaMemcpyHostToDevice) {
    if (x) {
      device.x = host.x;
    }
    if (f) {
      device.f = host.f;
    }
  } else if (kind == cudaMemcpyDeviceToHost) {
    if (x) {
      host.x = device.x;
    }
    if (f) {
      host.f = device.f;
    }
  } else {
    return cudaErrorInvalidMemcpyDirection;
  }

  return cudaSuccess;
}

} // namespace ra
