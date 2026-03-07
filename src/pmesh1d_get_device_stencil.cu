#include "ra/pmesh.cuh"

namespace ra {

__host__ __device__ Error
PMesh1D::get_device_stencil(Mesh1D::DeviceStencil& stencil) {
  return local.get_device_stencil(stencil);
}

} // namespace ra
