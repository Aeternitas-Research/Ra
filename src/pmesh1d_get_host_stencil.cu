#include "ra/pmesh.cuh"

namespace ra {

__host__ __device__ Error
PMesh1D::get_host_stencil(Mesh1D::HostStencil& stencil) {
  return local.get_host_stencil(stencil);
}

} // namespace ra
