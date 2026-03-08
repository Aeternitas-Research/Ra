#include "ra/pmesh.cuh"

namespace ra {

__host__ __device__ Error
PMesh1D::get_host_coordinate(Mesh1D::HostStencil& coordinate) {
  return local.get_host_coordinate(coordinate);
}

} // namespace ra
