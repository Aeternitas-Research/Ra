#include "ra/mesh.cuh"
#include <cuda/iterator>
#include <cuda/std/cstddef>

namespace ra {

__host__ __device__ Error
Mesh1D::get_device_coordinate(DeviceStencil& coordinate) {
  auto& config = this->config;
  auto& device = this->device;

  const auto offset = config.geometry.ghost_depth[0][0];
  const cuda::std::ptrdiff_t dof = 2 * 1;

  auto begin = device.x.begin();
  coordinate.x0 = cuda::make_strided_iterator(begin + offset * dof + 0, dof);
  coordinate.dx0 = cuda::make_strided_iterator(begin + offset * dof + 1, dof);

  return RA_SUCCESS;
}

} // namespace ra
