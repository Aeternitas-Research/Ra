#include "ra/index.cuh"
#include "ra/mesh.cuh"
#include <cuda/iterator>
#include <cuda/std/cstddef>

namespace ra {

__host__ __device__ Error
Mesh2D::get_device_coordinate(DeviceStencil& coordinate) {
  auto& config = this->config;
  auto& device = this->device;
  auto& geometry = config.geometry;

  std::size_t index[2];
  for (int d = 0; d < 2; ++d) {
    index[d] = geometry.ghost_depth[d][0];
  }
  const auto offset = sub2ind(index, geometry.extent, 2);
  const cuda::std::ptrdiff_t dof = 2 * 2;

  auto begin = device.x.begin();
  coordinate.x0 = cuda::make_strided_iterator(begin + offset * dof + 0, dof);
  coordinate.x1 = cuda::make_strided_iterator(begin + offset * dof + 1, dof);
  coordinate.dx0 = cuda::make_strided_iterator(begin + offset * dof + 2, dof);
  coordinate.dx1 = cuda::make_strided_iterator(begin + offset * dof + 3, dof);

  return RA_ERROR(ErrorValue::Success);
}

} // namespace ra
