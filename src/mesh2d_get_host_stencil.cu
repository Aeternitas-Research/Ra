#include "ra/index.cuh"
#include "ra/mesh.cuh"
#include <cuda/iterator>
#include <cuda/std/cstddef>

namespace ra {

__host__ __device__ Error
Mesh2D::get_host_stencil(HostStencil& stencil) {
  auto& config = this->config;
  auto& host = this->host;
  auto& geometry = config.geometry;

  std::size_t index[2];
  for (int d = 0; d < 2; ++d) {
    index[d] = geometry.ghost_depth[d][0];
  }
  const auto offset = sub2ind(index, geometry.extent, 2);
  const auto offset_x0 = 1;
  const auto offset_x1 = geometry.extent[0];
  const cuda::std::ptrdiff_t dof = geometry.element.dof;

  auto begin = host.f.begin();
  switch (dof) {
  case 4:
    stencil.f3 = cuda::make_strided_iterator(begin + offset * dof + 3, dof);
    stencil.f3_0l =
      cuda::make_strided_iterator(begin + (offset - offset_x0) * dof + 3, dof);
    stencil.f3_0r =
      cuda::make_strided_iterator(begin + (offset + offset_x0) * dof + 3, dof);
    stencil.f3_1l =
      cuda::make_strided_iterator(begin + (offset - offset_x1) * dof + 3, dof);
    stencil.f3_1r =
      cuda::make_strided_iterator(begin + (offset + offset_x1) * dof + 3, dof);
    [[fallthrough]];
  case 3:
    stencil.f2 = cuda::make_strided_iterator(begin + offset * dof + 2, dof);
    stencil.f2_0l =
      cuda::make_strided_iterator(begin + (offset - offset_x0) * dof + 2, dof);
    stencil.f2_0r =
      cuda::make_strided_iterator(begin + (offset + offset_x0) * dof + 2, dof);
    stencil.f2_1l =
      cuda::make_strided_iterator(begin + (offset - offset_x1) * dof + 2, dof);
    stencil.f2_1r =
      cuda::make_strided_iterator(begin + (offset + offset_x1) * dof + 2, dof);
    [[fallthrough]];
  case 2:
    stencil.f1 = cuda::make_strided_iterator(begin + offset * dof + 1, dof);
    stencil.f1_0l =
      cuda::make_strided_iterator(begin + (offset - offset_x0) * dof + 1, dof);
    stencil.f1_0r =
      cuda::make_strided_iterator(begin + (offset + offset_x0) * dof + 1, dof);
    stencil.f1_1l =
      cuda::make_strided_iterator(begin + (offset - offset_x1) * dof + 1, dof);
    stencil.f1_1r =
      cuda::make_strided_iterator(begin + (offset + offset_x1) * dof + 1, dof);
    [[fallthrough]];
  case 1:
    stencil.f0 = cuda::make_strided_iterator(begin + offset * dof + 0, dof);
    stencil.f0_0l =
      cuda::make_strided_iterator(begin + (offset - offset_x0) * dof + 0, dof);
    stencil.f0_0r =
      cuda::make_strided_iterator(begin + (offset + offset_x0) * dof + 0, dof);
    stencil.f0_1l =
      cuda::make_strided_iterator(begin + (offset - offset_x1) * dof + 0, dof);
    stencil.f0_1r =
      cuda::make_strided_iterator(begin + (offset + offset_x1) * dof + 0, dof);
    break;
  default:
    return RA_HOST_ERROR(ErrorValue::InvalidParameter);
  }

  return RA_HOST_ERROR(ErrorValue::Success);
}

} // namespace ra
