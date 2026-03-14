#include "ra/mesh.cuh"
#include <cuda/iterator>
#include <cuda/std/cstddef>

namespace ra {

__host__ __device__ Error
Mesh1D::get_device_stencil(DeviceStencil& stencil) {
  auto& config = this->config;
  auto& device = this->device;
  auto& geometry = config.geometry;

  const auto offset = geometry.ghost_depth[0][0];
  const cuda::std::ptrdiff_t dof = geometry.element.dof;

  auto begin = device.f.begin();
  switch (dof) {
  case 4:
    stencil.f3 = cuda::make_strided_iterator(begin + offset * dof + 3, dof);
    stencil.f3_l =
      cuda::make_strided_iterator(begin + (offset - 1) * dof + 3, dof);
    stencil.f3_r =
      cuda::make_strided_iterator(begin + (offset + 1) * dof + 3, dof);
    [[fallthrough]];
  case 3:
    stencil.f2 = cuda::make_strided_iterator(begin + offset * dof + 2, dof);
    stencil.f2_l =
      cuda::make_strided_iterator(begin + (offset - 1) * dof + 2, dof);
    stencil.f2_r =
      cuda::make_strided_iterator(begin + (offset + 1) * dof + 2, dof);
    [[fallthrough]];
  case 2:
    stencil.f1 = cuda::make_strided_iterator(begin + offset * dof + 1, dof);
    stencil.f1_l =
      cuda::make_strided_iterator(begin + (offset - 1) * dof + 1, dof);
    stencil.f1_r =
      cuda::make_strided_iterator(begin + (offset + 1) * dof + 1, dof);
    [[fallthrough]];
  case 1:
    stencil.f0 = cuda::make_strided_iterator(begin + offset * dof + 0, dof);
    stencil.f0_l =
      cuda::make_strided_iterator(begin + (offset - 1) * dof + 0, dof);
    stencil.f0_r =
      cuda::make_strided_iterator(begin + (offset + 1) * dof + 0, dof);
    break;
  default:
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
