#include "ra/pmesh.cuh"

namespace ra {

__host__ __device__ Error
PMesh1D::get_device_coordinate(Mesh1D::DeviceStencil& coordinate) {
  return local.get_device_coordinate(coordinate);
}

} // namespace ra
