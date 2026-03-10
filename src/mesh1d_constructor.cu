#include "ra/mesh.cuh"

namespace ra {

__host__ Mesh1D::~Mesh1D() {
  if (config.buffer.in) {
    delete[] config.buffer.in;
  }
  if (config.buffer.out) {
    delete[] config.buffer.out;
  }
}

__host__
Mesh1D::Mesh1D() {}

__host__
Mesh1D::Mesh1D(const MeshConfig& config)
    : config(config) {
  this->config.geometry.type = MeshElementType::Line;

  host.x.resize(2 * 1 * config.geometry.extent[0], thrust::no_init);
  host.f.resize(
    config.geometry.dof * config.geometry.extent[0], thrust::no_init);

  device.x.resize(2 * 1 * config.geometry.extent[0], thrust::no_init);
  device.f.resize(
    config.geometry.dof * config.geometry.extent[0], thrust::no_init);
}

} // namespace ra
