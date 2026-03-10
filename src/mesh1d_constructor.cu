#include "ra/mesh.cuh"

namespace ra {

Mesh1D::~Mesh1D() {
  if (config.buffer.in) {
    delete[] config.buffer.in;
  }
  if (config.buffer.out) {
    delete[] config.buffer.out;
  }
}

Mesh1D::Mesh1D() {}

Mesh1D::Mesh1D(const MeshConfig& config) : config(config) {
  this->config.geometry.element.type = MeshElementType::Line;

  const auto extent = this->config.geometry.extent;
  const auto dof = this->config.geometry.element.dof;
  host.x.resize(2 * 1 * extent[0], thrust::no_init);
  host.f.resize(dof * extent[0], thrust::no_init);
  device.x.resize(2 * 1 * extent[0], thrust::no_init);
  device.f.resize(dof * extent[0], thrust::no_init);
}

} // namespace ra
