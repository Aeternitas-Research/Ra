#include "ra/mesh.cuh"

namespace ra {

Mesh1D::~Mesh1D() {}

Mesh1D::Mesh1D() : Mesh() {}

Mesh1D::Mesh1D(const MeshConfig& in_config) : Mesh(in_config) {
  auto& config = this->config;

  config.geometry.element.type = MeshElementType::Line;

  const auto extent = config.geometry.extent;
  const auto dof = config.geometry.element.dof;
  host.x.resize(2 * 1 * extent[0], thrust::no_init);
  host.f.resize(dof * extent[0], thrust::no_init);
  device.x.resize(2 * 1 * extent[0], thrust::no_init);
  device.f.resize(dof * extent[0], thrust::no_init);
}

} // namespace ra
