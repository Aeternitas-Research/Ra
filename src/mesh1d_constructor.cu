#include "ra/mesh.cuh"

namespace ra {

Mesh1D::~Mesh1D() {}

Mesh1D::Mesh1D() : Mesh() {}

Mesh1D::Mesh1D(const MeshConfig& in_config) : Mesh(in_config) {
  auto& config = this->config;

  config.geometry.element.type = MeshElementType::Line;

  const auto extent = config.geometry.extent;
  const auto dof = config.geometry.element.dof;
  const auto n = extent[0];
  host.x.resize(2 * 1 * n, thrust::no_init);
  host.f.resize(dof * n, thrust::no_init);
  device.x.resize(2 * 1 * n, thrust::no_init);
  device.f.resize(dof * n, thrust::no_init);
}

} // namespace ra
