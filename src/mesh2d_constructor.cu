#include "ra/mesh.cuh"

namespace ra {

Mesh2D::~Mesh2D() {}

Mesh2D::Mesh2D() : Mesh() {}

Mesh2D::Mesh2D(const MeshConfig& in_config) : Mesh(in_config) {
  auto& config = this->config;

  const auto extent = config.geometry.extent;
  const auto dof = config.geometry.element.dof;
  if (config.geometry.element.type == MeshElementType::Rectangle) {
    const auto n = extent[0] * extent[1];
    host.x.resize(2 * 2 * n, thrust::no_init);
    host.f.resize(dof * n, thrust::no_init);
    device.x.resize(2 * 2 * n, thrust::no_init);
    device.f.resize(dof * n, thrust::no_init);
  } else {
    return RA_ERROR(ErrorValue::InvalidGeometry);
  }
}

} // namespace ra
