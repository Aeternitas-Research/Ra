#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::subtract(const OperationSpace space, const double c) {
  return local.subtract(space, c);
}

__host__ Error
PMesh1D::subtract(const OperationSpace space, PMesh1D& mesh_x) {
  return local.subtract(space, mesh_x.local);
}

__host__ Error
PMesh1D::subtract(
  const OperationSpace space, const double c, PMesh1D& mesh_x) {
  return local.subtract(space, c, mesh_x.local);
}

__host__ Error
PMesh1D::subtract(
  const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x) {
  return local.subtract(space, mesh_c.local, mesh_x.local);
}

} // namespace ra
