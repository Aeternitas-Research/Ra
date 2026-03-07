#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::divide(const OperationSpace space, const double c) {
  return local.divide(space, c);
}

__host__ Error
PMesh1D::divide(const OperationSpace space, PMesh1D& mesh_x) {
  return local.divide(space, mesh_x.local);
}

} // namespace ra
