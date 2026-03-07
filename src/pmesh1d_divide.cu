#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::divide(OperationSpace space, const double c) {
  return local.divide(space, c);
}

__host__ Error
PMesh1D::divide(OperationSpace space, PMesh1D& x) {
  return local.divide(space, x.local);
}

} // namespace ra
