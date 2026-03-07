#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::multiply(OperationSpace space, const double c) {
  return local.multiply(space, c);
}

__host__ Error
PMesh1D::multiply(OperationSpace space, PMesh1D& x) {
  return local.multiply(space, x.local);
}

} // namespace ra
