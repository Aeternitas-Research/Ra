#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::assign(const OperationSpace space, const double c) {
  return local.assign(space, c);
}

__host__ Error
PMesh1D::assign(const OperationSpace space, PMesh1D& x) {
  return local.assign(space, x.local);
}

} // namespace ra
