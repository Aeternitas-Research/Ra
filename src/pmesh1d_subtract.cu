#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::subtract(OperationSpace space, const double c) {
  return local.subtract(space, c);
}

__host__ Error
PMesh1D::subtract(OperationSpace space, PMesh1D& x) {
  return local.subtract(space, x.local);
}

__host__ Error
PMesh1D::subtract(OperationSpace space, const double c, PMesh1D& x) {
  return local.subtract(space, c, x.local);
}

__host__ Error
PMesh1D::subtract(OperationSpace space, PMesh1D& c, PMesh1D& x) {
  return local.subtract(space, c.local, x.local);
}

} // namespace ra
