#include "ra/pmesh.cuh"

namespace ra {

__host__ Error
PMesh1D::add(OperationSpace space, const double c) {
  return local.add(space, c);
}

__host__ Error
PMesh1D::add(OperationSpace space, PMesh1D& x) {
  return local.add(space, x.local);
}

__host__ Error
PMesh1D::add(OperationSpace space, const double c, PMesh1D& x) {
  return local.add(space, c, x.local);
}

__host__ Error
PMesh1D::add(OperationSpace space, PMesh1D& c, PMesh1D& x) {
  return local.add(space, c.local, x.local);
}

} // namespace ra
