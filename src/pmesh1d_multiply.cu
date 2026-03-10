#include "ra/pmesh.cuh"

namespace ra {

Error
PMesh1D::multiply(const OperationSpace space, const double c) {
  return local.multiply(space, c);
}

Error
PMesh1D::multiply(const OperationSpace space, PMesh1D& mesh_x) {
  return local.multiply(space, mesh_x.local);
}

} // namespace ra
