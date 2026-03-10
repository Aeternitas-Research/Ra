#include "ra/pmesh.cuh"

namespace ra {

Error
PMesh1D::assign(const OperationSpace space, const double c) {
  return local.assign(space, c);
}

Error
PMesh1D::assign(const OperationSpace space, PMesh1D& mesh_x) {
  return local.assign(space, mesh_x.local);
}

} // namespace ra
