#include "ra/pmesh.cuh"

namespace ra {

Error
PMesh1D::add(const OperationSpace space, const double c) {
  return local.add(space, c);
}

Error
PMesh1D::add(const OperationSpace space, PMesh1D& mesh_x) {
  return local.add(space, mesh_x.local);
}

Error
PMesh1D::add(const OperationSpace space, const double c, PMesh1D& mesh_x) {
  return local.add(space, c, mesh_x.local);
}

Error
PMesh1D::add(const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x) {
  return local.add(space, mesh_c.local, mesh_x.local);
}

} // namespace ra
