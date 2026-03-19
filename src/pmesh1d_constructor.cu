#include "ra/pmesh.cuh"

namespace ra {

PMesh1D::~PMesh1D() {}

PMesh1D::PMesh1D() : PMesh() {}

PMesh1D::PMesh1D(
  const int mpi_rank, const int* mpi_extent, const MeshConfig& config_global)
    : PMesh(mpi_rank, mpi_extent, config_global) {}

} // namespace ra
