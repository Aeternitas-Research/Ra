#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>

namespace ra {

PMesh::~PMesh() {}

PMesh::PMesh() {}

PMesh::PMesh(
  const int mpi_rank, const int* mpi_extent, const MeshConfig& config_global) {
  config.global = config_global;
  config.topology.rank.self = mpi_rank;
  for (int d = 0; d < DIMENSION_MAX; ++d) {
    config.topology.extent[d] = mpi_extent[d];
  }

  ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));
}

} // namespace ra
