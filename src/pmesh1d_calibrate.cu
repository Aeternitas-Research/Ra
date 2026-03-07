#include "ra/error.cuh"
#include "ra/index.cuh"
#include "ra/mpi.cuh"
#include "ra/pmesh.cuh"
#include <mpi.h>
#include <string>

namespace ra {

__host__ Error
PMesh1D::calibrate() {
  auto self   = config.topology.self;
  auto extent = config.topology.extent;
  ind2sub(self, config.topology.rank.self, extent, 1);

  auto neighbor = config.topology.neighbor;
  for (int d = 0; d < 1; ++d) {
    // upwind
    auto target = neighbor[2 * d + 0];
    for (int index = 0; index < DIMENSION_MAX; ++index) {
      target[index] = self[index];
    }
    target[d] = (target[d] - 1 + extent[d]) % extent[d];

    // downwind
    target = neighbor[2 * d + 1];
    for (int index = 0; index < DIMENSION_MAX; ++index) {
      target[index] = self[index];
    }
    target[d] = (target[d] + 1) % extent[d];
  }
  for (int d = 0; d < 1; ++d) {
    for (int direction = 0; direction < 2; ++direction) {
      config.topology.rank.neighbor[2 * d + direction] =
        sub2ind(neighbor[2 * d + direction], extent, 1);
    }
  }

  config.local.name           = config.global.name + std::string{".local"};
  config.local.info.mpi_rank  = config.topology.rank.self;
  config.local.info.step      = config.global.info.step;
  config.local.info.time      = config.global.info.time;
  config.local.file.handle    = config.global.file.handle;
  config.local.file.directory = config.global.file.directory;
  config.local.geometry.dof   = config.global.geometry.dof;
  for (int d = 0; d < 1; ++d) {
    config.local.geometry.extent[d] =
      config.global.geometry.extent[d] / config.topology.extent[d] +
      config.global.geometry.ghost_depth[d][0] +
      config.global.geometry.ghost_depth[d][1];

    for (int direction = 0; direction < 2; ++direction) {
      config.local.geometry.ghost_depth[d][direction] =
        config.global.geometry.ghost_depth[d][direction];
    }
  }

  Mesh1D mesh_local(config.local);
  ra_invoke(local.copy(mesh_local));

  mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));

  return cudaSuccess;
}

} // namespace ra
