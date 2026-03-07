#include "ra/index.cuh"
#include "ra/mesh.cuh"
#include "ra/mpi.cuh"
#include <algorithm>
#include <mpi.h>

namespace ra {

__host__ Error
Mesh1D::sync(const int other, const int dimension, const Direction direction) {
#ifdef RA_DEBUG
  if (dimension > 0) {
    cuda::std::terminate();
  }
#endif

  auto& buffer           = config.buffer;
  auto& geometry         = config.geometry;
  const auto dof         = geometry.dof;
  const auto i_direction = static_cast<int>(direction);

  for (int d = 0; d < 1; ++d) {
    buffer.extent[d] = geometry.extent[d];
  }
  for (int d = 0; d < 1; ++d) {
    if (d == dimension) {
      buffer.extent[d] =
        std::max(geometry.ghost_depth[d][0], geometry.ghost_depth[d][1]);
    } else {
      buffer.extent[d] = geometry.extent[d];
    }
  }

  const auto length = dof * buffer.extent[0] * sizeof(double);
  if (buffer.length <= length) {
    buffer.length = length;
    if (buffer.in) {
      delete[] buffer.in;
    }
    if (buffer.out) {
      delete[] buffer.out;
    }

    buffer.in  = new char[length * sizeof(double)];
    buffer.out = new char[length * sizeof(double)];
  }

  int position;
  std::size_t index[1] = {0};

  // pack
  position = 0;
  if (direction == Direction::Upwind) {
    for (std::size_t j0 = 0; j0 < geometry.ghost_depth[0][0]; ++j0) {
      index[0] = geometry.extent[0] -
                 (geometry.ghost_depth[0][0] + geometry.ghost_depth[0][1]) +
                 j0;
      const auto offset = sub2ind(index, geometry.extent, 1);
      mpi_invoke(MPI_Pack(
        host.f.data() + dof * offset, dof, MPI_DOUBLE, buffer.out,
        buffer.length, &position, MPI_COMM_WORLD));
    }
  } else if (direction == Direction::Downwind) {
    for (std::size_t j0 = 0; j0 < geometry.ghost_depth[0][1]; ++j0) {
      index[0]          = geometry.ghost_depth[0][0] + j0;
      const auto offset = sub2ind(index, geometry.extent, 1);
      mpi_invoke(MPI_Pack(
        host.f.data() + dof * offset, dof, MPI_DOUBLE, buffer.out,
        buffer.length, &position, MPI_COMM_WORLD));
    }
  } else {
    return cudaErrorInvalidValue;
  }

  // get
  auto id_window = config.window[dimension][i_direction];
  mpi_invoke(MPI_Win_create(
    buffer.out, buffer.length * sizeof(double), sizeof(double), MPI_INFO_NULL,
    MPI_COMM_WORLD, &id_window));
  mpi_invoke(MPI_Win_fence(0, id_window));
  mpi_invoke(MPI_Get(
    buffer.in, buffer.length, MPI_DOUBLE, other, 0, buffer.length, MPI_DOUBLE,
    id_window));
  mpi_invoke(MPI_Win_fence(0, id_window));

  // unpack
  position = 0;
  if (direction == Direction::Upwind) {
    for (std::size_t j0 = 0; j0 < geometry.ghost_depth[0][0]; ++j0) {
      index[0]          = j0;
      const auto offset = sub2ind(index, geometry.extent, 1);
      mpi_invoke(MPI_Unpack(
        buffer.in, buffer.length, &position, host.f.data() + dof * offset, dof,
        MPI_DOUBLE, MPI_COMM_WORLD));
    }
  } else if (direction == Direction::Downwind) {
    for (std::size_t j0 = 0; j0 < geometry.ghost_depth[0][1]; ++j0) {
      index[0]          = geometry.extent[0] - geometry.ghost_depth[0][1] + j0;
      const auto offset = sub2ind(index, geometry.extent, 1);
      mpi_invoke(MPI_Unpack(
        buffer.in, buffer.length, &position, host.f.data() + dof * offset, dof,
        MPI_DOUBLE, MPI_COMM_WORLD));
    }
  } else {
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}

} // namespace ra
