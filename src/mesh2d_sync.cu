#include "ra/index.cuh"
#include "ra/mesh.cuh"
#include "ra/mpi.cuh"
#include <algorithm>
#include <mpi.h>

namespace ra {

Error
Mesh2D::sync(const int other, const int dimension, const Direction direction) {
#ifdef RA_DEBUG
  if (dimension > 1) {
    return RA_ERROR_HOST(ErrorValue::InvalidParameter);
  }
  if (!((this->config.geometry.element.type == MeshElementType::Line) ||
        (this->config.geometry.element.type ==
         MeshElementType::CurvilinearRectangle))) {
    return RA_ERROR_HOST(ErrorValue::InvalidGeometry);
  }
#endif

  auto& config = this->config;
  auto& host = this->host;

  auto& buffer = config.buffer;
  auto& geometry = config.geometry;
  const auto dof = geometry.element.dof;
  const auto i_direction = static_cast<int>(direction);

  if (
    (geometry.element.type == MeshElementType::Rectangle) ||
    (geometry.element.type == MeshElementType::CurvilinearRectangle)) {
    // find buffer extent
    for (int d = 0; d < 2; ++d) {
      buffer.extent[d] = geometry.extent[d];
    }
    for (int d = 0; d < 2; ++d) {
      if (d == dimension) {
        buffer.extent[d] =
          std::max(geometry.ghost_depth[d][0], geometry.ghost_depth[d][1]);
      } else {
        buffer.extent[d] = geometry.extent[d];
      }
    }

    // allocate buffer
    const auto length =
      dof * buffer.extent[0] * buffer.extent[1] * sizeof(double);
    if (buffer.length <= length) {
      buffer.length = length;
      if (buffer.in) {
        delete[] buffer.in;
      }
      if (buffer.out) {
        delete[] buffer.out;
      }

      buffer.in = new char[length * sizeof(double)];
      buffer.out = new char[length * sizeof(double)];
    }
  } else {
    return RA_ERROR_HOST(ErrorValue::InvalidGeometry);
  }

  // pack
  if (
    (geometry.element.type == MeshElementType::Rectangle) ||
    (geometry.element.type == MeshElementType::CurvilinearRectangle)) {
    std::size_t j_max[2] = {geometry.extent[0], geometry.extent[1]};
    std::size_t j[2] = {0, 0};
    std::size_t index[2] = {0, 0};
    int position = 0;
    if (direction == Direction::Upwind) {
      j_max[dimension] = geometry.ghost_depth[dimension][0];
      for (j[1] = 0; j[1] < j_max[1]; ++j[1]) {
        for (j[0] = 0; j[0] < j_max[0]; ++j[0]) {
          // find index
          for (int d = 0; d < 2; ++d) {
            if (d == dimension) {
              index[d] =
                geometry.extent[d] -
                (geometry.ghost_depth[d][0] + geometry.ghost_depth[d][1]) +
                j[d];
            } else {
              index[d] = j[d];
            }
          }

          // fill buffer
          const auto offset = sub2ind(index, geometry.extent, 2);
          ra_mpi_invoke(MPI_Pack(
            host.f.data() + dof * offset, dof, MPI_DOUBLE, buffer.out,
            buffer.length, &position, MPI_COMM_WORLD));
        }
      }
    } else if (direction == Direction::Downwind) {
      j_max[dimension] = geometry.ghost_depth[dimension][1];
      for (j[1] = 0; j[1] < j_max[1]; ++j[1]) {
        for (j[0] = 0; j[0] < j_max[0]; ++j[0]) {
          // find index
          for (int d = 0; d < 2; ++d) {
            if (d == dimension) {
              index[d] = geometry.ghost_depth[d][0] + j[d];
            } else {
              index[d] = j[d];
            }
          }

          // fill buffer
          const auto offset = sub2ind(index, geometry.extent, 2);
          ra_mpi_invoke(MPI_Pack(
            host.f.data() + dof * offset, dof, MPI_DOUBLE, buffer.out,
            buffer.length, &position, MPI_COMM_WORLD));
        }
      }
    } else {
      return RA_ERROR_HOST(ErrorValue::InvalidParameter);
    }
  } else {
    return RA_ERROR_HOST(ErrorValue::InvalidGeometry);
  }

  // get
  auto id_window = config.window[dimension][i_direction];
  ra_mpi_invoke(MPI_Win_create(
    buffer.out, buffer.length * sizeof(double), sizeof(double), MPI_INFO_NULL,
    MPI_COMM_WORLD, &id_window));
  ra_mpi_invoke(MPI_Win_fence(0, id_window));
  ra_mpi_invoke(MPI_Get(
    buffer.in, buffer.length, MPI_DOUBLE, other, 0, buffer.length, MPI_DOUBLE,
    id_window));
  ra_mpi_invoke(MPI_Win_fence(0, id_window));

  // unpack
  if (
    (geometry.element.type == MeshElementType::Rectangle) ||
    (geometry.element.type == MeshElementType::CurvilinearRectangle)) {
    std::size_t j_max[2] = {geometry.extent[0], geometry.extent[1]};
    std::size_t j[2] = {0, 0};
    std::size_t index[2] = {0, 0};
    int position = 0;
    if (direction == Direction::Upwind) {
      j_max[dimension] = geometry.ghost_depth[dimension][0];
      for (j[1] = 0; j[1] < j_max[1]; ++j[1]) {
        for (j[0] = 0; j[0] < j_max[0]; ++j[0]) {
          // find index
          for (int d = 0; d < 2; ++d) {
            index[d] = j[d];
          }

          // fill buffer
          const auto offset = sub2ind(index, geometry.extent, 2);
          ra_mpi_invoke(MPI_Unpack(
            buffer.in, buffer.length, &position, host.f.data() + dof * offset,
            dof, MPI_DOUBLE, MPI_COMM_WORLD));
        }
      }
    } else if (direction == Direction::Downwind) {
      j_max[dimension] = geometry.ghost_depth[dimension][1];
      for (j[1] = 0; j[1] < j_max[1]; ++j[1]) {
        for (j[0] = 0; j[0] < j_max[0]; ++j[0]) {
          // find index
          for (int d = 0; d < 2; ++d) {
            if (d == dimension) {
              index[d] =
                geometry.extent[d] - geometry.ghost_depth[d][1] + j[d];
            } else {
              index[d] = j[d];
            }
          }

          // fill buffer
          const auto offset = sub2ind(index, geometry.extent, 2);
          ra_mpi_invoke(MPI_Unpack(
            buffer.in, buffer.length, &position, host.f.data() + dof * offset,
            dof, MPI_DOUBLE, MPI_COMM_WORLD));
        }
      }
    } else {
      return RA_ERROR_HOST(ErrorValue::InvalidParameter);
    }
  } else {
    return RA_ERROR_HOST(ErrorValue::InvalidGeometry);
  }

  return cudaSuccess;
}

} // namespace ra
