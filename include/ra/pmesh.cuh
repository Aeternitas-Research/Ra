#pragma once

#include "ra/error.cuh"
#include "ra/mesh.cuh"
#include <cuda/std/utility>
#include <mpi.h>

namespace ra {

struct PMeshConfig {
  MeshConfig global{};
  MeshConfig local{};
  struct {
    int extent[DIMENSION_MAX] = {
      0, 0, 0, 0, 0, 0,
    };
    int self[DIMENSION_MAX] = {
      0, 0, 0, 0, 0, 0,
    };
    int neighbor[2 * DIMENSION_MAX][DIMENSION_MAX] = {
      {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0},
    };
    struct {
      int self                        = 0;
      int neighbor[2 * DIMENSION_MAX] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      };
    } rank{};
  } topology{};
  struct {
    MPI_Comm communicator = MPI_COMM_WORLD;
  } mpi{};
};

struct PMesh1D {
  __host__ ~PMesh1D();
  __host__ PMesh1D();
  PMesh1D(const PMesh1D&)     = delete;
  PMesh1D(PMesh1D&&) noexcept = delete;
  __host__ PMesh1D(
    const int mpi_rank, const int* mpi_extent,
    const MeshConfig& config_global);
  PMesh1D& operator=(const PMesh1D&)     = delete;
  PMesh1D& operator=(PMesh1D&&) noexcept = delete;

  __host__ Error copy(const PMesh1D& other);
  __host__ Error calibrate();
  __host__ Error
  transfer(const cudaMemcpyKind kind, const bool x, const bool f);
  __host__ Error sync();
  __host__ Error write();
  __host__ Error read();

  // arithmetic operations
  __host__ Error assign(const OperationSpace space, const double c);
  __host__ Error assign(const OperationSpace space, PMesh1D& mesh_x);
  __host__ Error multiply(const OperationSpace space, const double c);
  __host__ Error multiply(const OperationSpace space, PMesh1D& mesh_x);
  __host__ Error add(const OperationSpace space, const double c);
  __host__ Error add(const OperationSpace space, PMesh1D& mesh_x);
  __host__ Error
  add(const OperationSpace space, const double c, PMesh1D& mesh_x);
  __host__ Error
  add(const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x);
  __host__ Error divide(const OperationSpace space, const double c);
  __host__ Error divide(const OperationSpace space, PMesh1D& mesh_x);
  __host__ Error subtract(const OperationSpace space, const double c);
  __host__ Error subtract(const OperationSpace space, PMesh1D& mesh_x);
  __host__ Error
  subtract(const OperationSpace space, const double c, PMesh1D& mesh_x);
  __host__ Error
  subtract(const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x);
  __host__ Error
  norm(const OperationSpace space, double& r, const std::string type);

  // stencil operations
  __host__ __device__ Error get_host_stencil(Mesh1D::HostStencil& stencil);
  __host__ __device__ Error get_device_stencil(Mesh1D::DeviceStencil& stencil);

  PMeshConfig config{};
  Mesh1D local{};
};

} // namespace ra
