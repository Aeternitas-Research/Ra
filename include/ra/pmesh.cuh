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
      int self = 0;
      int neighbor[2 * DIMENSION_MAX] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      };
    } rank{};
  } topology{};
  struct {
    MPI_Comm communicator = MPI_COMM_WORLD;
  } mpi{};
};

struct PMesh {
  ~PMesh();
  PMesh();
  PMesh(const PMesh&) = delete;
  PMesh(PMesh&&) noexcept = delete;
  PMesh(
    const int mpi_rank, const int* mpi_extent,
    const MeshConfig& config_global);
  PMesh& operator=(const PMesh&) = delete;
  PMesh& operator=(PMesh&&) noexcept = delete;

  Error calibrate(const int d_max);

  PMeshConfig config{};
};

struct PMesh1D final : PMesh {
  ~PMesh1D();
  PMesh1D();
  PMesh1D(const PMesh1D&) = delete;
  PMesh1D(PMesh1D&&) noexcept = delete;
  PMesh1D(
    const int mpi_rank, const int* mpi_extent,
    const MeshConfig& config_global);
  PMesh1D& operator=(const PMesh1D&) = delete;
  PMesh1D& operator=(PMesh1D&&) noexcept = delete;

  Error copy(const PMesh1D& other);
  Error calibrate();
  Error transfer(const cudaMemcpyKind kind, const bool x, const bool f);
  Error sync();
  Error write();
  Error read();

  // arithmetic operations
  Error assign(const OperationSpace space, const double c);
  Error assign(const OperationSpace space, PMesh1D& mesh_x);
  Error multiply(const OperationSpace space, const double c);
  Error multiply(const OperationSpace space, PMesh1D& mesh_x);
  Error add(const OperationSpace space, const double c);
  Error add(const OperationSpace space, PMesh1D& mesh_x);
  Error add(const OperationSpace space, const double c, PMesh1D& mesh_x);
  Error add(const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x);
  Error divide(const OperationSpace space, const double c);
  Error divide(const OperationSpace space, PMesh1D& mesh_x);
  Error subtract(const OperationSpace space, const double c);
  Error subtract(const OperationSpace space, PMesh1D& mesh_x);
  Error subtract(const OperationSpace space, const double c, PMesh1D& mesh_x);
  Error subtract(const OperationSpace space, PMesh1D& mesh_c, PMesh1D& mesh_x);
  Error norm(const OperationSpace space, double& r, const std::string type);

  // coordinate operations
  __host__ __device__ Error
  get_host_coordinate(Mesh1D::HostStencil& coordinate);
  __host__ __device__ Error
  get_device_coordinate(Mesh1D::DeviceStencil& coordinate);

  // stencil operations
  __host__ __device__ Error get_host_stencil(Mesh1D::HostStencil& stencil);
  __host__ __device__ Error get_device_stencil(Mesh1D::DeviceStencil& stencil);

  Mesh1D local{};
};

} // namespace ra
