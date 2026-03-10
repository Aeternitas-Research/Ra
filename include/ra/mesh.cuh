#pragma once

#include "ra/error.cuh"
#include "ra/mesh_op.cuh"
#include <cuda/iterator>
#include <cuda_runtime.h>
#include <mpi.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ra {

inline constexpr int DIMENSION_MAX = 6;

enum struct Direction : int {
  Upwind = 0,
  Downwind = 1,
};

enum struct OperationSpace : int {
  Host = 0,
  Device = 1,
  HostDevice = 2,
};

enum struct MeshElementType : int {
  Unknown = -1,
  // 1D
  Line = 0,
  // 2D
  Rectangle = 1,
  CurvilinearRectangle = 2,
  Triangle = 3,
  CurvilinearTriangle = 4,
  // 3D
  Prism = 5,
  CurvilinearPrism = 6,
  // 5D
  GyroPrismSpectral = 7,
};

struct MeshConfig {
  std::string name{};
  struct {
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;
    int mpi_rank = 0;
    std::size_t step = 0;
    double time = 0.0;
  } info{};
  struct {
    std::string name{};
    std::string handle = "mesh";
    std::string directory = "./";
  } file{};
  struct {
    struct {
      MeshElementType type = MeshElementType::Unknown;
      std::size_t dof = 1;
    } element{};
    std::size_t extent[DIMENSION_MAX] = {
      0, 0, 0, 0, 0, 0,
    };
    std::size_t ghost_depth[DIMENSION_MAX][2] = {
      {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    };
  } geometry{};
  struct {
    std::size_t extent[DIMENSION_MAX] = {
      0, 0, 0, 0, 0, 0,
    };
    std::size_t length = 0;
    char* in = nullptr;
    char* out = nullptr;
  } buffer{};
  MPI_Win window[DIMENSION_MAX][2] = {
    {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
    {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
  };
  struct {
    struct {
      int file = 0;
      struct {
        int extent[DIMENSION_MAX] = {
          0, 0, 0, 0, 0, 0,
        };
        int x[1] = {0};
        int f[1] = {0};
      } dimension{};
      struct {
        int mpi_rank = 0;
        int step = 0;
        int time = 0;
        int x = 0;
        int f = 0;
      } variable{};
    } id{};
    struct {
      struct {
        std::string x = "x";
        std::string f = "f";
      } variable{};
    } name{};
  } netcdf{};
};

struct Mesh {
  ~Mesh();
  Mesh();
  Mesh(const Mesh&) = delete;
  Mesh(Mesh&&) noexcept = delete;
  explicit Mesh(const MeshConfig& config);
  Mesh& operator=(const Mesh&) = delete;
  Mesh& operator=(Mesh&&) noexcept = delete;

  Error transfer(const cudaMemcpyKind kind, const bool x, const bool f);
  virtual Error
  sync(const int other, const int dimension, const Direction direction) = 0;
  virtual Error write(const int mpi_rank) = 0;
  virtual Error read(const int mpi_rank) = 0;

  // arithmetic operations
  Error assign(const OperationSpace space, const double c);
  Error multiply(const OperationSpace space, const double c);
  Error add(const OperationSpace space, const double c);
  Error divide(const OperationSpace space, const double c);
  Error subtract(const OperationSpace space, const double c);
  Error norm(const OperationSpace space, double& r, const std::string type);
  Error norm_1(const OperationSpace space, double& r);
  Error norm_2(const OperationSpace space, double& r);
  Error norm_infinity(const OperationSpace space, double& r);

  MeshConfig config{};
  struct {
    thrust::host_vector<double> x{};
    thrust::host_vector<double> f{};
    MeshOp<thrust::host_vector<double>> op{};
  } host{};
  struct {
    thrust::device_vector<double> x{};
    thrust::device_vector<double> f{};
    MeshOp<thrust::device_vector<double>> op{};
  } device{};
};

struct Mesh1D final : Mesh {
  using HostStencilIterator =
    cuda::strided_iterator<thrust::host_vector<double>::iterator>;
  using DeviceStencilIterator =
    cuda::strided_iterator<thrust::device_vector<double>::iterator>;

  struct HostStencil {
    HostStencilIterator x0{};
    HostStencilIterator dx0{};

    HostStencilIterator f0{};
    HostStencilIterator f0_l{};
    HostStencilIterator f0_r{};
    HostStencilIterator f1{};
    HostStencilIterator f1_l{};
    HostStencilIterator f1_r{};
    HostStencilIterator f2{};
    HostStencilIterator f2_l{};
    HostStencilIterator f2_r{};
    HostStencilIterator f3{};
    HostStencilIterator f3_l{};
    HostStencilIterator f3_r{};
  };

  struct DeviceStencil {
    DeviceStencilIterator x0{};
    DeviceStencilIterator dx0{};

    DeviceStencilIterator f0{};
    DeviceStencilIterator f0_l{};
    DeviceStencilIterator f0_r{};
    DeviceStencilIterator f1{};
    DeviceStencilIterator f1_l{};
    DeviceStencilIterator f1_r{};
    DeviceStencilIterator f2{};
    DeviceStencilIterator f2_l{};
    DeviceStencilIterator f2_r{};
    DeviceStencilIterator f3{};
    DeviceStencilIterator f3_l{};
    DeviceStencilIterator f3_r{};
  };

  ~Mesh1D();
  Mesh1D();
  Mesh1D(const Mesh1D&) = delete;
  Mesh1D(Mesh1D&&) noexcept = delete;
  explicit Mesh1D(const MeshConfig& in_config);
  Mesh1D& operator=(const Mesh1D&) = delete;
  Mesh1D& operator=(Mesh1D&&) noexcept = delete;

  Error copy(const Mesh1D& other);
  Error sync(
    const int other, const int dimension, const Direction direction) override;
  Error write(const int mpi_rank) override;
  Error read(const int mpi_rank) override;

  // arithmetic operations
  Error assign(const OperationSpace space, Mesh1D& mesh_x);
  Error multiply(const OperationSpace space, Mesh1D& mesh_x);
  Error add(const OperationSpace space, Mesh1D& mesh_x);
  Error add(const OperationSpace space, const double c, Mesh1D& mesh_x);
  Error add(const OperationSpace space, Mesh1D& mesh_c, Mesh1D& mesh_x);
  Error divide(const OperationSpace space, Mesh1D& mesh_x);
  Error subtract(const OperationSpace space, Mesh1D& mesh_x);
  Error subtract(const OperationSpace space, const double c, Mesh1D& mesh_x);
  Error subtract(const OperationSpace space, Mesh1D& mesh_c, Mesh1D& mesh_x);

  // coordinate operations
  __host__ __device__ Error get_host_coordinate(HostStencil& coordinate);
  __host__ __device__ Error get_device_coordinate(DeviceStencil& coordinate);

  // stencil operations
  __host__ __device__ Error get_host_stencil(HostStencil& stencil);
  __host__ __device__ Error get_device_stencil(DeviceStencil& stencil);
};

struct Mesh2D final : Mesh {
  using HostStencilIterator =
    cuda::strided_iterator<thrust::host_vector<double>::iterator>;
  using DeviceStencilIterator =
    cuda::strided_iterator<thrust::device_vector<double>::iterator>;

  struct HostStencil {
    HostStencilIterator x0{};
    HostStencilIterator x1{};
    HostStencilIterator dx0{};
    HostStencilIterator dx1{};

    HostStencilIterator f0{};
    HostStencilIterator f0_l{};
    HostStencilIterator f0_r{};
    HostStencilIterator f1{};
    HostStencilIterator f1_l{};
    HostStencilIterator f1_r{};
    HostStencilIterator f2{};
    HostStencilIterator f2_l{};
    HostStencilIterator f2_r{};
    HostStencilIterator f3{};
    HostStencilIterator f3_l{};
    HostStencilIterator f3_r{};
  };

  struct DeviceStencil {
    DeviceStencilIterator x0{};
    DeviceStencilIterator x1{};
    DeviceStencilIterator dx0{};
    DeviceStencilIterator dx1{};

    DeviceStencilIterator f0{};
    DeviceStencilIterator f0_l{};
    DeviceStencilIterator f0_r{};
    DeviceStencilIterator f1{};
    DeviceStencilIterator f1_l{};
    DeviceStencilIterator f1_r{};
    DeviceStencilIterator f2{};
    DeviceStencilIterator f2_l{};
    DeviceStencilIterator f2_r{};
    DeviceStencilIterator f3{};
    DeviceStencilIterator f3_l{};
    DeviceStencilIterator f3_r{};
  };

  ~Mesh2D();
  explicit Mesh2D(const MeshConfig& config);

  Error copy(const Mesh2D& other);
  Error sync(
    const int other, const int dimension, const Direction direction) override;
  Error write(const int mpi_rank) override;
  Error read(const int mpi_rank) override;

  // arithmetic operations
  Error assign(const OperationSpace space, Mesh2D& mesh_x);
  Error multiply(const OperationSpace space, Mesh2D& mesh_x);
  Error add(const OperationSpace space, Mesh2D& mesh_x);
  Error add(const OperationSpace space, const double c, Mesh2D& mesh_x);
  Error add(const OperationSpace space, Mesh2D& mesh_c, Mesh2D& mesh_x);
  Error divide(const OperationSpace space, Mesh2D& mesh_x);
  Error subtract(const OperationSpace space, Mesh2D& mesh_x);
  Error subtract(const OperationSpace space, const double c, Mesh2D& mesh_x);
  Error subtract(const OperationSpace space, Mesh2D& mesh_c, Mesh2D& mesh_x);

  // coordinate operations
  __host__ __device__ Error get_host_coordinate(HostStencil& coordinate);
  __host__ __device__ Error get_device_coordinate(DeviceStencil& coordinate);

  // stencil operations
  __host__ __device__ Error get_host_stencil(HostStencil& stencil);
  __host__ __device__ Error get_device_stencil(DeviceStencil& stencil);
};

} // namespace ra
