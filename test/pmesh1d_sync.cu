#include "ra/pmesh.cuh"
#include "ra/test.cuh"
#include <thrust/sequence.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("PMesh1D::sync", "[pmesh]") {
  using ra::MeshConfig;
  using ra::PMesh1D;

  MeshConfig config = {
    .name = "test.PMesh1D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
    .geometry =
      {
        .dof = 3,
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{5, 7}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
  int mpi_extent[6];
  REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, mpi_extent) == MPI_SUCCESS);
  PMesh1D m1(mpi_rank, mpi_extent, config);
  m1.calibrate();

  const auto stride_x = m1.local.host.x.size();
  const auto stride_f = m1.local.host.f.size();
  thrust::sequence(
    m1.local.host.x.begin(), m1.local.host.x.end(),
    static_cast<double>(mpi_rank * stride_x));
  thrust::sequence(
    m1.local.host.f.begin(), m1.local.host.f.end(),
    static_cast<double>(mpi_rank * stride_f));

  const auto r = m1.sync();
  REQUIRE(r == cudaSuccess);

  const auto block_x = 2 * 1;
  const auto block_f = m1.config.local.geometry.dof;
  auto ghost_depth = m1.config.local.geometry.ghost_depth;
  auto extent = m1.config.local.geometry.extent;
  for (std::size_t index = 0; index < extent[0]; ++index) {
    for (std::size_t point = 0; point < block_x; ++point) {
      REQUIRE(
        m1.local.host.x[point + block_x * index] ==
        static_cast<double>(point + stride_x * mpi_rank + block_x * index));
    }
  }
  for (std::size_t index = 0; index < ghost_depth[0][0]; ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.local.host.f[point + block_f * index] ==
        static_cast<double>(
          point + stride_f * ((mpi_rank == 0) ? mpi_extent[0] : mpi_rank) -
          block_f * (ghost_depth[0][0] + ghost_depth[0][1]) +
          block_f * index));
    }
  }
  for (std::size_t index = extent[0] - ghost_depth[0][1]; index < extent[0];
       ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.local.host.f[point + block_f * index] ==
        static_cast<double>(
          point +
          stride_f *
            ((mpi_rank == (mpi_extent[0] - 1)) ? (0) : (mpi_rank + 1)) +
          block_f * ghost_depth[0][0] +
          block_f * (index - (extent[0] - ghost_depth[0][1]))));
    }
  }
  for (std::size_t index = ghost_depth[0][0];
       index < extent[0] - ghost_depth[0][1]; ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.local.host.f[point + block_f * index] ==
        static_cast<double>(point + stride_f * mpi_rank + block_f * index));
    }
  }
}
