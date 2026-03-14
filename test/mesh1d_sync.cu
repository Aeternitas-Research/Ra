#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/sequence.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::sync", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
    .geometry =
      {
        .element =
          {
            .dof = 3,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{5, 7}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
  int mpi_size{};
  REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) == MPI_SUCCESS);
  const auto mpi_rank_0 = mpi_rank;
  const auto mpi_rank_1m = (mpi_rank_0 + mpi_size - 1) % mpi_size;
  const auto mpi_rank_1p = (mpi_rank_0 + 1) % mpi_size;
  const auto stride_x = m1.host.x.size();
  const auto stride_f = m1.host.f.size();
  thrust::sequence(
    m1.host.x.begin(), m1.host.x.end(),
    static_cast<double>(mpi_rank * stride_x));
  thrust::sequence(
    m1.host.f.begin(), m1.host.f.end(),
    static_cast<double>(mpi_rank * stride_f));

  // upwind
  auto r = m1.sync(mpi_rank_1m, 0, ra::Direction::Upwind);
  REQUIRE(r == RA_SUCCESS);

  const auto block_x = 2 * 1;
  const auto block_f = m1.config.geometry.element.dof;
  auto ghost_depth = m1.config.geometry.ghost_depth;
  auto extent = m1.config.geometry.extent;
  for (std::size_t index = 0; index < extent[0]; ++index) {
    for (std::size_t point = 0; point < block_x; ++point) {
      REQUIRE(
        m1.host.x[point + block_x * index] ==
        static_cast<double>(point + stride_x * mpi_rank + block_x * index));
    }
  }
  for (std::size_t index = 0; index < ghost_depth[0][0]; ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.host.f[point + block_f * index] ==
        static_cast<double>(
          point + stride_f * ((mpi_rank == 0) ? mpi_size : mpi_rank) -
          block_f * (ghost_depth[0][0] + ghost_depth[0][1]) +
          block_f * index));
    }
  }
  for (std::size_t index = ghost_depth[0][0]; index < extent[0]; ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.host.f[point + block_f * index] ==
        static_cast<double>(point + stride_f * mpi_rank + block_f * index));
    }
  }

  // downwind
  r = m1.sync(mpi_rank_1p, 0, ra::Direction::Downwind);
  REQUIRE(r == RA_SUCCESS);

  for (std::size_t index = 0; index < extent[0]; ++index) {
    for (std::size_t point = 0; point < block_x; ++point) {
      REQUIRE(
        m1.host.x[point + block_x * index] ==
        static_cast<double>(point + stride_x * mpi_rank + block_x * index));
    }
  }
  for (std::size_t index = extent[0] - ghost_depth[0][1]; index < extent[0];
       ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.host.f[point + block_f * index] ==
        static_cast<double>(
          point +
          stride_f * ((mpi_rank == (mpi_size - 1)) ? (0) : (mpi_rank + 1)) +
          block_f * ghost_depth[0][0] +
          block_f * (index - (extent[0] - ghost_depth[0][1]))));
    }
  }
  for (std::size_t index = ghost_depth[0][0];
       index < extent[0] - ghost_depth[0][1]; ++index) {
    for (std::size_t point = 0; point < block_f; ++point) {
      REQUIRE(
        m1.host.f[point + block_f * index] ==
        static_cast<double>(point + stride_f * mpi_rank + block_f * index));
    }
  }
}
