#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/generate.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::read", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .file =
      {
        .handle = "read",
        .directory = "./output-test/",
      },
    .geometry =
      {
        .element =
          {
            .type = ra::MeshElementType::Line,
            .dof = 2,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  m1.config.info.step = 1;
  m1.config.info.time = 1.0;

  thrust::generate(m1.host.x.begin(), m1.host.x.end(), [&] { return 1.0; });
  thrust::generate(m1.host.f.begin(), m1.host.f.end(), [&] { return 2.0; });

  int mpi_rank{};
  REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);

  auto r = m1.write(mpi_rank);
  REQUIRE(r == RA_SUCCESS);

  m1.config.info.step = 2;
  m1.config.info.time = 2.0;

  r = m1.read(mpi_rank);
  REQUIRE(r == RA_SUCCESS);
  REQUIRE(m1.config.info.step == 1);

  using Catch::Matchers::WithinRel;
  REQUIRE_THAT(m1.config.info.time, WithinRel(1.0, 1e-14));
}
