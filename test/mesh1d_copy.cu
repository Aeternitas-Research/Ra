#include "ra/mesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::copy", "[mesh]") {
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
            .dof = 2,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  Mesh1D m2{};
  const auto r = m2.copy(m1);
  REQUIRE(r == RA_SUCCESS);
  REQUIRE(m2.config.name == m1.config.name);
  REQUIRE(m2.config.file.handle == m1.config.file.handle);
  REQUIRE(m2.config.file.directory == m1.config.file.directory);
}
