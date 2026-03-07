#include "ra/mesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::Mesh1D", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  Mesh1D m1{};

  MeshConfig config = {
    .name = "test.Mesh1D",
    .file =
      {
        .handle    = "test",
        .directory = "./",
      },
    .geometry =
      {
        .dof         = 3,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m2(config);
  REQUIRE(m2.config.name == config.name);
  REQUIRE(m2.config.file.handle == config.file.handle);
  REQUIRE(m2.config.file.directory == config.file.directory);
  REQUIRE(m2.host.x.size() == 2'000'000);
  REQUIRE(m2.host.f.size() == 3'000'000);
  REQUIRE(m2.device.x.size() == 2'000'000);
  REQUIRE(m2.device.f.size() == 3'000'000);
}
