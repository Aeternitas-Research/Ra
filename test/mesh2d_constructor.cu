#include "ra/mesh.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh2D::Mesh2D", "[mesh]") {
  using ra::Mesh2D;
  using ra::MeshConfig;

  Mesh2D m1{};

  MeshConfig config = {
    .name = "test.Mesh2D",
    .file =
      {
        .handle = "test",
        .directory = "./",
      },
    .geometry =
      {
        .element =
          {
            .type = ra::MeshElementType::Rectangle,
            .dof = 3,
          },
        .extent = {1 << 10, 1 << 10, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh2D m2(config);
  REQUIRE(m2.config.name == config.name);
  REQUIRE(m2.config.file.handle == config.file.handle);
  REQUIRE(m2.config.file.directory == config.file.directory);
  REQUIRE(m2.config.geometry.element.type == ra::MeshElementType::Line);
  REQUIRE(m2.host.x.size() == (2 * 2 * (1 << 20)));
  REQUIRE(m2.host.f.size() == (3 * (1 << 20)));
  REQUIRE(m2.device.x.size() == (2 * 2 * (1 << 20)));
  REQUIRE(m2.device.f.size() == (3 * (1 << 20)));
}
