#include "ra/test.cuh"
#include "ra/timestepper.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("TimeStepperExplicitRK1D::calibrate", "[timestepper]") {
  using ra::TimeStepperConfig;
  using ra::TimeStepperExplicitRK1D;
  using ra::TimeStepperType;

  TimeStepperConfig config = {
    .name = "test.TimeStepperExplicitRK1D",
    .type = TimeStepperType::RK,
    .parameter =
      {
        .order =
          {
            .time  = 4,
            .space = 4,
          },
      },
    .time =
      {
        .stop = 10.0,
      },
  };
  config.space.h[0]    = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D s1(config);
  const auto r = s1.calibrate();
  REQUIRE(r == cudaSuccess);
  REQUIRE(s1.mesh.local.config.geometry.dof == 4);
  REQUIRE(
    s1.mesh.local.config.geometry.extent[0] ==
    (200 / s1.mesh.config.topology.extent[0] +
     s1.mesh.local.config.geometry.ghost_depth[0][0] +
     s1.mesh.local.config.geometry.ghost_depth[0][1]));

  using Catch::Matchers::WithinRel;
  // todo
}
