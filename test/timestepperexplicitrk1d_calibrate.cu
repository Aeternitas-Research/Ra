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
            .time = 4,
            .space = 4,
          },
      },
    .time =
      {
        .stop = 10.0,
      },
  };
  config.space.h[0] = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D t1(config);
  const auto r = t1.calibrate();
  REQUIRE(r == RA_SUCCESS);
  REQUIRE(t1.mesh.local.config.geometry.element.dof == 4);
  REQUIRE(
    t1.mesh.local.config.geometry.extent[0] ==
    (200 / t1.mesh.config.topology.extent[0] +
     t1.mesh.local.config.geometry.ghost_depth[0][0] +
     t1.mesh.local.config.geometry.ghost_depth[0][1]));

  using Catch::Matchers::WithinRel;
  // todo
}
