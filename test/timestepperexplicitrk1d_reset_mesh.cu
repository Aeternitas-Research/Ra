#include "ra/test.cuh"
#include "ra/timestepper.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("TimeStepperExplicitRK1D::reset_mesh", "[timestepper]") {
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
  ra_invoke(t1.calibrate());

  using Catch::Matchers::WithinRel;
  // todo
}
