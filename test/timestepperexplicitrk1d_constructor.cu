#include "ra/test.cuh"
#include "ra/timestepper.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE(
  "TimeStepperExplicitRK1D::TimeStepperExplicitRK1D", "[timestepper]") {
  using ra::TimeStepperConfig;
  using ra::TimeStepperExplicitRK1D;
  using ra::TimeStepperType;

  TimeStepperExplicitRK1D t1{};

  TimeStepperConfig config = {
    .name = "test.TimeStepperExplicitRK1D",
    .type = TimeStepperType::RK,
    .time =
      {
        .stop = 10.0,
      },
  };
  config.space.h[0] = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D s2(config);
  REQUIRE(s2.config.name == config.name);
  REQUIRE(s2.config.type == config.type);
  REQUIRE(s2.config.parameter.order.time == config.parameter.order.time);
  REQUIRE(s2.config.parameter.order.space == config.parameter.order.space);

  using Catch::Matchers::WithinRel;
  REQUIRE_THAT(s2.config.time.initial, WithinRel(config.time.initial, 1e-14));
  REQUIRE_THAT(s2.config.time.stop, WithinRel(config.time.stop, 1e-14));
}
