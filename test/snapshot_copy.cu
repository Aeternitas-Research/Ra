#include "ra/snapshot.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Snapshot::copy", "[snapshot]") {
  using ra::Snapshot;
  using ra::SnapshotConfig;

  SnapshotConfig config = {
    .name = "test.Snapshot",
    .time =
      {
        .start = 0.0,
        .stop  = 1.0,
        .now   = 0.5,
        .delta = 0.1,
      },
  };
  Snapshot s1(config);

  Snapshot s2{};
  const auto r = s2.copy(s1);
  REQUIRE(r == cudaSuccess);
  REQUIRE(s2.config.get() == s1.config.get());
}
