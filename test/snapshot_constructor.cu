#include "ra/snapshot.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Snapshot::Snapshot", "[snapshot]") {
  using ra::Snapshot;
  using ra::SnapshotConfig;

  Snapshot s1{};
  REQUIRE(s1.config.get() == nullptr);

  SnapshotConfig config = {
    .name = "test.Snapshot",
    .time =
      {
        .start = 0.0,
        .stop = 1.0,
        .now = 0.5,
        .delta = 0.1,
      },
  };
  Snapshot s2(config);
  REQUIRE(s2.config.get() != nullptr);
}
