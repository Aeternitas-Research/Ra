#include "ra/error.cuh"
#include "ra/test.cuh"

RA_TEST_MAIN(argc, argv);

TEST_CASE("Error::operator==", "[error]") {
  using ra::Error;
  using ra::ErrorCategory;
  using ra::ErrorValue;

  Error e1(ErrorValue::Success, ErrorCategory::Host);
  Error e2(ErrorValue::Success, ErrorCategory::Host);
  REQUIRE(e1 == e2);

  e2.category = ErrorCategory::Device;
  Error e3(cudaSuccess);
  REQUIRE(e2 == e3);
}

TEST_CASE("Error::operator!=", "[error]") {
  using ra::Error;
  using ra::ErrorCategory;
  using ra::ErrorValue;

  Error e1(ErrorValue::Success, ErrorCategory::Host);
  Error e2(ErrorValue::Success, ErrorCategory::Device);
  REQUIRE(e1 != e2);

  e2.category = ErrorCategory::Host;
  e1.value = 1;
  REQUIRE(e1 != e2);
}
