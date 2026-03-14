#include "ra/error.cuh"
#include "ra/test.cuh"
#include <utility>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Error::Error", "[error]") {
  using ra::Error;
  using ra::ErrorCategory;
  using ra::ErrorValue;

  Error e1;
  Error e2(e1);
  Error e3(std::move(e2));
  Error e4(0, ErrorCategory::Host);
  Error e5(cudaSuccess);
  Error e6(ErrorValue::Success, ErrorCategory::Device);

  e6 = s5;
  e6 = std::move(e4);
}

TEST_CASE("Error::operator==", "[error]") {
  using ra::Error;
  using ra::ErrorCategory;
  using ra::ErrorValue;

  Error e1(ErrorValue::Success, ErrorCategory::Host);
  Error e2(ErrorValue::Success, ErrorCategory::Host);
  REQUIRE(e1 == e2);

  e2.category = ErrorCategory::Device;
  REQUIRE(e1 != e2);

  Error e3(cudaSuccess);
  REQUIRE(e2 == e3);
}
