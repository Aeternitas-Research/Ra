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

  e6 = e5;
  e6 = std::move(e4);
}
