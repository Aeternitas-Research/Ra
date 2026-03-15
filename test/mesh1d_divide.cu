#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/fill.h>
#include <thrust/for_each.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::divide 1", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .element =
          {
            .type = ra::MeshElementType::Line,
            .dof = 2,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 6.0);

    const auto r = m1.divide(ra::OperationSpace::Host, 2.0);
    REQUIRE(r == RA_SUCCESS);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 20.0);

    const auto r = m1.divide(ra::OperationSpace::Device, 4.0);
    REQUIRE(r == RA_SUCCESS);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(5.0, 1e-14));
    });
  }
}

TEST_CASE("Mesh1D::divide 2", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .element =
          {
            .type = ra::MeshElementType::Line,
            .dof = 2,
          },
        .extent = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);
  Mesh1D m2(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);
    thrust::fill(m2.host.f.begin(), m2.host.f.end(), 6.0);

    const auto r = m2.divide(ra::OperationSpace::Host, m1);
    REQUIRE(r == RA_SUCCESS);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 4.0);
    thrust::fill(m2.device.f.begin(), m2.device.f.end(), 20.0);

    const auto r = m2.divide(ra::OperationSpace::Device, m1);
    REQUIRE(r == RA_SUCCESS);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    m2.transfer(cudaMemcpyDeviceToHost, false, true);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(4.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(5.0, 1e-14));
    });
  }
}
