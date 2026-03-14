#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/fill.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::norm 1", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .element =
          {
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
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Host, norm, "1");
    REQUIRE(r == RA_SUCCESS);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(4e+6, 1e-14));
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 3.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Device, norm, "1");
    REQUIRE(r == RA_SUCCESS);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(6e+6, 1e-14));
  }
}

TEST_CASE("Mesh1D::norm 2", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .element =
          {
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
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Host, norm, "2");
    REQUIRE(r == RA_SUCCESS);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(2828.4271247461902021, 1e-14));
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 3.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Device, norm, "2");
    REQUIRE(r == RA_SUCCESS);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(4242.6406871192848484, 1e-14));
  }
}

TEST_CASE("Mesh1D::norm infinity", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .element =
          {
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
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Host, norm, "infinity");
    REQUIRE(r == RA_SUCCESS);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(2.0, 1e-14));
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 3.0);

    double norm{};
    const auto r = m1.norm(ra::OperationSpace::Device, norm, "infinity");
    REQUIRE(r == RA_SUCCESS);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
    REQUIRE_THAT(norm, WithinRel(3.0, 1e-14));
  }
}
