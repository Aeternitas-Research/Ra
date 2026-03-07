#include "ra/mesh.cuh"
#include "ra/test.cuh"
#include <thrust/fill.h>
#include <thrust/for_each.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("Mesh1D::add 1", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .dof         = 2,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 1.0);

    const auto r = m1.add(ra::OperationSpace::Host, 2.0);
    REQUIRE(r == cudaSuccess);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 3.0);

    const auto r = m1.add(ra::OperationSpace::Device, 4.0);
    REQUIRE(r == cudaSuccess);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(7.0, 1e-14));
    });
  }
}

TEST_CASE("Mesh1D::add 2", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .dof         = 2,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);
  Mesh1D m2(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 1.0);
    thrust::fill(m2.host.f.begin(), m2.host.f.end(), 2.0);

    const auto r = m2.add(ra::OperationSpace::Host, m1);
    REQUIRE(r == cudaSuccess);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(1.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 3.0);
    thrust::fill(m2.device.f.begin(), m2.device.f.end(), 4.0);

    const auto r = m2.add(ra::OperationSpace::Device, m1);
    REQUIRE(r == cudaSuccess);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    m2.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(7.0, 1e-14));
    });
  }
}

TEST_CASE("Mesh1D::add 3", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .dof         = 2,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);
  Mesh1D m2(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);
    thrust::fill(m2.host.f.begin(), m2.host.f.end(), 3.0);

    const auto r = m2.add(ra::OperationSpace::Host, 4.0, m1);
    REQUIRE(r == cudaSuccess);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(11.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 5.0);
    thrust::fill(m2.device.f.begin(), m2.device.f.end(), 6.0);

    const auto r = m2.add(ra::OperationSpace::Device, 7.0, m1);
    REQUIRE(r == cudaSuccess);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    m2.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(5.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(41.0, 1e-14));
    });
  }
}

TEST_CASE("Mesh1D::add 4", "[mesh]") {
  using ra::Mesh1D;
  using ra::MeshConfig;

  MeshConfig config = {
    .name = "test.Mesh1D",
    .geometry =
      {
        .dof         = 2,
        .extent      = {1'000'000, 0, 0, 0, 0, 0},
        .ghost_depth = {{1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
      },
  };
  Mesh1D m1(config);
  Mesh1D c(config);
  Mesh1D m2(config);

  using Catch::Matchers::WithinRel;

  // host
  {
    thrust::fill(m1.host.f.begin(), m1.host.f.end(), 2.0);
    thrust::fill(c.host.f.begin(), c.host.f.end(), 3.0);
    thrust::fill(m2.host.f.begin(), m2.host.f.end(), 4.0);

    const auto r = m2.add(ra::OperationSpace::Host, c, m1);
    REQUIRE(r == cudaSuccess);

    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(2.0, 1e-14));
    });
    thrust::for_each(c.host.f.begin(), c.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(3.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(10.0, 1e-14));
    });
  }

  // device
  {
    thrust::fill(m1.device.f.begin(), m1.device.f.end(), 5.0);
    thrust::fill(c.device.f.begin(), c.device.f.end(), 6.0);
    thrust::fill(m2.device.f.begin(), m2.device.f.end(), 7.0);

    const auto r = m2.add(ra::OperationSpace::Device, c, m1);
    REQUIRE(r == cudaSuccess);

    m1.transfer(cudaMemcpyDeviceToHost, false, true);
    c.transfer(cudaMemcpyDeviceToHost, false, true);
    m2.transfer(cudaMemcpyDeviceToHost, false, true);
    thrust::for_each(m1.host.f.begin(), m1.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(5.0, 1e-14));
    });
    thrust::for_each(c.host.f.begin(), c.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(6.0, 1e-14));
    });
    thrust::for_each(m2.host.f.begin(), m2.host.f.end(), [&](double value) {
      REQUIRE_THAT(value, WithinRel(37.0, 1e-14));
    });
  }
}
