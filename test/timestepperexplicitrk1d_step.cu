#include "ra/dg.cuh"
#include "ra/test.cuh"
#include "ra/timestepper.cuh"
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <thrust/copy.h>
#include <thrust/transform.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("TimeStepperExplicitRK1D::step", "[timestepper]") {
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
            .time  = 4,
            .space = 4,
          },
        .adaptivity =
          {
            .time =
              {
                .k1 = +0.58,
                .k2 = -0.21,
                .k3 = +0.1,
                .k4 = +0.0,
                .k5 = +0.0,
              },
          },
      },
    .time =
      {
        .stop  = 10.0,
        .delta = 1e-6,
      },
  };
  config.space.h[0]    = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D t1(config);
  ra_invoke(t1.calibrate());

  const double velocity = 1.0;
  const double dx       = t1.config.space.h[0];
  const double t_now    = t1.config.time.now;

  auto f_initial = [=] __device__(const double& x) -> double {
    return cuda::std::sin(x - velocity * t_now);
  };

  // sample
  auto op_sample_0 = [=] __device__(const double& x_center) -> double {
    const auto x = x_center + dx * ra::dg::x3_0 / 2.0;

    return f_initial(x);
  };
  auto op_sample_1 = [=] __device__(const double& x_center) -> double {
    const auto x = x_center + dx * ra::dg::x3_1 / 2.0;

    return f_initial(x);
  };
  auto op_sample_2 = [=] __device__(const double& x_center) -> double {
    const auto x = x_center + dx * ra::dg::x3_2 / 2.0;

    return f_initial(x);
  };
  auto op_sample_3 = [=] __device__(const double& x_center) -> double {
    const auto x = x_center + dx * ra::dg::x3_3 / 2.0;

    return f_initial(x);
  };

  // projection
  RA_DG_GET_PROJECTION_1D_3();

  // set initial condition
  using DeviceStencil  = ra::Mesh1D::DeviceStencil;
  t1.config.op.initial = [=](ra::PMesh1D& f, ra::PMesh1D& buffer) {
    const auto& geometry = f.local.config.geometry;
    const auto n = geometry.extent[0] -
                   (geometry.ghost_depth[0][0] + geometry.ghost_depth[0][1]);

    DeviceStencil stencil_buffer{};
    ra_invoke(buffer.get_device_coordinate(stencil_buffer));
    ra_invoke(buffer.get_device_stencil(stencil_buffer));
    thrust::transform_n(stencil_buffer.x0, n, stencil_buffer.f0, op_sample_0);
    thrust::transform_n(stencil_buffer.x0, n, stencil_buffer.f1, op_sample_1);
    thrust::transform_n(stencil_buffer.x0, n, stencil_buffer.f2, op_sample_2);
    thrust::transform_n(stencil_buffer.x0, n, stencil_buffer.f3, op_sample_3);

    // projection
    auto op_projection_0 = cuda::make_zip_transform_iterator(
      ra_dg_projection_0, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_1 = cuda::make_zip_transform_iterator(
      ra_dg_projection_1, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_2 = cuda::make_zip_transform_iterator(
      ra_dg_projection_2, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_3 = cuda::make_zip_transform_iterator(
      ra_dg_projection_3, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);

    DeviceStencil stencil_f{};
    ra_invoke(f.get_device_stencil(stencil_f));
    thrust::copy_n(op_projection_0, n, stencil_f.f0);
    thrust::copy_n(op_projection_1, n, stencil_f.f1);
    thrust::copy_n(op_projection_2, n, stencil_f.f2);
    thrust::copy_n(op_projection_3, n, stencil_f.f3);

    return cudaSuccess;
  };

  // set boundary conditions
  t1.config.op.boundary = [=](ra::PMesh1D& f, const double, ra::PMesh1D&) {
    return f.sync();
  };

  auto op_volume_0 = [=] __device__(
                       const double& y0, const double& y1, const double& y2,
                       const double& y3) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_volume_3_0(y0, y1, y2, y3);
  };
  auto op_volume_1 = [=] __device__(
                       const double& y0, const double& y1, const double& y2,
                       const double& y3) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_volume_3_1(y0, y1, y2, y3);
  };
  auto op_volume_2 = [=] __device__(
                       const double& y0, const double& y1, const double& y2,
                       const double& y3) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_volume_3_2(y0, y1, y2, y3);
  };
  auto op_volume_3 = [=] __device__(
                       const double& y0, const double& y1, const double& y2,
                       const double& y3) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_volume_3_3(y0, y1, y2, y3);
  };
  auto op_surface_0 = [=] __device__(
                        const double& y0_l, const double& y1_l,
                        const double& y2_l, const double& y3_l,
                        const double& y0_r, const double& y1_r,
                        const double& y2_r, const double& y3_r) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_surface_3_0(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_1 = [=] __device__(
                        const double& y0_l, const double& y1_l,
                        const double& y2_l, const double& y3_l,
                        const double& y0_r, const double& y1_r,
                        const double& y2_r, const double& y3_r) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_surface_3_1(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_2 = [=] __device__(
                        const double& y0_l, const double& y1_l,
                        const double& y2_l, const double& y3_l,
                        const double& y0_r, const double& y1_r,
                        const double& y2_r, const double& y3_r) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_surface_3_2(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_3 = [=] __device__(
                        const double& y0_l, const double& y1_l,
                        const double& y2_l, const double& y3_l,
                        const double& y0_r, const double& y1_r,
                        const double& y2_r, const double& y3_r) -> double {
    return (-velocity) * (2.0 / dx) *
           ra::dg::linear::op_surface_3_3(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };

  // set RHS
  t1.config.op.rhs = [=](ra::PMesh1D& f, const double, ra::PMesh1D& y) {
    DeviceStencil stencil_y{};
    ra_invoke(y.get_device_stencil(stencil_y));

    DeviceStencil stencil_f{};
    ra_invoke(f.get_device_stencil(stencil_f));

    {
      // generate kernels
      RA_DG_GET_KERNEL_1D_3(
        stencil_y, op_volume_0, op_surface_0, op_volume_1, op_surface_1,
        op_volume_2, op_surface_2, op_volume_3, op_surface_3);

      const auto& geometry = y.local.config.geometry;
      const auto n = geometry.extent[0] -
                     (geometry.ghost_depth[0][0] + geometry.ghost_depth[0][1]);

      // output
      thrust::copy_n(ra_dg_kernel_0, n, stencil_f.f0);
      thrust::copy_n(ra_dg_kernel_1, n, stencil_f.f1);
      thrust::copy_n(ra_dg_kernel_2, n, stencil_f.f2);
      thrust::copy_n(ra_dg_kernel_3, n, stencil_f.f3);
    }

    return cudaSuccess;
  };

  const auto r = t1.step();
  REQUIRE(r == cudaSuccess);
}
