#include "ra/dg.cuh"
#include "ra/test.cuh"
#include "ra/timestepper.cuh"
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <thrust/copy.h>
#include <thrust/transform.h>

RA_TEST_MAIN(argc, argv);

TEST_CASE("TimeStepperExplicitRK1D::find_epsilon", "[timestepper]") {
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
      },
    .time =
      {
        .initial = 1e-6,
        .stop    = 10.0,
      },
  };
  config.space.h[0]    = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D t1(config);
  ra_invoke(t1.calibrate());

  const double velocity = 1.0;
  const double dx       = t1.config.space.h[0];
  auto x_center         = cuda::make_strided_iterator(
    t1.local.device.x.begin() +
      t1.local.config.geometry.ghost_depth[0][0] * 2 + 0,
    2);
  const auto n_x =
    t1.local.geometry.extent[0] - (t1.local.geometry.ghost_depth[0][0] +
                                   t1.local.geometry.ghost_depth[0][1]);

  // set initial condition
  t1.config.op.initial =
    [=] __host__(ra::PMesh1D & f, const double t, ra::PMesh1D& buffer) {
    // sample
    auto op_sample_0 = [=] __host__ __device__(const double& x_center) {
      const auto x = x_center + dx * ra::dg::x3_0 / 2.0;

      return cuda::std::sin(x - velocity * t);
    };
    auto op_sample_1 = [=] __host__ __device__(const double& x_center) {
      const auto x = x_center + dx * ra::dg::x3_1 / 2.0;

      return cuda::std::sin(x - velocity * t);
    };
    auto op_sample_2 = [=] __host__ __device__(const double& x_center) {
      const auto x = x_center + dx * ra::dg::x3_2 / 2.0;

      return cuda::std::sin(x - velocity * t);
    };
    auto op_sample_3 = [=] __host__ __device__(const double& x_center) {
      const auto x = x_center + dx * ra::dg::x3_3 / 2.0;

      return cuda::std::sin(x - velocity * t);
    };

    DeviceStencil stencil_buffer{};
    ra_invoke(buffer.get_device_stencil(stencil_buffer));
    thrust::transform_n(x_center, n_x, stencil_buffer.f0, op_sample_0);
    thrust::transform_n(x_center, n_x, stencil_buffer.f1, op_sample_1);
    thrust::transform_n(x_center, n_x, stencil_buffer.f2, op_sample_2);
    thrust::transform_n(x_center, n_x, stencil_buffer.f3, op_sample_3);

    // projection
    auto op_projection_0 = cuda::make_zip_transform_iterator(
      ra::dg::projection::op::1D_3_0, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_1 = cuda::make_zip_transform_iterator(
      ra::dg::projection::op::1D_3_1, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_2 = cuda::make_zip_transform_iterator(
      ra::dg::projection::op::1D_3_2, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);
    auto op_projection_3 = cuda::make_zip_transform_iterator(
      ra::dg::projection::op::1D_3_3, stencil_buffer.f0, stencil_buffer.f1,
      stencil_buffer.f2, stencil_buffer.f3);

    DeviceStencil stencil_f{};
    ra_invoke(f.get_device_stencil(stencil_f));
    thrust::copy_n(op_projection_0, n_x, stencil_f.f0);
    thrust::copy_n(op_projection_1, n_x, stencil_f.f1);
    thrust::copy_n(op_projection_2, n_x, stencil_f.f2);
    thrust::copy_n(op_projection_3, n_x, stencil_f.f3);

    return cudaSuccess;
  };

  // set boundary conditions
  t1.config.op.boundary =
    [=] __host__(ra::PMesh1D & f, const double, PMesh1D&) { return f.sync(); };

  const double dt = t1.config.time.initial;
  auto op_volume_0 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_volume_3_0(y0, y1, y2, y3);
  };
  auto op_volume_1 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_volume_3_1(y0, y1, y2, y3);
  };
  auto op_volume_2 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_volume_3_2(y0, y1, y2, y3);
  };
  auto op_volume_3 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_volume_3_3(y0, y1, y2, y3);
  };
  auto op_surface_0 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_surface_3_0(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_1 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_surface_3_1(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_2 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_surface_3_2(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_3 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) * (dt / dx) *
           ra::dg::linear::op_surface_3_3(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };

  // set RHS
  using DeviceStencil = ra::Mesh1D::DeviceStencil;
  t1.config.op.rhs =
    [=] __host__(ra::PMesh1D & f, const double, ra::PMesh1D& y) {
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

  auto& rhs        = t1.config.op.rhs;
  auto& time       = t1.config.time;
  auto& rk         = t1.config.parameter.table.rk_explicit;
  const auto space = ra::OperationSpace::Device;
  double h         = time.delta;
  ra_invoke(rhs(t1.k[0], time.now, t1.mesh));
  for (int stage = 1; stage < rk.stage; ++stage) {
    const double* a = rk.a[stage];
    const double c  = rk.c[stage];

    t1.buffer.assign(space, t1.backup);
    for (int index = 0; index < stage; ++index) {
      t1.buffer.add(space, a[index] * h, t1.k[index]);
    }

    ra_invoke(rhs(t1.k[stage], time.now + c * h, t1.buffer));
  }

  double epsilon = 0.0;
  const auto r   = t1.find_epsilon(epsilon);
  REQUIRE(r == cudaSuccess);
}
