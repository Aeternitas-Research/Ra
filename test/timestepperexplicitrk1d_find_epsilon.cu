#include "ra/dg.cuh"
#include "ra/test.cuh"
#include "ra/timestepper.cuh"
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
        .stop = 10.0,
      },
  };
  config.space.h[0]    = 0.1;
  config.space.x[0][0] = -10.0;
  config.space.x[0][1] = +10.0;

  TimeStepperExplicitRK1D t1(config);
  ra_invoke(t1.calibrate());

  const double velocity = 1.0;
  auto op_volume_0 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
      return (-velocity) * ra::dg::linear::op_volume_3_0(y0, y1, y2, y3);
    };
  auto op_volume_1 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
      return (-velocity) * ra::dg::linear::op_volume_3_1(y0, y1, y2, y3);
    };
  auto op_volume_2 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
      return (-velocity) * ra::dg::linear::op_volume_3_2(y0, y1, y2, y3);
    };
  auto op_volume_3 =
    [=] __host__ __device__(double y0, double y1, double y2, double y3) {
      return (-velocity) * ra::dg::linear::op_volume_3_3(y0, y1, y2, y3);
    };
  auto op_surface_0 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) *
           ra::dg::linear::op_surface_3_0(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_1 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) *
           ra::dg::linear::op_surface_3_1(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_2 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) *
           ra::dg::linear::op_surface_3_2(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };
  auto op_surface_3 = [=] __host__ __device__(
                        double y0_l, double y1_l, double y2_l, double y3_l,
                        double y0_r, double y1_r, double y2_r, double y3_r) {
    return (-velocity) *
           ra::dg::linear::op_surface_3_3(
             velocity, y0_l, y1_l, y2_l, y3_l, y0_r, y1_r, y2_r, y3_r);
  };

  using DeviceStencil = ra::Mesh1D::DeviceStencil;
  t1.config.op.rhs    = [=] __host__(ra::PMesh1D & f, double, ra::PMesh1D& y) {
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
      thrust::transform_n(
        ra_dg_kernel_s0, n, ra_dg_kernel_v0, stencil_f.f0,
        cuda::std::minus<double>{});
      thrust::transform_n(
        ra_dg_kernel_s1, n, ra_dg_kernel_v1, stencil_f.f1,
        cuda::std::minus<double>{});
      thrust::transform_n(
        ra_dg_kernel_s2, n, ra_dg_kernel_v2, stencil_f.f2,
        cuda::std::minus<double>{});
      thrust::transform_n(
        ra_dg_kernel_s3, n, ra_dg_kernel_v3, stencil_f.f3,
        cuda::std::minus<double>{});
    }

    return cudaSuccess;
  };

  auto& rhs        = t1.config.op.rhs;
  auto& time       = t1.config.time;
  const auto space = ra::OperationSpace::Device;
  double h         = time.delta;
  ra_invoke(rhs(t1.k[0], time.now, t1.mesh));
  for (int stage = 1; stage < t1.config_extra.stage; ++stage) {
    const double* a = t1.config_extra.a[stage];
    const double c  = t1.config_extra.c[stage];

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
