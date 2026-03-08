#pragma once

#include <cuda/iterator>
#include <cuda/std/functional>

namespace ra::dg {

// basis 0
constexpr double b0_scale = +0.5;
constexpr double b0_0_0   = +1.0;
constexpr double b0_1_0   = +1.0;
constexpr double b0_1_1   = +1.0;
constexpr double b0_2_0   = +1.0;
constexpr double b0_2_1   = +1.0;
constexpr double b0_2_2   = +1.0;
constexpr double b0_3_0   = +1.0;
constexpr double b0_3_1   = +1.0;
constexpr double b0_3_2   = +1.0;
constexpr double b0_3_3   = +1.0;
constexpr double b0_n1    = +1.0;
constexpr double b0_p1    = +1.0;
constexpr double b0dx_0_0 = +0.0;
constexpr double b0dx_1_0 = +0.0;
constexpr double b0dx_1_1 = +0.0;
constexpr double b0dx_2_0 = +0.0;
constexpr double b0dx_2_1 = +0.0;
constexpr double b0dx_2_2 = +0.0;
constexpr double b0dx_3_0 = +0.0;
constexpr double b0dx_3_1 = +0.0;
constexpr double b0dx_3_2 = +0.0;
constexpr double b0dx_3_3 = +0.0;

// basis 1
constexpr double b1_scale = +0.66666666666666666667;
constexpr double b1_0_0   = +0.0;
constexpr double b1_1_0   = -0.57735026918962584208;
constexpr double b1_1_1   = +0.57735026918962584208;
constexpr double b1_2_0   = -0.77459666924148329326;
constexpr double b1_2_1   = +0.0;
constexpr double b1_2_2   = +0.77459666924148329326;
constexpr double b1_3_0   = -0.86113631159405257254;
constexpr double b1_3_1   = -0.33998104358485631282;
constexpr double b1_3_2   = +0.33998104358485631282;
constexpr double b1_3_3   = +0.86113631159405257254;
constexpr double b1_n1    = -1.0;
constexpr double b1_p1    = +1.0;
constexpr double b1dx_0_0 = +1.0;
constexpr double b1dx_1_0 = +1.0;
constexpr double b1dx_1_1 = +1.0;
constexpr double b1dx_2_0 = +1.0;
constexpr double b1dx_2_1 = +1.0;
constexpr double b1dx_2_2 = +1.0;
constexpr double b1dx_3_0 = +1.0;
constexpr double b1dx_3_1 = +1.0;
constexpr double b1dx_3_2 = +1.0;
constexpr double b1dx_3_3 = +1.0;

// basis 2
constexpr double b2_scale = +0.4;
constexpr double b2_0_0   = -0.5;
constexpr double b2_1_0   = +0.0;
constexpr double b2_1_1   = +0.0;
constexpr double b2_2_0   = +0.4;
constexpr double b2_2_1   = -0.5;
constexpr double b2_2_2   = +0.4;
constexpr double b2_3_0   = +0.61233362071871377807;
constexpr double b2_3_1   = +0.32661933500442807965;
constexpr double b2_3_2   = +0.32661933500442807965;
constexpr double b2_3_3   = +0.61233362071871377807;
constexpr double b2_n1    = +1.0;
constexpr double b2_p1    = +1.0;
constexpr double b2dx_0_0 = +0.0;
constexpr double b2dx_1_0 = -1.73205080756887763727;
constexpr double b2dx_1_1 = +1.73205080756887763727;
constexpr double b2dx_2_0 = -2.32379000772445021283;
constexpr double b2dx_2_1 = +0.0;
constexpr double b2dx_2_2 = +2.32379000772445021283;
constexpr double b2dx_3_0 = -2.58340893478215782864;
constexpr double b2dx_3_1 = -1.01994313075456899398;
constexpr double b2dx_3_2 = +1.01994313075456899398;
constexpr double b2dx_3_3 = +2.58340893478215782864;

// basis 3
constexpr double b3_scale = +0.28571428571428571429;
constexpr double b3_0_0   = +0.0;
constexpr double b3_1_0   = +0.38490017945975046887;
constexpr double b3_1_1   = -0.38490017945975046887;
constexpr double b3_2_0   = +0.0;
constexpr double b3_2_1   = +0.0;
constexpr double b3_2_2   = +0.0;
constexpr double b3_3_0   = -0.30474698495520602393;
constexpr double b3_3_1   = +0.41172799967289963607;
constexpr double b3_3_2   = -0.41172799967289963607;
constexpr double b3_3_3   = +0.30474698495520602393;
constexpr double b3_n1    = -1.0;
constexpr double b3_p1    = +1.0;
constexpr double b3dx_0_0 = -1.5;
constexpr double b3dx_1_0 = +1.0;
constexpr double b3dx_1_1 = +1.0;
constexpr double b3dx_2_0 = +3.0;
constexpr double b3dx_2_1 = -1.5;
constexpr double b3dx_2_2 = +3.0;
constexpr double b3dx_3_0 = +4.06166810359356844629;
constexpr double b3dx_3_1 = -0.63309667502214028723;
constexpr double b3dx_3_2 = -0.63309667502214028723;
constexpr double b3dx_3_3 = +4.06166810359356844629;

// point 0
constexpr double x0_0 = +0.0;

// point 1
constexpr double x1_0 = -0.57735026918962584208;
constexpr double x1_1 = +0.57735026918962584208;

// point 2
constexpr double x2_0 = -0.77459666924148340428;
constexpr double x2_1 = +0.0;
constexpr double x2_2 = +0.77459666924148340428;

// point 3
constexpr double x3_0 = -0.86113631159405257254;
constexpr double x3_1 = -0.33998104358485631282;
constexpr double x3_2 = +0.33998104358485631282;
constexpr double x3_3 = +0.86113631159405257254;

// weight 0
constexpr double w0_0 = 2.0;

// weight 1
constexpr double w1_0 = 1.0;
constexpr double w1_1 = 1.0;

// weight 2
constexpr double w2_0 = 0.55555555555555555556;
constexpr double w2_1 = 0.88888888888888888889;
constexpr double w2_2 = 0.55555555555555555556;

// weight 3
constexpr double w3_0 = 0.34785484513745384971;
constexpr double w3_1 = 0.65214515486254620580;
constexpr double w3_2 = 0.65214515486254620580;
constexpr double w3_3 = 0.34785484513745384971;

} // namespace ra::dg

namespace ra::dg::projection {

__host__ __device__ double op_1D_0_0(const double& f0);

__host__ __device__ double op_1D_1_0(const double& f0, const double& f1);

__host__ __device__ double op_1D_1_1(const double& f0, const double& f1);

__host__ __device__ double
op_1D_2_0(const double& f0, const double& f1, const double& f2);

__host__ __device__ double
op_1D_2_1(const double& f0, const double& f1, const double& f2);

__host__ __device__ double
op_1D_2_2(const double& f0, const double& f1, const double& f2);

__host__ __device__ double op_1D_3_0(
  const double& f0, const double& f1, const double& f2, const double f3);

__host__ __device__ double op_1D_3_1(
  const double& f0, const double& f1, const double& f2, const double f3);

__host__ __device__ double op_1D_3_2(
  const double& f0, const double& f1, const double& f2, const double f3);

__host__ __device__ double op_1D_3_3(
  const double& f0, const double& f1, const double& f2, const double f3);

} // namespace ra::dg::projection

namespace ra::dg::linear {

__host__ __device__ double op_volume_0_0(const double& y0);

__host__ __device__ double op_volume_1_0(const double& y0, const double& y1);

__host__ __device__ double op_volume_1_1(const double& y0, const double& y1);

__host__ __device__ double
op_volume_2_0(const double& y0, const double& y1, const double& y2);

__host__ __device__ double
op_volume_2_1(const double& y0, const double& y1, const double& y2);

__host__ __device__ double
op_volume_2_2(const double& y0, const double& y1, const double& y2);

__host__ __device__ double op_volume_3_0(
  const double& y0, const double& y1, const double& y2, const double& y3);

__host__ __device__ double op_volume_3_1(
  const double& y0, const double& y1, const double& y2, const double& y3);

__host__ __device__ double op_volume_3_2(
  const double& y0, const double& y1, const double& y2, const double& y3);

__host__ __device__ double op_volume_3_3(
  const double& y0, const double& y1, const double& y2, const double& y3);

__host__ __device__ double
op_surface_0_0(const double& s, const double& y0_l, const double& y0_r);

__host__ __device__ double op_surface_1_0(
  const double& s, const double& y0_l, const double& y0_r, const double& y1_l,
  const double& y1_r);

__host__ __device__ double op_surface_1_1(
  const double& s, const double& y0_l, const double& y0_r, const double& y1_l,
  const double& y1_r);

__host__ __device__ double op_surface_2_0(
  const double& s, const double& y0_l, const double& y0_r, const double& y1_l,
  const double& y1_r, const double& y2_l, const double& y2_r);

__host__ __device__ double op_surface_2_1(
  const double& s, const double& y0_l, const double& y0_r, const double& y1_l,
  const double& y1_r, const double& y2_l, const double& y2_r);

__host__ __device__ double op_surface_2_2(
  const double& s, const double& y0_l, const double& y0_r, const double& y1_l,
  const double& y1_r, const double& y2_l, const double& y2_r);

__host__ __device__ double op_surface_3_0(
  const double& s, const double& y0_l, const double& y1_l, const double& y2_l,
  const double& y3_l, const double& y0_r, const double& y1_r,
  const double& y2_r, const double& y3_r);

__host__ __device__ double op_surface_3_1(
  const double& s, const double& y0_l, const double& y1_l, const double& y2_l,
  const double& y3_l, const double& y0_r, const double& y1_r,
  const double& y2_r, const double& y3_r);

__host__ __device__ double op_surface_3_2(
  const double& s, const double& y0_l, const double& y1_l, const double& y2_l,
  const double& y3_l, const double& y0_r, const double& y1_r,
  const double& y2_r, const double& y3_r);

__host__ __device__ double op_surface_3_3(
  const double& s, const double& y0_l, const double& y1_l, const double& y2_l,
  const double& y3_l, const double& y0_r, const double& y1_r,
  const double& y2_r, const double& y3_r);

} // namespace ra::dg::linear

/* macro: RA_DG_GET_KERNEL_1D_0 */
#define RA_DG_GET_KERNEL_1D_0(_x, _v0, _s0) \
  auto ra_dg_kernel_v0 = cuda::make_zip_transform_iterator((_v0), (_x).f0); \
  auto ra_dg_kernel_s0_l = \
    cuda::make_zip_transform_iterator((_s0), (_x).f0_l, (_x).f0); \
  auto ra_dg_kernel_s0_r = \
    cuda::make_zip_transform_iterator((_s0), (_x).f0, (_x).f0_r); \
  auto ra_dg_kernel_s0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0_r, ra_dg_kernel_s0_l); \
  auto ra_dg_kernel_0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0, ra_dg_kernel_v0); \
/* macro: RA_DG_GET_KERNEL_1D_0 */

/* macro: RA_DG_GET_KERNEL_1D_1 */
#define RA_DG_GET_KERNEL_1D_1(_x, _v0, _s0, _v1, _s1) \
  auto ra_dg_kernel_v0 = \
    cuda::make_zip_transform_iterator((_v0), (_x).f0, (_x).f1); \
  auto ra_dg_kernel_v1 = \
    cuda::make_zip_transform_iterator((_v1), (_x).f0, (_x).f1); \
  auto ra_dg_kernel_s0_l = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0_l, (_x).f1_l, (_x).f0, (_x).f1); \
  auto ra_dg_kernel_s1_l = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0_l, (_x).f1_l, (_x).f0, (_x).f1); \
  auto ra_dg_kernel_s0_r = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0, (_x).f1, (_x).f0_r, (_x).f1_r); \
  auto ra_dg_kernel_s1_r = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0, (_x).f1, (_x).f0_r, (_x).f1_r); \
  auto ra_dg_kernel_s0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0_r, ra_dg_kernel_s0_l); \
  auto ra_dg_kernel_s1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1_r, ra_dg_kernel_s1_l); \
  auto ra_dg_kernel_0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0, ra_dg_kernel_v0); \
  auto ra_dg_kernel_1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1, ra_dg_kernel_v1); \
/* macro: RA_DG_GET_KERNEL_1D_1 */

/* macro: RA_DG_GET_KERNEL_1D_2 */
#define RA_DG_GET_KERNEL_1D_2(_x, _v0, _s0, _v1, _s1, _v2, _s2) \
  auto ra_dg_kernel_v0 = \
    cuda::make_zip_transform_iterator((_v0), (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_v1 = \
    cuda::make_zip_transform_iterator((_v1), (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_v2 = \
    cuda::make_zip_transform_iterator((_v2), (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_s0_l = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_s1_l = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_s2_l = cuda::make_zip_transform_iterator( \
    (_s2), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f0, (_x).f1, (_x).f2); \
  auto ra_dg_kernel_s0_r = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0, (_x).f1, (_x).f2, (_x).f0_r, (_x).f1_r, (_x).f2_r); \
  auto ra_dg_kernel_s1_r = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0, (_x).f1, (_x).f2, (_x).f0_r, (_x).f1_r, (_x).f2_r); \
  auto ra_dg_kernel_s2_r = cuda::make_zip_transform_iterator( \
    (_s2), (_x).f0, (_x).f1, (_x).f2, (_x).f0_r, (_x).f1_r, (_x).f2_r); \
  auto ra_dg_kernel_s0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0_r, ra_dg_kernel_s0_l); \
  auto ra_dg_kernel_s1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1_r, ra_dg_kernel_s1_l); \
  auto ra_dg_kernel_s2 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s2_r, ra_dg_kernel_s2_l); \
  auto ra_dg_kernel_0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0, ra_dg_kernel_v0); \
  auto ra_dg_kernel_1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1, ra_dg_kernel_v1); \
  auto ra_dg_kernel_2 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s2, ra_dg_kernel_v2); \
/* macro: RA_DG_GET_KERNEL_1D_2 */

/* macro: RA_DG_GET_KERNEL_1D_3 */
#define RA_DG_GET_KERNEL_1D_3(_x, _v0, _s0, _v1, _s1, _v2, _s2, _v3, _s3) \
  auto ra_dg_kernel_v0 = cuda::make_zip_transform_iterator( \
    (_v0), (_x).f0, (_x).f1, (_x).f2, (_x).f3); \
  auto ra_dg_kernel_v1 = cuda::make_zip_transform_iterator( \
    (_v1), (_x).f0, (_x).f1, (_x).f2, (_x).f3); \
  auto ra_dg_kernel_v2 = cuda::make_zip_transform_iterator( \
    (_v2), (_x).f0, (_x).f1, (_x).f2, (_x).f3); \
  auto ra_dg_kernel_v3 = cuda::make_zip_transform_iterator( \
    (_v3), (_x).f0, (_x).f1, (_x).f2, (_x).f3); \
  auto ra_dg_kernel_s0_l = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f3_l, (_x).f0, (_x).f1, \
    (_x).f2, (_x).f3); \
  auto ra_dg_kernel_s1_l = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f3_l, (_x).f0, (_x).f1, \
    (_x).f2, (_x).f3); \
  auto ra_dg_kernel_s2_l = cuda::make_zip_transform_iterator( \
    (_s2), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f3_l, (_x).f0, (_x).f1, \
    (_x).f2, (_x).f3); \
  auto ra_dg_kernel_s3_l = cuda::make_zip_transform_iterator( \
    (_s3), (_x).f0_l, (_x).f1_l, (_x).f2_l, (_x).f3_l, (_x).f0, (_x).f1, \
    (_x).f2, (_x).f3); \
  auto ra_dg_kernel_s0_r = cuda::make_zip_transform_iterator( \
    (_s0), (_x).f0, (_x).f1, (_x).f2, (_x).f3, (_x).f0_r, (_x).f1_r, \
    (_x).f2_r, (_x).f3_r); \
  auto ra_dg_kernel_s1_r = cuda::make_zip_transform_iterator( \
    (_s1), (_x).f0, (_x).f1, (_x).f2, (_x).f3, (_x).f0_r, (_x).f1_r, \
    (_x).f2_r, (_x).f3_r); \
  auto ra_dg_kernel_s2_r = cuda::make_zip_transform_iterator( \
    (_s2), (_x).f0, (_x).f1, (_x).f2, (_x).f3, (_x).f0_r, (_x).f1_r, \
    (_x).f2_r, (_x).f3_r); \
  auto ra_dg_kernel_s3_r = cuda::make_zip_transform_iterator( \
    (_s3), (_x).f0, (_x).f1, (_x).f2, (_x).f3, (_x).f0_r, (_x).f1_r, \
    (_x).f2_r, (_x).f3_r); \
  auto ra_dg_kernel_s0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0_r, ra_dg_kernel_s0_l); \
  auto ra_dg_kernel_s1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1_r, ra_dg_kernel_s1_l); \
  auto ra_dg_kernel_s2 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s2_r, ra_dg_kernel_s2_l); \
  auto ra_dg_kernel_s3 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s3_r, ra_dg_kernel_s3_l); \
  auto ra_dg_kernel_0 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s0, ra_dg_kernel_v0); \
  auto ra_dg_kernel_1 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s1, ra_dg_kernel_v1); \
  auto ra_dg_kernel_2 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s2, ra_dg_kernel_v2); \
  auto ra_dg_kernel_3 = cuda::make_zip_transform_iterator( \
    cuda::std::minus<double>{}, ra_dg_kernel_s3, ra_dg_kernel_v3); \
/* macro: RA_DG_GET_KERNEL_1D_3 */
