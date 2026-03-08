#include "ra/dg.cuh"

namespace ra::dg::projection {

__host__ __device__ double
op_1D_0_0(const double& f0) {
  return (b0_0_0 * f0 * w0_0) / b0_scale;
}

__host__ __device__ double
op_1D_1_0(const double& f0, const double& f1) {
  return (b0_1_0 * f0 * w1_0 + b0_1_1 * f1 * w1_1) / b0_scale;
}

__host__ __device__ double
op_1D_1_1(const double& f0, const double& f1) {
  return (b1_1_0 * f0 * w1_0 + b1_1_1 * f1 * w1_1) / b1_scale;
}

__host__ __device__ double
op_1D_2_0(const double& f0, const double& f1, const double& f2) {
  return (b0_2_0 * f0 * w2_0 + b0_2_1 * f1 * w2_1 + b0_2_2 * f2 * w2_2) /
         b0_scale;
}

__host__ __device__ double
op_1D_2_1(const double& f0, const double& f1, const double& f2) {
  return (b1_2_0 * f0 * w2_0 + b1_2_1 * f1 * w2_1 + b1_2_2 * f2 * w2_2) /
         b1_scale;
}

__host__ __device__ double
op_1D_2_2(const double& f0, const double& f1, const double& f2) {
  return (b2_2_0 * f0 * w2_0 + b2_2_1 * f1 * w2_1 + b2_2_2 * f2 * w2_2) /
         b2_scale;
}

__host__ __device__ double
op_1D_3_0(
  const double& f0, const double& f1, const double& f2, const double f3) {
  return (b0_3_0 * f0 * w3_0 + b0_3_1 * f1 * w3_1 + b0_3_2 * f2 * w3_2 +
          b0_3_3 * f3 * w3_3) /
         b0_scale;
}

__host__ __device__ double
op_1D_3_1(
  const double& f0, const double& f1, const double& f2, const double f3) {
  return (b1_3_0 * f0 * w3_0 + b1_3_1 * f1 * w3_1 + b1_3_2 * f2 * w3_2 +
          b1_3_3 * f3 * w3_3) /
         b1_scale;
}

__host__ __device__ double
op_1D_3_2(
  const double& f0, const double& f1, const double& f2, const double f3) {
  return (b2_3_0 * f0 * w3_0 + b2_3_1 * f1 * w3_1 + b2_3_2 * f2 * w3_2 +
          b2_3_3 * f3 * w3_3) /
         b2_scale;
}

__host__ __device__ double
op_1D_3_3(
  const double& f0, const double& f1, const double& f2, const double f3) {
  return (b3_3_0 * f0 * w3_0 + b3_3_1 * f1 * w3_1 + b3_3_2 * f2 * w3_2 +
          b3_3_3 * f3 * w3_3) /
         b3_scale;
}

} // namespace ra::dg::projection
