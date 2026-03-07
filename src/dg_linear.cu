#include "ra/dg.cuh"

namespace ra::dg::linear {

__host__ __device__ double
op_volume_3_0(double y0, double y1, double y2, double y3) {
  return (y0 + b0_3_0 + y1 * b1_3_0 + y2 * b2_3_0 + y3 * b3_3_0) * b0dx_3_0 *
           w3_0 +
         (y0 + b0_3_1 + y1 * b1_3_1 + y2 * b2_3_1 + y3 * b3_3_1) * b0dx_3_1 *
           w3_1 +
         (y0 + b0_3_2 + y1 * b1_3_2 + y2 * b2_3_2 + y3 * b3_3_2) * b0dx_3_2 *
           w3_2 +
         (y0 + b0_3_3 + y1 * b1_3_3 + y2 * b2_3_3 + y3 * b3_3_3) * b0dx_3_3 *
           w3_3;
}

__host__ __device__ double
op_volume_3_1(double y0, double y1, double y2, double y3) {
  return (y0 + b0_3_0 + y1 * b1_3_0 + y2 * b2_3_0 + y3 * b3_3_0) * b1dx_3_0 *
           w3_0 +
         (y0 + b0_3_1 + y1 * b1_3_1 + y2 * b2_3_1 + y3 * b3_3_1) * b1dx_3_1 *
           w3_1 +
         (y0 + b0_3_2 + y1 * b1_3_2 + y2 * b2_3_2 + y3 * b3_3_2) * b1dx_3_2 *
           w3_2 +
         (y0 + b0_3_3 + y1 * b1_3_3 + y2 * b2_3_3 + y3 * b3_3_3) * b1dx_3_3 *
           w3_3;
}

__host__ __device__ double
op_volume_3_2(double y0, double y1, double y2, double y3) {
  return (y0 + b0_3_0 + y1 * b1_3_0 + y2 * b2_3_0 + y3 * b3_3_0) * b2dx_3_0 *
           w3_0 +
         (y0 + b0_3_1 + y1 * b1_3_1 + y2 * b2_3_1 + y3 * b3_3_1) * b2dx_3_1 *
           w3_1 +
         (y0 + b0_3_2 + y1 * b1_3_2 + y2 * b2_3_2 + y3 * b3_3_2) * b2dx_3_2 *
           w3_2 +
         (y0 + b0_3_3 + y1 * b1_3_3 + y2 * b2_3_3 + y3 * b3_3_3) * b2dx_3_3 *
           w3_3;
}

__host__ __device__ double
op_volume_3_3(double y0, double y1, double y2, double y3) {
  return (y0 + b0_3_0 + y1 * b1_3_0 + y2 * b2_3_0 + y3 * b3_3_0) * b3dx_3_0 *
           w3_0 +
         (y0 + b0_3_1 + y1 * b1_3_1 + y2 * b2_3_1 + y3 * b3_3_1) * b3dx_3_1 *
           w3_1 +
         (y0 + b0_3_2 + y1 * b1_3_2 + y2 * b2_3_2 + y3 * b3_3_2) * b3dx_3_2 *
           w3_2 +
         (y0 + b0_3_3 + y1 * b1_3_3 + y2 * b2_3_3 + y3 * b3_3_3) * b3dx_3_3 *
           w3_3;
}

__host__ __device__ double
op_surface_3_0(
  double s, double y0_l, double y1_l, double y2_l, double y3_l, double y0_r,
  double y1_r, double y2_r, double y3_r) {
  if (s > 0.0) {
    return (y0_l * b0_p1 + y1_l * b1_p1 + y2_l * b2_p1 + y3_l * b3_p1) * b0_p1;
  } else if (s < 0.0) {
    return (y0_r * b0_n1 + y1_r * b1_n1 + y2_r * b2_n1 + y3_r * b3_n1) * b0_n1;
  } else {
    return 0.0;
  }
}

__host__ __device__ double
op_surface_3_1(
  double s, double y0_l, double y1_l, double y2_l, double y3_l, double y0_r,
  double y1_r, double y2_r, double y3_r) {
  if (s > 0.0) {
    return (y0_l * b0_p1 + y1_l * b1_p1 + y2_l * b2_p1 + y3_l * b3_p1) * b1_p1;
  } else if (s < 0.0) {
    return (y0_r * b0_n1 + y1_r * b1_n1 + y2_r * b2_n1 + y3_r * b3_n1) * b1_n1;
  } else {
    return 0.0;
  }
}

__host__ __device__ double
op_surface_3_2(
  double s, double y0_l, double y1_l, double y2_l, double y3_l, double y0_r,
  double y1_r, double y2_r, double y3_r) {
  if (s > 0.0) {
    return (y0_l * b0_p1 + y1_l * b1_p1 + y2_l * b2_p1 + y3_l * b3_p1) * b2_p1;
  } else if (s < 0.0) {
    return (y0_r * b0_n1 + y1_r * b1_n1 + y2_r * b2_n1 + y3_r * b3_n1) * b2_n1;
  } else {
    return 0.0;
  }
}

__host__ __device__ double
op_surface_3_3(
  double s, double y0_l, double y1_l, double y2_l, double y3_l, double y0_r,
  double y1_r, double y2_r, double y3_r) {
  if (s > 0.0) {
    return (y0_l * b0_p1 + y1_l * b1_p1 + y2_l * b2_p1 + y3_l * b3_p1) * b3_p1;
  } else if (s < 0.0) {
    return (y0_r * b0_n1 + y1_r * b1_n1 + y2_r * b2_n1 + y3_r * b3_n1) * b3_n1;
  } else {
    return 0.0;
  }
}

} // namespace ra::dg::linear
