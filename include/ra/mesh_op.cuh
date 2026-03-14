#pragma once

#include "ra/error.cuh"
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace ra {

template <typename T>
struct MeshOp {
  using Scalar = typename T::value_type;

  Error
  assign(T& y, const Scalar c) {
    thrust::fill(y.begin(), y.end(), c);

    return RA_SUCCESS;
  }

  Error
  assign(T& y, T& x) {
    thrust::copy(x.begin(), x.end(), y.begin());

    return RA_SUCCESS;
  }

  Error
  multiply(T& y, const Scalar c) {
    auto op = [=] __host__ __device__(const Scalar& value) {
      return value * c;
    };
    thrust::transform(y.begin(), y.end(), y.begin(), op);

    return RA_SUCCESS;
  }

  Error
  multiply(T& y, T& x) {
    cuda::zip_transform_iterator kernel{
      cuda::std::multiplies<Scalar>{}, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  add(T& y, const Scalar c) {
    auto op = [=] __host__ __device__(const Scalar& value) {
      return value + c;
    };
    thrust::transform(y.begin(), y.end(), y.begin(), op);

    return RA_SUCCESS;
  }

  Error
  add(T& y, T& x) {
    cuda::zip_transform_iterator kernel{
      cuda::std::plus<Scalar>{}, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  add(T& y, const Scalar c, T& x) {
    auto op =
      [=] __host__ __device__(const Scalar& value_1, const Scalar& value_2) {
      return value_1 + c * value_2;
    };
    cuda::zip_transform_iterator kernel{op, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  add(T& y, T& c, T& x) {
    auto op = [] __host__ __device__(
                const Scalar& value_1, const Scalar& value_2,
                const Scalar& value_3) { return value_1 + value_2 * value_3; };
    cuda::zip_transform_iterator kernel{op, y.begin(), c.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  divide(T& y, const Scalar c) {
    auto op = [=] __host__ __device__(const Scalar& value) {
      return value / c;
    };
    thrust::transform(y.begin(), y.end(), y.begin(), op);

    return RA_SUCCESS;
  }

  Error
  divide(T& y, T& x) {
    cuda::zip_transform_iterator kernel{
      cuda::std::divides<Scalar>{}, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  subtract(T& y, const Scalar c) {
    auto op = [=] __host__ __device__(const Scalar& value) {
      return value - c;
    };
    thrust::transform(y.begin(), y.end(), y.begin(), op);

    return RA_SUCCESS;
  }

  Error
  subtract(T& y, T& x) {
    cuda::zip_transform_iterator kernel{
      cuda::std::minus<Scalar>{}, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  subtract(T& y, const Scalar c, T& x) {
    auto op =
      [=] __host__ __device__(const Scalar& value_1, const Scalar& value_2) {
      return value_1 - c * value_2;
    };
    cuda::zip_transform_iterator kernel{op, y.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  subtract(T& y, T& c, T& x) {
    auto op = [] __host__ __device__(
                const Scalar& value_1, const Scalar& value_2,
                const Scalar& value_3) { return value_1 - value_2 * value_3; };
    cuda::zip_transform_iterator kernel{op, y.begin(), c.begin(), x.begin()};
    thrust::copy(kernel, kernel + y.size(), y.begin());

    return RA_SUCCESS;
  }

  Error
  norm_1(Scalar& r, T& y) {
    auto op = [] __host__ __device__(const Scalar& value) {
      return cuda::std::abs(value);
    };
    r = thrust::transform_reduce(
      y.begin(), y.end(), op, static_cast<Scalar>(0),
      cuda::std::plus<Scalar>{});

    return RA_SUCCESS;
  }

  Error
  norm_2(Scalar& r, T& y) {
    auto op = [] __host__ __device__(const Scalar& value) {
      return value * value;
    };
    r = thrust::transform_reduce(
      y.begin(), y.end(), op, static_cast<Scalar>(0),
      cuda::std::plus<Scalar>{});
    r = cuda::std::sqrt(r);

    return RA_SUCCESS;
  }

  Error
  norm_infinity(Scalar& r, T& y) {
    auto op = [] __host__ __device__(const Scalar& value) {
      return cuda::std::abs(value);
    };
    r = thrust::transform_reduce(
      y.begin(), y.end(), op, static_cast<Scalar>(0), cuda::maximum<Scalar>{});

    return RA_SUCCESS;
  }
};

} // namespace ra
