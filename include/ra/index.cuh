#pragma once

#include <cuda/std/__exception/terminate.h>

namespace ra {

enum struct IndexOrder {
  Left,
  Right,
};

template <typename T>
__host__ __device__ T
sub2ind_left(T* input, T* extent, const int dimension) {
  T stride = 1;
  T output = 0;
  for (int d = 0; d < dimension; ++d) {
    output += stride * input[d];
    stride *= extent[d];
  }

  return output;
}

template <typename T>
__host__ __device__ T
sub2ind_right(T* input, T* extent, const int dimension) {
  T stride = 1;
  T output = 0;
  for (int d = dimension - 1; d > -1; --d) {
    output += stride * input[d];
    stride *= extent[d];
  }

  return output;
}

template <IndexOrder order = IndexOrder::Left, typename T>
__host__ __device__ T
sub2ind(T* input, T* extent, const int dimension) {
  if constexpr (order == IndexOrder::Left) {
    return sub2ind_left(input, extent, dimension);
  } else if constexpr (order == IndexOrder::Right) {
    return sub2ind_right(input, extent, dimension);
  } else {
#ifdef RA_MODE_DEBUG
    cuda::std::terminate();
#endif

    return 0;
  }
}

template <typename T>
__host__ __device__ void
ind2sub_left(T* output, const T input, T* extent, const int dimension) {
  output[0] = input % extent[0];
  for (int d = 1; d < dimension; ++d) {
    output[d] = (input / extent[d - 1]) % extent[d];
  }
}

template <typename T>
__host__ __device__ void
ind2sub_right(T* output, const T input, T* extent, const int dimension) {
  output[dimension - 1] = input % extent[dimension - 1];
  for (int d = dimension - 2; d > -1; --d) {
    output[d] = (input / extent[d + 1]) % extent[d];
  }
}

template <IndexOrder order = IndexOrder::Left, typename T>
__host__ __device__ void
ind2sub(T* output, const T input, T* extent, const int dimension) {
  if constexpr (order == IndexOrder::Left) {
    return ind2sub_left(output, input, extent, dimension);
  } else if constexpr (order == IndexOrder::Right) {
    return ind2sub_right(output, input, extent, dimension);
  } else {
#ifdef RA_MODE_DEBUG
    cuda::std::terminate();
#endif
  }
}

} // namespace ra
