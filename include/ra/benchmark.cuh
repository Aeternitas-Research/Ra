#pragma once

#include "ra/mesh.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace ra::benchmark {

template <typename T, int block_size_x>
__global__ void
block_reduce(const T* data, T* result, std::size_t n) {
  using BlockReduce = cub::BlockReduce<T, block_size_x>;
  __shared__ typename BlockReduce::TempStorage storage;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  T sum = 0;
  if (index < n) {
    sum += data[index];
  }

  sum = BlockReduce(storage).Sum(sum);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<T, cuda::thread_scope_device> result_atomic(*result);
    result_atomic.fetch_add(sum, cuda::memory_order_relaxed);
  }
}

__host__ void
mesh1d_add(const ra::OperationSpace space, Mesh1D& y, Mesh1D& c, Mesh1D& x);

template <typename T>
__host__ void
stream_sum(
  thrust::device_vector<T>& r, const thrust::device_vector<T> x1,
  const thrust::device_vector<T> x2, const thrust::device_vector<T> x3,
  const thrust::device_vector<T> x4) {
  cuda::stream s{cuda::devices[0]};
  auto policy = thrust::cuda::par_nosync.on(s.get());

  thrust::copy(policy, x1.begin(), x1.end(), r.begin());

  cuda::zip_transform_iterator k2{cuda::std::plus<T>(), r.begin(), x2.begin()};
  cuda::zip_transform_iterator k3{cuda::std::plus<T>(), r.begin(), x3.begin()};
  cuda::zip_transform_iterator k4{cuda::std::plus<T>(), r.begin(), x4.begin()};

  const auto n = r.size();
  thrust::copy(policy, k2, k2 + n, r.begin());
  thrust::copy(policy, k3, k3 + n, r.begin());
  thrust::copy(policy, k4, k4 + n, r.begin());

  s.sync();
}

} // namespace ra::benchmark
