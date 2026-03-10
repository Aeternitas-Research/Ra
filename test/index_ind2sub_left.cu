#include "ra/index.cuh"
#include "ra/test.cuh"
#include <cuda/std/random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

RA_TEST_MAIN(argc, argv);

constexpr int N = 262144;

__host__ void
test_ind2sub_left_host(
  thrust::host_vector<int>& output, thrust::host_vector<int>& input,
  thrust::host_vector<int>& extent, const int dimension) {
  for (int index = 0; index < N; ++index) {
    const auto offset = 6 * index;
    input[index] =
      ra::sub2ind_left(output.data() + offset, extent.data(), dimension);
  }
  thrust::fill(output.begin(), output.end(), 0);
  for (int index = 0; index < N; ++index) {
    const auto offset = 6 * index;
    const int input_value = input[index];
    ra::ind2sub_left(
      output.data() + offset, input_value, extent.data(), dimension);

    switch (dimension) {
    case 6:
      REQUIRE(output[offset + 5] == (input[index] / extent[4]) % extent[5]);
      [[fallthrough]];
    case 5:
      REQUIRE(output[offset + 4] == (input[index] / extent[3]) % extent[4]);
      [[fallthrough]];
    case 4:
      REQUIRE(output[offset + 3] == (input[index] / extent[2]) % extent[3]);
      [[fallthrough]];
    case 3:
      REQUIRE(output[offset + 2] == (input[index] / extent[1]) % extent[2]);
      [[fallthrough]];
    case 2:
      REQUIRE(output[offset + 1] == (input[index] / extent[0]) % extent[1]);
      [[fallthrough]];
    case 1:
      REQUIRE(output[offset + 0] == input[index] % extent[0]);
      break;
    default:
      REQUIRE(false);
    }
  }
}

template <typename T>
__global__ void
invoke_sub2ind_left_device(
  thrust::device_ptr<T> output, thrust::device_ptr<T> input,
  thrust::device_ptr<T> extent, const int dimension) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    output[index] =
      ra::sub2ind_left(input.get() + 6 * index, extent.get(), dimension);
  }
}

template <typename T>
__global__ void
invoke_ind2sub_left_device(
  thrust::device_ptr<T> output, thrust::device_ptr<T> input,
  thrust::device_ptr<T> extent, const int dimension) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    const int input_value = input[index];
    ra::ind2sub_left(
      output.get() + 6 * index, input_value, extent.get(), dimension);
  }
}

__host__ void
test_ind2sub_left_device(
  thrust::device_vector<int>& output, thrust::device_vector<int>& input,
  thrust::device_vector<int>& extent, const int dimension,
  const thrust::host_vector<int>& output_ref) {
  invoke_sub2ind_left_device<<<N / 256, 256>>>(
    input.data(), output.data(), extent.data(), dimension);
  thrust::fill(output.begin(), output.end(), 0);
  invoke_ind2sub_left_device<<<N / 256, 256>>>(
    output.data(), input.data(), extent.data(), dimension);
  for (int index = 0; index < 6 * N; ++index) {
    REQUIRE(output[index] == output_ref[index]);
  }
}

TEST_CASE("ind2sub_left", "[index]") {
  struct {
    thrust::host_vector<int> output;
    thrust::host_vector<int> input;
    thrust::host_vector<int> extent;
  } host{};
  struct {
    thrust::device_vector<int> output;
    thrust::device_vector<int> input;
    thrust::device_vector<int> extent;
  } device{};

  host.output.resize(6 * N);
  host.input.resize(N);
  host.extent.resize(6);
  device.output.resize(6 * N);
  device.input.resize(N);
  device.extent.resize(6);

  for (int d = 0; d < 6; ++d) {
    host.extent[d] = 16;
  }
  device.extent = host.extent;

  cuda::std::philox4x64 rng{};
  cuda::std::uniform_int_distribution<int> dist(0, 15);
  for (int dimension = 1; dimension <= 6; ++dimension) {
    for (int index = 0; index < N; ++index) {
      const auto offset = index * 6;
      for (int d = 0; d < 6; ++d) {
        host.output[offset + d] = dist(rng);
      }
    }
    device.output = host.output;

    test_ind2sub_left_host(host.output, host.input, host.extent, dimension);
    test_ind2sub_left_device(
      device.output, device.input, device.extent, dimension, host.output);
  }
}
