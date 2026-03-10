#include "ra/index.cuh"
#include "ra/test.cuh"
#include <cuda/std/random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

RA_TEST_MAIN(argc, argv);

constexpr int N = 262144;

void
test_sub2ind_left_host(
  thrust::host_vector<int>& output, thrust::host_vector<int>& input,
  thrust::host_vector<int>& extent, const int dimension) {
  for (int index = 0; index < N; ++index) {
    const auto offset = 6 * index;
    output[index] =
      ra::sub2ind_left(input.data() + offset, extent.data(), dimension);
    if (dimension == 1) {
      REQUIRE(output[index] == input[offset + 0]);
    } else if (dimension == 2) {
      REQUIRE(
        output[index] == (input[offset + 0] + extent[0] * input[offset + 1]));
    } else if (dimension == 3) {
      REQUIRE(
        output[index] ==
        (input[offset + 0] +
         extent[0] * (input[offset + 1] + extent[1] * input[offset + 2])));
    } else if (dimension == 4) {
      REQUIRE(
        output[index] ==
        (input[offset + 0] +
         extent[0] *
           (input[offset + 1] +
            extent[1] * (input[offset + 2] + extent[2] * input[offset + 3]))));
    } else if (dimension == 5) {
      REQUIRE(
        output[index] ==
        (input[offset + 0] +
         extent[0] *
           (input[offset + 1] +
            extent[1] * (input[offset + 2] +
                         extent[2] * (input[offset + 3] +
                                      extent[3] * input[offset + 4])))));
    } else if (dimension == 6) {
      REQUIRE(
        output[index] ==
        (input[offset + 0] +
         extent[0] *
           (input[offset + 1] +
            extent[1] *
              (input[offset + 2] +
               extent[2] * (input[offset + 3] +
                            extent[3] * (input[offset + 4] +
                                         extent[4] * input[offset + 5]))))));
    } else {
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

void
test_sub2ind_left_device(
  thrust::device_vector<int>& output, thrust::device_vector<int>& input,
  thrust::device_vector<int>& extent, const int dimension,
  const thrust::host_vector<int>& output_ref) {
  invoke_sub2ind_left_device<<<N / 256, 256>>>(
    output.data(), input.data(), extent.data(), dimension);
  for (int index = 0; index < N; ++index) {
    REQUIRE(output[index] == output_ref[index]);
  }
}

TEST_CASE("sub2ind_left", "[index]") {
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

  host.output.resize(N);
  host.input.resize(6 * N);
  host.extent.resize(6);
  device.output.resize(N);
  device.input.resize(6 * N);
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
        host.input[offset + d] = dist(rng);
      }
    }
    device.input = host.input;

    test_sub2ind_left_host(host.output, host.input, host.extent, dimension);
    test_sub2ind_left_device(
      device.output, device.input, device.extent, dimension, host.output);
  }
}
