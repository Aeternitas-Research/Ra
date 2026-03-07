#include "ra/error.cuh"
#include <cuda/std/__exception/terminate.h>
#include <iostream>

namespace ra {

__host__ __device__ Error
invoke_impl(const Error result, const char* file, const int line) {
  if (result != cudaSuccess) {
#if !defined(__CUDA_ARCH__)
    std::cerr << "RA error " << result << " in " << file << ":" << line
              << std::endl;
#endif
    cuda::std::terminate();
  }

  return result;
}

} // namespace ra
