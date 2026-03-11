#include "ra/error.cuh"
#include <cub/util_debug.cuh>
#include <cuda/std/__exception/terminate.h>

namespace ra {

__host__ __device__ Error
invoke_impl(const Error result, const char* file, const int line) {
  cub::Debug(result.value, file, line);
  if (result) {
    cuda::std::terminate();
  }

  return result;
}

} // namespace ra
