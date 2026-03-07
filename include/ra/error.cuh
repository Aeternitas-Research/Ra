#pragma once

#ifdef RA_DEBUG
#define ra_invoke(_expr) ra::invoke_impl((_expr), __FILE__, __LINE__)
#else
#define ra_invoke(_expr) (_expr)
#endif

namespace ra {

using Error = cudaError_t;

__host__ __device__ Error
invoke_impl(const Error result, const char* file, const int line);

} // namespace ra
