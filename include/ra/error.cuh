#pragma once

#include "ra/utility.cuh"
#include <string>

#if !defined(__CUDA_ARCH__)
#define RA_ERROR(_value) ra::Error((_value), ErrorCategory::Host)
#else
#define RA_ERROR(_value) ra::Error((_value), ErrorCategory::Device)
#endif

#ifdef RA_DEBUG
#define ra_invoke(_expr) ra::invoke_impl((_expr), __FILE__, __LINE__)
#else
#define ra_invoke(_expr) (_expr)
#endif

namespace ra {

enum struct ErrorCategory : int {
  Unknown = 0,
  Host,
  Device,
};

enum struct ErrorValue : int {
  Success = 0,
  InvalidParameter,
  InvalidSize,
  InvalidOption,
  InvalidGeometry,
};

struct Error {
  __host__ __device__ ~Error();
  __host__ __device__ Error();
  __host__ __device__ Error(const Error& other);
  __host__ __device__ Error(Error&& other) noexcept;
  __host__ __device__ Error(const int value, const ErrorCategory& category);
  __host__ __device__ Error(const cudaError& value);
  __host__ __device__ Error& operator=(const Error& other);
  __host__ __device__ Error& operator=(Error&& other) noexcept;

  template <typename T>
  __host__ __device__
  Error(const T& value, const ErrorCategory& category)
      : Error(ra::to_underlying(value), category) {}

  inline __host__ __device__ bool
  operator==(const Error& other) const {
    return (value == other.value) && (category == other.category);
  }

  inline __host__ __device__ bool
  operator==(const cudaError& other) const {
    return value == other;
  }

  template <typename T>
  __host__ __device__ bool
  operator!=(const T& other) const {
    return !(*this == other);
  }

  void get_message(std::string& output) const;

  int value = 0;
  ErrorCategory category = ErrorCategory::Unknown;
};

__host__ __device__ Error
invoke_impl(const Error result, const char* file, const int line);

} // namespace ra
