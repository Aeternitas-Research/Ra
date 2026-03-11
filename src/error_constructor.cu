#include "ra/error.cuh"

namespace ra {

__host__ __device__ Error::~Error() {}

__host__ __device__
Error::Error() {}

__host__ __device__
Error::Error(const Error& other)
    : value{other.value}, category{other.category} {}

__host__ __device__
Error::Error(Error&& other) noexcept
    : value{other.value}, category{other.category} {}

__host__ __device__
Error::Error(const int value, const ErrorCategory& category)
    : value{value}, category{category} {}

__host__ __device__
Error::Error(const cudaError& value)
    : value{value} {}

__host__ __device__ Error&
Error::operator=(const Error& other) {
  value = other.value;
  category = other.category;

  return *this;
}

__host__ __device__ Error&
Error::operator=(Error&& other) noexcept {
  value = other.value;
  category = other.category;

  return *this;
}

} // namespace ra
