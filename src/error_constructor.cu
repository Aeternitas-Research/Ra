#include "ra/error.cuh"

namespace ra {

Error::~Error() {}

Error::Error() {}

Error::Error(const Error& other)
    : value{other.value}, category{other.category} {}

Error::Error(Error&&) noexcept
    : value{other.value}, category{other.category} {}

Error::Error(const int value, const ErrorCategory& category)
    : value{value}, category{category} {}

Error::Error(const cudaError& value) : value{value} {}

Error&
Error::operator=(const Error& other) {
  value = other.value;
  category = other.category;

  return *this;
}

Error&
Error::operator=(Error&&) noexcept {
  value = other.value;
  category = other.category;

  return *this;
}

Error::
operator bool() const {
  return (value != 0);
}

} // namespace ra
