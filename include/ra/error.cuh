#pragma once

#include <string>

#ifdef RA_DEBUG
#define ra_invoke(_expr) ra::invoke_impl((_expr), __FILE__, __LINE__)
#else
#define ra_invoke(_expr) (_expr)
#endif

namespace ra {

enum struct ErrorCategory : int {
  Unknown = 0,
  Host = 1,
  Device = 2,
};

struct Error {
  ~Error();
  Error();
  Error(const Error& other);
  Error(Error&& other) noexcept;
  Error(const int value, const ErrorCategory& category);
  Error(const cudaError& value);
  Error& operator=(const Error& other);
  Error& operator=(Error&& other) noexcept;
  operator bool() const;

  void get_message(std::string& output) const;

  int value = 0;
  ErrorCategory category = ErrorCategory::Unknown;
};

__host__ __device__ Error
invoke_impl(const Error result, const char* file, const int line);

} // namespace ra
