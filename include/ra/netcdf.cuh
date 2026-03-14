#pragma once

#ifdef RA_MODE_DEBUG
#define ra_netcdf_invoke(_expr)                                               \
  ra::netcdf_invoke_impl((_expr), __FILE__, __LINE__)
#else
#define ra_netcdf_invoke(_expr) (_expr)
#endif

namespace ra {

int netcdf_invoke_impl(const int result, const char* file, const int line);

} // namespace ra
