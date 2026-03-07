#pragma once

#ifdef RA_DEBUG
#define netcdf_invoke(_expr) \
  ra::netcdf_invoke_impl((_expr), __FILE__, __LINE__)
#else
#define netcdf_invoke(_expr) (_expr)
#endif

namespace ra {

__host__ int
netcdf_invoke_impl(const int result, const char* file, const int line);

} // namespace ra
