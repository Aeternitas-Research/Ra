#pragma once

#include <climits>
#include <cstdint>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "Unknown SIZE_MAX"
#endif

#ifdef RA_DEBUG
#define mpi_invoke(_expr) ra::mpi_invoke_impl((_expr), __FILE__, __LINE__)
#else
#define mpi_invoke(_expr) (_expr)
#endif

namespace ra {

__host__ int
mpi_invoke_impl(const int result, const char* file, const int line);

} // namespace ra
