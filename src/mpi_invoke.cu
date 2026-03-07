#include "ra/mpi.cuh"
#include <iostream>
#include <mpi.h>

namespace ra {

__host__ int
mpi_invoke_impl(const int result, const char* file, const int line) {
  if (result != MPI_SUCCESS) {
    std::cerr << "MPI error " << result << " in " << file << ":" << line
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, result);
  }

  return result;
}

} // namespace ra
