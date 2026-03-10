#include "ra/netcdf.cuh"
#include <iostream>
#include <mpi.h>
#include <netcdf.h>

namespace ra {

int
netcdf_invoke_impl(const int result, const char* file, const int line) {
  if (result != NC_NOERR) {
    std::cerr << "NetCDF error " << result << " in " << file << ":" << line
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, result);
  }

  return result;
}

} // namespace ra
