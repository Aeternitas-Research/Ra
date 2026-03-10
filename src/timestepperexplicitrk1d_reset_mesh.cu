#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/timestepper.cuh"
#include <mpi.h>

namespace ra {

Error
TimeStepperExplicitRK1D::reset_mesh() {
  ra_invoke(mesh.copy(backup));

  return cudaSuccess;
}

} // namespace ra
