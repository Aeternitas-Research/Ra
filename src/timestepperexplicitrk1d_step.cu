#include "ra/error.cuh"
#include "ra/mpi.cuh"
#include "ra/timestepper.cuh"
#include <mpi.h>

namespace ra {

__host__ Error
TimeStepperExplicitRK1D::step() {
  mesh.transfer(cudaMemcpyHostToDevice, true, true);

  // apply initial conditions
  auto& time    = this->config.time;
  auto& initial = this->config.op.initial;
  ra_invoke(initial(mesh, time.now, buffer));

  bool stop = false;
  while (!stop) {
    const auto space = OperationSpace::Device;

    time.n_fail = 0;
    ra_invoke(backup.assign(space, mesh));

    bool success   = false;
    double epsilon = 0.0;
    while (!success) {
      if (time.n_fail > 0) {
        ra_invoke(reset_mesh());
      }

      ra_invoke(try_step(success, epsilon));
    }

    time.history_delta.push_back(time.delta);
    time.history_error.push_back(epsilon);
    time.now += time.delta;

    if ((time.now + 1e-12) >= time.stop) {
      stop = true;
    }

    ra_mpi_invoke(MPI_Barrier(MPI_COMM_WORLD));
  }

  return cudaSuccess;
}

} // namespace ra
