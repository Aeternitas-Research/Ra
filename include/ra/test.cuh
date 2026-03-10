#pragma once

#include "ra/mpi.cuh"
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>

#define RA_TEST_MAIN(argc, argv)                                              \
  int main(int argc, char* argv[]) {                                          \
    ra_mpi_invoke(MPI_Init(&argc, &argv));                                    \
                                                                              \
    int mpi_rank{};                                                           \
    ra_mpi_invoke(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));                  \
    if (mpi_rank) {                                                           \
      std::cout.rdbuf(nullptr);                                               \
    }                                                                         \
                                                                              \
    int mpi_rank_local{};                                                     \
    {                                                                         \
      MPI_Comm communicator_local{};                                          \
      MPI_Comm_split_type(                                                    \
        MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL,        \
        &communicator_local);                                                 \
      MPI_Comm_rank(communicator_local, &mpi_rank_local);                     \
      MPI_Comm_free(&communicator_local);                                     \
    }                                                                         \
    int n_device = 0;                                                         \
    cudaGetDeviceCount(&n_device);                                            \
    cudaSetDevice(mpi_rank_local % n_device);                                 \
                                                                              \
    const auto result = Catch::Session().run(argc, argv);                     \
                                                                              \
    ra_mpi_invoke(MPI_Finalize());                                            \
                                                                              \
    return result;                                                            \
  }
