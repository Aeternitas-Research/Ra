#pragma once

#include "ra/error.cuh"
#include <string>
#include <thrust/memory.h>

namespace ra {

struct SnapshotConfig {
  std::string name{};
  struct {
    double start = 0.0;
    double stop  = 0.0;
    double now   = 0.0;
    double delta = 0.0;
  } time{};
  struct {
    bool initialized = false;
    int size         = 0;
    int rank         = 0;
  } mpi{};
  struct {
    int n_species = 2;
  } mesh{};
};

struct Snapshot {
  __host__ ~Snapshot();
  __host__ Snapshot();
  Snapshot(const Snapshot&)     = delete;
  Snapshot(Snapshot&&) noexcept = delete;
  __host__ explicit Snapshot(SnapshotConfig& config);
  Snapshot& operator=(const Snapshot&)     = delete;
  Snapshot& operator=(Snapshot&&) noexcept = delete;

  __host__ Error copy(const Snapshot& other);
  __host__ Error calibrate();

  thrust::pointer<SnapshotConfig, thrust::host_system_tag> config{};
};

} // namespace ra
