#pragma once

#include "ra/error.cuh"
#include <string>
#include <thrust/memory.h>

namespace ra {

struct SnapshotConfig {
  std::string name{};
  struct {
    double start = 0.0;
    double stop = 0.0;
    double now = 0.0;
    double delta = 0.0;
  } time{};
  struct {
    bool initialized = false;
    int size = 0;
    int rank = 0;
  } mpi{};
  struct {
    int n_species = 2;
  } mesh{};
};

struct Snapshot {
  ~Snapshot();
  Snapshot();
  Snapshot(const Snapshot&) = delete;
  Snapshot(Snapshot&&) noexcept = delete;
  explicit Snapshot(SnapshotConfig& config);
  Snapshot& operator=(const Snapshot&) = delete;
  Snapshot& operator=(Snapshot&&) noexcept = delete;

  Error copy(const Snapshot& other);
  Error calibrate();

  thrust::pointer<SnapshotConfig, thrust::host_system_tag> config{};
};

} // namespace ra
