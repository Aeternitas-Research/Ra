#include "ra/snapshot.cuh"

namespace ra {

__host__ Snapshot::~Snapshot() {}

__host__
Snapshot::Snapshot() {}

__host__
Snapshot::Snapshot(SnapshotConfig& config)
    : config(&config) {}

} // namespace ra
