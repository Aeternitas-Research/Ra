#include "ra/snapshot.cuh"

namespace ra {

Snapshot::~Snapshot() {}

Snapshot::Snapshot() {}

Snapshot::Snapshot(SnapshotConfig& config) : config(&config) {}

} // namespace ra
