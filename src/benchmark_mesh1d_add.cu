#include "ra/benchmark.cuh"
#include "ra/error.cuh"

namespace ra::benchmark {

void
mesh1d_add(const ra::OperationSpace space, Mesh1D& y, Mesh1D& c, Mesh1D& x) {
  ra_invoke(y.add(space, c, x));
}

} // namespace ra::benchmark
