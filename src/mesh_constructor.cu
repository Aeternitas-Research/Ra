#include "ra/mesh.cuh"

namespace ra {

Mesh::~Mesh() {
  if (config.buffer.in) {
    delete[] config.buffer.in;
  }
  if (config.buffer.out) {
    delete[] config.buffer.out;
  }
}

Mesh::Mesh() {}

Mesh::Mesh(const MeshConfig& config) : config(config) {}

} // namespace ra
