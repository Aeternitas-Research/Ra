#include "ra/timestepper.cuh"

namespace ra {

__host__ Error
TimeStepperExplicitRK1D::try_step(bool& success, double& epsilon) {
  if (success) {
    return cudaSuccess;
  }

  auto& time = this->config.time;
  if ((time.now + time.delta + 1e-12) >= time.stop) {
    time.delta = time.stop - time.now;
  }

  // apply boundary conditions
  auto& boundary = this->config.op.boundary;
  ra_invoke(boundary(mesh, time.now, buffer));

  auto& rhs = this->config.op.rhs;
  auto& rk = this->config.parameter.table.rk_explicit;
  double h = time.delta;
  const auto space = OperationSpace::Device;

  // compute k
  ra_invoke(rhs(k[0], time.now, mesh));
  for (int stage = 1; stage < rk.stage; ++stage) {
    const double* a = rk.a[stage];
    const double c = rk.c[stage];

    buffer.assign(space, backup);
    for (int index = 0; index < stage; ++index) {
      buffer.add(space, a[index] * h, k[index]);
    }

    ra_invoke(rhs(k[stage], time.now + c * h, buffer));
  }

  // compute epsilon
  ra_invoke(find_epsilon(epsilon));

  // check epsilon
  const auto& history_h = time.history_delta;
  const auto n_step = history_h.size();
  int flag_modify_h = 0;
  if (cuda::std::isnan(epsilon)) {
    flag_modify_h = -1;
  } else if (epsilon < 0.5) {
    flag_modify_h = +1;
  } else if (epsilon > 10.0) {
    flag_modify_h = -1;
  } else {
    flag_modify_h = 0;
  }
  if (flag_modify_h >= 0) {
    time.n_fail = 0;
    time.delta_good = h;
    success = true;

    // update mesh
    mesh.assign(space, backup);
    for (int stage = 0; stage < rk.stage; ++stage) {
      const double b = rk.b[stage];

      mesh.add(space, h * b, k[stage]);
    }
  } else {
    time.n_fail += 1;
    success = false;
  }

  if (flag_modify_h == 0) {
    return cudaSuccess;
  }

  // compute h for the next attempt
  const auto& history_e = time.history_error;
  const auto& adaptivity = this->config.parameter.adaptivity.time;
  double p = static_cast<double>(this->config.parameter.order.time) + 1.0;
  if (n_step >= 2) {
    double h1 = history_h[n_step - 1];
    double h2 = history_h[n_step - 2];
    double e1 = history_e[n_step - 1];
    double e2 = history_e[n_step - 2];
    h *= std::pow(epsilon, -adaptivity.k1 / p) *
         std::pow(e1, -adaptivity.k2 / p) * std::pow(e2, -adaptivity.k3 / p) *
         std::pow(h / h1, adaptivity.k4) * std::pow(h1 / h2, adaptivity.k5);
  } else if (n_step >= 1) {
    double h1 = history_h[n_step - 1];
    double e1 = history_e[n_step - 1];
    h *= std::pow(epsilon, -adaptivity.k1 / p) *
         std::pow(e1, -adaptivity.k2 / p) * std::pow(h / h1, adaptivity.k4);
  } else {
    h *= std::pow(epsilon, adaptivity.k1 / p);
  }

  time.delta = h;

  return cudaSuccess;
}

} // namespace ra
