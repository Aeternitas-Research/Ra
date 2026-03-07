#include "ra/timestepper.cuh"

namespace ra {

__host__ Error
TimeStepperExplicitRK1D::try_step(bool& success, double& epsilon) {
  if (success) {
    return cudaSuccess;
  }

  auto& rhs        = this->config.op.rhs;
  auto& time       = this->config.time;
  double h         = time.delta;
  const auto space = OperationSpace::Device;

  // compute k
  ra_invoke(rhs(k[0], time.now, mesh));
  for (int stage = 1; stage < config_extra.stage; ++stage) {
    const double* a = config_extra.a[stage];
    const double c  = config_extra.c[stage];

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
  const auto n_step     = history_h.size();
  int flag_modify_h     = 0;
  if (epsilon < 0.5) {
    flag_modify_h = +1;
    if (n_step == 0) {
      flag_modify_h = 0;
    }
  } else if (epsilon > 1.1) {
    flag_modify_h = -1;
  } else {
    flag_modify_h = 0;
  }

  // good epsilon
  if (flag_modify_h == 0) {
    success = true;

    // update mesh
    mesh.assign(space, backup);
    for (int stage = 0; stage < config_extra.stage; ++stage) {
      const double b = config_extra.b[stage];

      mesh.add(space, h * b, k[stage]);
    }

    return cudaSuccess;
  }

  time.n_fail += 1;

  // compute h for the next attempt
  const auto& history_e  = time.history_error;
  const auto& adaptivity = this->config.parameter.adaptivity.time;
  double order = static_cast<double>(this->config.parameter.order.time);
  if (flag_modify_h == +1) {
    if (n_step >= 3) {
      double h1 = history_h[n_step - 1];
      double h2 = history_h[n_step - 2];
      double h3 = history_h[n_step - 3];
      double e1 = history_e[n_step - 1];
      double e2 = history_e[n_step - 2];
      double e3 = history_e[n_step - 3];
      h = h1 * std::pow(e1, -adaptivity.k1 / order) *
          std::pow(e2, -adaptivity.k2 / order) *
          std::pow(e3, -adaptivity.k3 / order) *
          std::pow(h1 / h2, adaptivity.k4) * std::pow(h2 / h3, adaptivity.k5);
    } else if (n_step == 2) {
      double h1 = history_h[n_step - 1];
      double h2 = history_h[n_step - 2];
      double e1 = history_e[n_step - 1];
      double e2 = history_e[n_step - 2];
      h         = h1 * std::pow(e1, -adaptivity.k1 / order) *
                  std::pow(e2, -adaptivity.k2 / order) *
                  std::pow(h1 / h2, adaptivity.k4);
    } else if (n_step == 1) {
      double h1 = history_h[n_step - 1];
      double e1 = history_e[n_step - 1];
      h         = h1 * std::pow(e1, -adaptivity.k1 / order);
    } else {
      cuda::std::terminate();
    }

    success = false;
  } else if (flag_modify_h == -1) {
    h *= 0.8;

    success = false;
  } else {
    cuda::std::terminate();
  }

  time.delta = h;

  return cudaSuccess;
}

} // namespace ra
