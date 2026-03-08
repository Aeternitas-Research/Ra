#pragma once

#include "ra/error.cuh"
#include "ra/pmesh.cuh"
#include <nvfunctional>
#include <string>
#include <thrust/host_vector.h>

namespace ra {

enum struct TimeStepperType : int {
  RK = 0,
};

struct TimeStepperConfig {
  std::string name{};
  TimeStepperType type{};
  struct {
    struct {
      int time  = 1;
      int space = 1;
    } order{};
    struct {
      struct {
        int stage = 0;
        double a[9][9]{
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        };
        double b[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double b_star[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double c[9]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      } rk_explicit{};
    } table{};
    struct {
      struct {
        double absolute = 1e-9;
        double relative = 1e-6;
      } time{};
    } tolerance{};
    struct {
      struct {
        double k1 = +1.25;
        double k2 = +0.5;
        double k3 = -0.75;
        double k4 = +0.25;
        double k5 = +0.75;
      } time{};
    } adaptivity{};
  } parameter{};
  struct {
    double now       = 0.0;
    double initial   = 0.0;
    double stop      = 0.0;
    double delta     = 0.0;
    double delta_min = 0.0;
    double delta_max = 0.0;
    thrust::host_vector<double> history_delta{};
    thrust::host_vector<double> history_error{};
    int n_fail = 0;
  } time{};
  struct {
    double h[6]    = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double x[6][2] = {
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    };
  } space{};
  struct {
    nvstd::function<Error(PMesh1D&, PMesh1D&)> initial{};
    nvstd::function<Error(PMesh1D&, const double, PMesh1D&)> boundary{};
    nvstd::function<Error(PMesh1D&, const double, PMesh1D&)> rhs{};
  } op{};
};

struct TimeStepper {
  __host__ ~TimeStepper();
  __host__ TimeStepper();
  TimeStepper(const TimeStepper&)     = delete;
  TimeStepper(TimeStepper&&) noexcept = delete;
  __host__ TimeStepper(const TimeStepperConfig& config);
  TimeStepper& operator=(const TimeStepper&)     = delete;
  TimeStepper& operator=(TimeStepper&&) noexcept = delete;

  __host__ virtual Error calibrate() = 0;
  __host__ virtual Error step()      = 0;

  __host__ virtual Error find_epsilon(double& epsilon)            = 0;
  __host__ virtual Error try_step(bool& success, double& epsilon) = 0;
  __host__ virtual Error reset_mesh()                             = 0;

  TimeStepperConfig config{};
};

struct TimeStepperExplicitRK1D : TimeStepper {
  __host__ ~TimeStepperExplicitRK1D();
  __host__ TimeStepperExplicitRK1D();
  __host__ TimeStepperExplicitRK1D(const TimeStepperConfig& config);

  __host__ Error calibrate() override;
  __host__ Error step() override;

  __host__ Error find_epsilon(double& epsilon) override;
  __host__ Error try_step(bool& success, double& epsilon) override;
  __host__ Error reset_mesh() override;

  PMesh1D mesh{};
  PMesh1D backup{};
  PMesh1D buffer{};

  PMesh1D error{};
  PMesh1D k[9]{};
};

} // namespace ra
