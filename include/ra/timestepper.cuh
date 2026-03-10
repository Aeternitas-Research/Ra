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
      int time = 1;
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
    double now = 0.0;
    double initial = 0.0;
    double stop = 0.0;
    double delta = 0.0;
    double delta_min = 0.0;
    double delta_max = 0.0;
    double delta_good = 0.0;
    thrust::host_vector<double> history_delta{};
    thrust::host_vector<double> history_error{};
    int n_fail = 0;
  } time{};
  struct {
    double h[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
  ~TimeStepper();
  TimeStepper();
  TimeStepper(const TimeStepper&) = delete;
  TimeStepper(TimeStepper&&) noexcept = delete;
  TimeStepper(const TimeStepperConfig& config);
  TimeStepper& operator=(const TimeStepper&) = delete;
  TimeStepper& operator=(TimeStepper&&) noexcept = delete;

  virtual Error calibrate() = 0;
  virtual Error step() = 0;

  virtual Error find_epsilon(double& epsilon) = 0;
  virtual Error try_step(bool& success, double& epsilon) = 0;
  virtual Error reset_mesh() = 0;

  TimeStepperConfig config{};
};

struct TimeStepperExplicitRK1D : TimeStepper {
  ~TimeStepperExplicitRK1D();
  TimeStepperExplicitRK1D();
  TimeStepperExplicitRK1D(const TimeStepperConfig& config);

  Error calibrate() override;
  Error step() override;

  Error find_epsilon(double& epsilon) override;
  Error try_step(bool& success, double& epsilon) override;
  Error reset_mesh() override;

  PMesh1D mesh{};
  PMesh1D backup{};
  PMesh1D buffer{};

  PMesh1D error{};
  PMesh1D k[9]{};
};

} // namespace ra
