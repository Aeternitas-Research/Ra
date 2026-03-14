#include "ra/mpi.cuh"
#include "ra/timestepper.cuh"
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <mpi.h>
#include <thrust/fill.h>
#include <thrust/tabulate.h>

namespace ra {

Error
TimeStepperExplicitRK1D::calibrate() {
  MeshConfig mesh_config = mesh.config.global;
  mesh_config.name = this->config.name;
  mesh_config.geometry.element.dof = this->config.parameter.order.space;

  auto x = this->config.space.x;
  auto h = this->config.space.h;
  for (int d = 0; d < 1; ++d) {
    mesh_config.geometry.extent[d] =
      static_cast<size_t>(cuda::std::ceil((x[d][1] - x[d][0]) / h[d]));
  }

  for (int d = 0; d < 1; ++d) {
    mesh_config.geometry.ghost_depth[d][0] = 1;
    mesh_config.geometry.ghost_depth[d][1] = 1;
  }

  int mpi_rank{};
  ra_mpi_invoke(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  int mpi_size{};
  ra_mpi_invoke(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  int mpi_extent[6] = {mpi_size, 0, 0, 0, 0, 0};
  PMesh1D result(mpi_rank, mpi_extent, mesh_config);
  ra_invoke(mesh.copy(result));
  ra_invoke(mesh.calibrate());

  auto ghost_depth = mesh.local.config.geometry.ghost_depth;
  std::size_t stride[6];
  for (int d = 0; d < 1; ++d) {
    stride[d] = mesh.local.config.geometry.extent[d] -
                (ghost_depth[d][0] + ghost_depth[d][1]);
  };

  ra_mpi_invoke(
    MPI_Scan(MPI_IN_PLACE, stride, 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD));

  auto start_0 = cuda::make_strided_iterator(mesh.local.host.x.begin(), 2);
  auto stop_0 = cuda::make_strided_iterator(mesh.local.host.x.end(), 2);
  thrust::tabulate(start_0, stop_0, [&](const size_t j0) {
    return x[0][0] + static_cast<double>(stride[0] + j0) * h[0] -
           static_cast<double>(ghost_depth[0][0]) * h[0];
  });
  auto start_1 = cuda::make_strided_iterator(mesh.local.host.x.begin() + 1, 2);
  thrust::fill_n(start_1, mesh.local.config.geometry.extent[0], h[0]);

  // set RK coefficients
  auto& order = this->config.parameter.order;
  auto& rk = this->config.parameter.table.rk_explicit;
  auto& a = rk.a;
  auto& b = rk.b;
  auto& b_star = rk.b_star;
  auto& c = rk.c;
  if (order.time == 4) {
    rk.stage = 7;

    a[1][0] = 247.0 / 1000.0;
    a[2][0] = 247.0 / 4000.0;
    a[2][1] = 2694949928731.0 / 7487940209513.0;
    a[3][0] = 464650059369.0 / 8764239774964.0;
    a[3][1] = 878889893998.0 / 2444806327765.0;
    a[3][2] = -952945855348.0 / 12294611323341.0;
    a[4][0] = 476636172619.0 / 8159180917465.0;
    a[4][1] = -1271469283451.0 / 7793814740893.0;
    a[4][2] = -859560642026.0 / 4356155882851.0;
    a[4][3] = 1723805262919.0 / 4571918432560.0;
    a[5][0] = 6338158500785.0 / 11769362343261.0;
    a[5][1] = -4970555480458.0 / 10924838743837.0;
    a[5][2] = 3326578051521.0 / 2647936831840.0;
    a[5][3] = -880713585975.0 / 1841400956686.0;
    a[5][4] = -1428733748635.0 / 8843423958496.0;
    a[6][0] = 760814592956.0 / 3276306540349.0;
    a[6][1] = 760814592956.0 / 3276306540349.0;
    a[6][2] = -47223648122716.0 / 6934462133451.0;
    a[6][3] = 71187472546993.0 / 9669769126921.0;
    a[6][4] = -13330509492149.0 / 9695768672337.0;
    a[6][5] = 11565764226357.0 / 8513123442827.0;

    b[2] = 9164257142617.0 / 17756377923965.0;
    b[3] = -10812980402763.0 / 74029279521829.0;
    b[4] = 1335994250573.0 / 5691609445217.0;
    b[5] = 2273837961795.0 / 8368240463276.0;
    b[6] = 247.0 / 2000.0;

    b_star[2] = 4469248916618.0 / 8635866897933.0;
    b_star[3] = -621260224600.0 / 4094290005349.0;
    b_star[4] = 696572312987.0 / 2942599194819.0;
    b_star[5] = 1532940081127.0 / 5565293938103.0;
    b_star[6] = 2441.0 / 20000.0;

    c[1] = 247.0 / 1000.0;
    c[2] = 4276536705230.0 / 10142255878289.0;
    c[3] = 67.0 / 200.0;
    c[4] = 3.0 / 40.0;
    c[5] = 7.0 / 10.0;
    c[6] = 1.0;
  } else {
    return RA_ERROR(ErrorValue::InvalidOption);
  }

  // initialize buffers
  backup.copy(mesh);
  error.copy(mesh);
  buffer.copy(mesh);
  for (int stage = 0; stage < rk.stage; ++stage) {
    k[stage].copy(mesh);
  }

  return cudaSuccess;
}

} // namespace ra
