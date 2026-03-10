#include "ra/mesh.cuh"
#include "ra/mpi.cuh"
#include "ra/netcdf.cuh"
#include <filesystem>
#include <iomanip>
#include <mpi.h>
#include <netcdf.h>
#include <sstream>

namespace ra {

__host__ Error
Mesh1D::write(const int mpi_rank) {
  config.info.mpi_rank = mpi_rank;

  std::filesystem::create_directory(config.file.directory);
  std::stringstream buffer;
  buffer << config.file.directory << config.name << "." << config.file.handle
         << "." << std::setw(4) << std::setfill('0') << config.info.mpi_rank
         << ".nc";
  config.file.name = buffer.str();

  auto& file = config.netcdf.id.file;
  ra_netcdf_invoke(
    nc_create(config.file.name.c_str(), NC_NETCDF4 | NC_CLOBBER, &file));

  auto& variable = config.netcdf.id.variable;
  ra_netcdf_invoke(
    nc_def_var(file, "mpi_rank", NC_INT, 0, nullptr, &(variable.mpi_rank)));
  ra_netcdf_invoke(
    nc_def_var(file, "step", NC_UINT64, 0, nullptr, &(variable.step)));
  ra_netcdf_invoke(
    nc_def_var(file, "time", NC_DOUBLE, 0, nullptr, &(variable.time)));

  auto& info = config.info;
  ra_netcdf_invoke(nc_put_var(file, variable.mpi_rank, &(info.mpi_rank)));
  ra_netcdf_invoke(nc_put_var(file, variable.step, &(info.step)));
  ra_netcdf_invoke(nc_put_var(file, variable.time, &(info.time)));

  auto& dimension = config.netcdf.id.dimension;
  auto& geometry = config.geometry;
  ra_netcdf_invoke(
    nc_def_dim(file, "extent.0", geometry.extent[0], &(dimension.extent[0])));

  const std::size_t length_x = 2 * geometry.extent[0];
  ra_netcdf_invoke(nc_def_dim(file, "x", length_x, &(dimension.x[0])));

  const std::size_t length_f = geometry.element.dof * geometry.extent[0];
  ra_netcdf_invoke(nc_def_dim(file, "f", length_f, &(dimension.f[0])));

  auto& name = config.netcdf.name;
  ra_netcdf_invoke(nc_def_var(
    file, name.variable.x.c_str(), NC_DOUBLE, 1, dimension.x, &(variable.x)));
  ra_netcdf_invoke(nc_def_var(
    file, name.variable.f.c_str(), NC_DOUBLE, 1, dimension.f, &(variable.f)));

#ifdef RA_DEBUG
  if (host.x.size() != length_x) {
    return cudaErrorInvalidValue;
  }
#endif
  ra_netcdf_invoke(nc_put_var(file, variable.x, host.x.data()));
#ifdef RA_DEBUG
  if (host.f.size() != length_f) {
    return cudaErrorInvalidValue;
  }
#endif
  ra_netcdf_invoke(nc_put_var(file, variable.f, host.f.data()));

  ra_netcdf_invoke(nc_close(file));

  return cudaSuccess;
}

} // namespace ra
