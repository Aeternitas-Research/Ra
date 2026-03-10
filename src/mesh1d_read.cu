#include "ra/mesh.cuh"
#include "ra/mpi.cuh"
#include "ra/netcdf.cuh"
#include <filesystem>
#include <iomanip>
#include <mpi.h>
#include <netcdf.h>
#include <sstream>

namespace ra {

Error
Mesh1D::read(const int mpi_rank) {
  config.info.mpi_rank = mpi_rank;

  std::stringstream buffer;
  buffer << config.file.directory << config.name << "." << config.file.handle
         << "." << std::setw(4) << std::setfill('0') << config.info.mpi_rank
         << ".nc";
  config.file.name = buffer.str();

  auto& file = config.netcdf.id.file;
  ra_netcdf_invoke(nc_open(config.file.name.c_str(), NC_NOWRITE, &file));

  auto& variable = config.netcdf.id.variable;
  ra_netcdf_invoke(nc_inq_varid(file, "mpi_rank", &(variable.mpi_rank)));
  ra_netcdf_invoke(nc_inq_varid(file, "step", &(variable.step)));
  ra_netcdf_invoke(nc_inq_varid(file, "time", &(variable.time)));

  auto& info = config.info;
  ra_netcdf_invoke(nc_get_var(file, variable.mpi_rank, &(info.mpi_rank)));
  ra_netcdf_invoke(nc_get_var(file, variable.step, &(info.step)));
  ra_netcdf_invoke(nc_get_var(file, variable.time, &(info.time)));

  auto& dimension = config.netcdf.id.dimension;
  auto& geometry = config.geometry;
  ra_netcdf_invoke(nc_inq_dimid(file, "extent.0", &(dimension.extent[0])));
  ra_netcdf_invoke(nc_inq_dimid(file, "x", &(dimension.x[0])));
  ra_netcdf_invoke(nc_inq_dimid(file, "f", &(dimension.f[0])));
  ra_netcdf_invoke(
    nc_inq_dimlen(file, dimension.extent[0], &(geometry.extent[0])));

  auto& name = config.netcdf.name;
  ra_netcdf_invoke(nc_inq_varid(file, name.variable.x.c_str(), &(variable.x)));
  ra_netcdf_invoke(nc_inq_varid(file, name.variable.f.c_str(), &(variable.f)));

  const std::size_t length_x = 2 * geometry.extent[0];
  if (host.x.size() != length_x) {
    host.x.resize(length_x);
  }
  ra_netcdf_invoke(nc_get_var(file, variable.x, host.x.data()));

  const std::size_t length_f = geometry.element.dof * geometry.extent[0];
  if (host.f.size() != length_f) {
    host.f.resize(length_f);
  }
  ra_netcdf_invoke(nc_get_var(file, variable.f, host.f.data()));

  ra_netcdf_invoke(nc_close(file));

  return cudaSuccess;
}

} // namespace ra
