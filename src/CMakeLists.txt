# error
target_sources(
  ra
  PRIVATE
  error_constructor.cu
  error_get_message.cu
  error_invoke.cu
)

# mpi
target_sources(
  ra
  PRIVATE
  mpi_invoke.cu
)

# netcdf
target_sources(
  ra
  PRIVATE
  netcdf_invoke.cu
)

# Snapshot
target_sources(
  ra
  PRIVATE
  snapshot_constructor.cu
  snapshot_copy.cu
  snapshot_calibrate.cu
)

# Mesh
target_sources(
  ra
  PRIVATE
  mesh_constructor.cu
  mesh_transfer.cu
  mesh_assign.cu
  mesh_multiply.cu
  mesh_add.cu
  mesh_divide.cu
  mesh_subtract.cu
  mesh_norm.cu
)

# Mesh1D
target_sources(
  ra
  PRIVATE
  mesh1d_constructor.cu
  mesh1d_copy.cu
  mesh1d_sync.cu
  mesh1d_write.cu
  mesh1d_read.cu
  mesh1d_assign.cu
  mesh1d_multiply.cu
  mesh1d_add.cu
  mesh1d_divide.cu
  mesh1d_subtract.cu
  mesh1d_get_host_coordinate.cu
  mesh1d_get_device_coordinate.cu
  mesh1d_get_host_stencil.cu
  mesh1d_get_device_stencil.cu
)

# Mesh2D
target_sources(
  ra
  PRIVATE
  mesh2d_constructor.cu
  mesh2d_copy.cu
  mesh2d_sync.cu
  mesh2d_write.cu
  mesh2d_read.cu
  mesh2d_assign.cu
  mesh2d_multiply.cu
  mesh2d_add.cu
  mesh2d_divide.cu
  mesh2d_subtract.cu
  mesh2d_get_host_coordinate.cu
)

# PMesh
target_sources(
  ra
  PRIVATE
  pmesh_constructor.cu
  pmesh_calibrate.cu
)

# PMesh1D
target_sources(
  ra
  PRIVATE
  pmesh1d_constructor.cu
  pmesh1d_copy.cu
  pmesh1d_calibrate.cu
  pmesh1d_transfer.cu
  pmesh1d_sync.cu
  pmesh1d_write.cu
  pmesh1d_read.cu
  pmesh1d_assign.cu
  pmesh1d_multiply.cu
  pmesh1d_add.cu
  pmesh1d_divide.cu
  pmesh1d_subtract.cu
  pmesh1d_norm.cu
  pmesh1d_get_host_coordinate.cu
  pmesh1d_get_device_coordinate.cu
  pmesh1d_get_host_stencil.cu
  pmesh1d_get_device_stencil.cu
)

# DG
target_sources(
  ra
  PRIVATE
  dg_projection.cu
  dg_linear.cu
)

# TimeStepper
target_sources(
  ra
  PRIVATE
  timestepper_constructor.cu
)

# TimeStepperExplicitRK1D
target_sources(
  ra
  PRIVATE
  timestepperexplicitrk1d_constructor.cu
  timestepperexplicitrk1d_calibrate.cu
  timestepperexplicitrk1d_step.cu
  timestepperexplicitrk1d_find_epsilon.cu
  timestepperexplicitrk1d_try_step.cu
  timestepperexplicitrk1d_reset_mesh.cu
)

# benchmark
target_sources(
  ra
  PRIVATE
  benchmark_mesh1d_add.cu
)
