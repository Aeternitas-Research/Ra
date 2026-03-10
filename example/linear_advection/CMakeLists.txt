# 1D linear advection
set(_name sim.LinearAdvection.1D)
add_executable(${_name} main_linear_advection_1d.cu)
target_link_libraries(${name} PRIVATE ra)
