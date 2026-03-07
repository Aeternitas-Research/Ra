list(APPEND MPIEXEC_PREFLAGS --mca shmem_mmap_enable_nfs_warning 0)

function(ra_add_benchmark)
  set(arg_one_value "NAME")
  set(arg_multi_value "LABELS")
  cmake_parse_arguments(
    PARSE_ARGV 0
    RA_BENCHMARK_ARG
    ""
    "${arg_one_value}"
    "${arg_multi_value}"
  )

  set(_name_exe benchmark_${RA_BENCHMARK_ARG_NAME})
  set(_name_benchmark benchmark/${RA_BENCHMARK_ARG_NAME})
  add_executable(${_name_exe} ${RA_BENCHMARK_ARG_NAME}.cu)
  target_link_libraries(${_name_exe} PRIVATE ra Catch2::Catch2)
  add_test(
    NAME ${_name_benchmark}
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${RA_BENCHMARK_MPI_SIZE}
      ${MPIEXEC_PREFLAGS}
      ./${_name_exe}
        --benchmark-samples ${RA_BENCHMARK_SAMPLE_SIZE}
        --colour-mode=none
        [!benchmark]
      ${MPIEXEC_POSTFLAGS}
  )
  set(_benchmark_labels "")
  list(APPEND _benchmark_labels benchmark)
  if(RA_BENCHMARK_ARG_LABELS)
    list(APPEND _benchmark_labels "${RA_BENCHMARK_ARG_LABELS}")
  endif()
  set_tests_properties(${_name_benchmark} PROPERTIES LABELS "${_benchmark_labels}")
endfunction()

# cub
ra_add_benchmark(NAME block_reduce LABELS cub)

# stream
ra_add_benchmark(NAME stream_sum LABELS stream)

# mesh
ra_add_benchmark(NAME mesh1d_add LABELS mesh)
