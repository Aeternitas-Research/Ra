list(APPEND MPIEXEC_PREFLAGS --mca shmem_mmap_enable_nfs_warning 0)

function(ra_add_test)
  set(arg_one_value "MODULE" "NAME")
  set(arg_multi_value "LABELS")
  cmake_parse_arguments(
    PARSE_ARGV 0
    RA_TEST_ARG
    ""
    "${arg_one_value}"
    "${arg_multi_value}"
  )

  set(_name_exe test_${RA_TEST_ARG_MODULE}_${RA_TEST_ARG_NAME})
  set(_name_test test/${RA_TEST_ARG_MODULE}/${RA_TEST_ARG_NAME})
  add_executable(${_name_exe} ${RA_TEST_ARG_MODULE}_${RA_TEST_ARG_NAME}.cu)
  target_link_libraries(${_name_exe} PRIVATE ra Catch2::Catch2)
  add_test(
    NAME ${_name_test}
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} ${RA_TEST_MPI_SIZE}
      ${MPIEXEC_PREFLAGS}
      ./${_name_exe} --colour-mode=none
      ${MPIEXEC_POSTFLAGS}
  )
  set(_test_labels "")
  list(APPEND _test_labels test ${RA_TEST_ARG_MODULE})
  if(RA_TEST_ARG_LABELS)
    list(APPEND _test_labels "${RA_TEST_ARG_LABELS}")
  endif()
  set_tests_properties(${_name_test} PROPERTIES LABELS "${_test_labels}")
endfunction()

# index
ra_add_test(MODULE index NAME sub2ind_left)
ra_add_test(MODULE index NAME sub2ind_right)
ra_add_test(MODULE index NAME ind2sub_left)
ra_add_test(MODULE index NAME ind2sub_right)

# Snapshot
ra_add_test(MODULE snapshot NAME constructor)
ra_add_test(MODULE snapshot NAME copy)
ra_add_test(MODULE snapshot NAME calibrate)

# Mesh1D
ra_add_test(MODULE mesh1d NAME constructor LABELS mesh)
ra_add_test(MODULE mesh1d NAME copy LABELS mesh)
ra_add_test(MODULE mesh1d NAME transfer LABELS mesh)
ra_add_test(MODULE mesh1d NAME sync LABELS mesh)
ra_add_test(MODULE mesh1d NAME write LABELS mesh)
ra_add_test(MODULE mesh1d NAME read LABELS mesh)
ra_add_test(MODULE mesh1d NAME assign LABELS mesh)
ra_add_test(MODULE mesh1d NAME multiply LABELS mesh)
ra_add_test(MODULE mesh1d NAME add LABELS mesh)
ra_add_test(MODULE mesh1d NAME divide LABELS mesh)
ra_add_test(MODULE mesh1d NAME subtract LABELS mesh)
ra_add_test(MODULE mesh1d NAME norm LABELS mesh)

# PMesh1D
ra_add_test(MODULE pmesh1d NAME constructor LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME copy LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME calibrate LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME transfer LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME sync LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME write LABELS pmesh)
ra_add_test(MODULE pmesh1d NAME read LABELS pmesh)

# TimeStepperExplicitRK1D
ra_add_test(MODULE timestepperexplicitrk1d NAME constructor LABELS timestepper)
ra_add_test(MODULE timestepperexplicitrk1d NAME calibrate LABELS timestepper)
ra_add_test(MODULE timestepperexplicitrk1d NAME step LABELS timestepper)
ra_add_test(MODULE timestepperexplicitrk1d NAME find_epsilon LABELS timestepper)
ra_add_test(MODULE timestepperexplicitrk1d NAME try_step LABELS timestepper)
ra_add_test(MODULE timestepperexplicitrk1d NAME reset_mesh LABELS timestepper)
