add_executable(ac_run_mpi main.cc host_memory.cc host_forcing.cc config_loader.cc)
target_link_libraries(ac_run_mpi astaroth_utils astaroth_core)
