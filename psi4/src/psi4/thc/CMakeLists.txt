list(APPEND sources
  decomposition.cc
  mp2.cc
)
psi4_add_module(bin thc sources)

target_link_libraries(thc PUBLIC diis)