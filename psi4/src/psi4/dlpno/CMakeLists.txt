list(APPEND sources
  dlpno.cc
  orbitals.cc
  mp2.cc
  ccsd.cc
  ccsd_lambda.cc
  ccsd_gradient_lambda.cc
  ccsd_pair_response.cc
  ccsd_df_response.cc
  pno_truncation_response.cc
  screened_pair_response.cc
  lmp2_response.cc
  response_kernels.cc
  local_basis_response.cc
  occupied_projection.cc
  triples.cc
  triples_response.cc
  triples_df_response.cc
  ro_triples.cc
  wrapper.cc
  sparse.cc
  )

if(TARGET Einsums::Einsums)
    list(APPEND sources quads.cc)
endif()

psi4_add_module(bin dlpno sources)
if (psi4_ENABLE_PRECOMPILE_HEADERS)
    target_precompile_headers(dlpno REUSE_FROM psio)
endif()

target_link_libraries(dlpno PUBLIC diis)
