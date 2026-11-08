
list(APPEND sources
  scf_products
  scf_response
)

foreach(py ${sources})
    execute_process(COMMAND ${Python_EXECUTABLE} -c
        "from Cython.Build import cythonize; cythonize('${CMAKE_CURRENT_SOURCE_DIR}/${py}.py', language_level=3)"
        RESULT_VARIABLE _exec_python_err)
    file(COPY ${py}.c DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(REMOVE ${py}.c)

    add_library(${py} SHARED "${py}.c")
    target_include_directories(${py} PRIVATE "${Python_INCLUDE_DIRS}")
    set_target_properties(${py} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"  # for python module
                                           OUTPUT_NAME ${py}
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")  # for python module
 
    install(TARGETS ${py}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/psi4/driver/procrouting/response)
endforeach()

# Keep the complete DLPNO response dependency graph in Cython-enabled installs.
# The regular Python installation copies this directory recursively.
install(FILES
        dlpno_ccsd_curvature.py
        dlpno_ccsd_mixed.py
        dlpno_ccsd_response.py
        dlpno_frame_curvature.py
        dlpno_hessian.py
        dlpno_hessian_inputs.py
        dlpno_localization_curvature.py
        dlpno_orbital.py
        dlpno_pno_curvature.py
        dlpno_rhf_curvature.py
        dlpno_screened_ccsd.py
        dlpno_screened_curvature.py
        dlpno_screened_hessian.py
        dlpno_native.py
        dlpno_screened_pno.py
        dlpno_screened_triples.py
        dlpno_triples_curvature.py
        dlpno_triples_response.py
        DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/psi4/driver/procrouting/response)
