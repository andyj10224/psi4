include(TestingMacros)

if(ENABLE_Einsums)
    add_regression_test(dlpnobcc-2 "psi;dlpno;cc")
endif()
