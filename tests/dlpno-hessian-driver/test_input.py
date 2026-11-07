from addons import *


@ctest_labeler("dlpno;cc;hessian;d2ints;quick")
def test_dlpno_hessian_driver():
    ctest_runner(__file__)
