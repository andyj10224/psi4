from addons import *


@ctest_labeler("dlpno;cc;gradient;quick")
def test_dlpno_grad_model():
    ctest_runner(__file__)
