from psi4.driver.procrouting import proc
from psi4.driver.procrouting.proc_table import procedures
from psi4.driver.driver_cbs import VARH


def test_dlpno_energy_routing():
    assert procedures["energy"]["dlpno-ccsd"] is proc.run_dlpnoccsd
    assert procedures["energy"]["dlpno-bccd"] is proc.run_dlpnoccsd

    triples_names = {
        "dlpno-ccsd(t0)",
        "dlpno-ccsd(t)",
        "dlpno-ccsd(ct)",
        "dlpno-ccsd(t)_l",
        "dlpno-ccsd(at)",
        "dlpno-bccd(t)",
        "dlpno-bccd(ct)",
        "dlpno-bccd(t)_l",
        "dlpno-bccd(at)",
    }
    for name in triples_names:
        assert procedures["energy"][name] is proc.run_dlpnoccsd_t

    assert "dlpno-ccsd_l" not in procedures["energy"]

    assert proc._DLPNO_TRIPLES_METHOD_SETTINGS == {
        "dlpno-ccsd(t0)": (False, False, True, False),
        "dlpno-ccsd(t)": (False, False, False, False),
        "dlpno-ccsd(ct)": (False, False, False, False),
        "dlpno-ccsd(t)_l": (False, True, False, False),
        "dlpno-ccsd(at)": (False, True, False, False),
        "dlpno-bccd(t)": (True, False, False, True),
        "dlpno-bccd(ct)": (True, False, False, True),
        "dlpno-bccd(t)_l": (True, True, False, False),
        "dlpno-bccd(at)": (True, True, False, False),
    }


def test_dlpno_property_routing():
    assert procedures["properties"]["dlpno-ccsd"] is proc.run_dlpnoccsd_property
    assert procedures["properties"]["dlpno-bccd"] is proc.run_dlpnoccsd_property


def test_dlpno_asymmetric_triples_cbs_aliases():
    assert VARH["dlpno-ccsd(at)"]["dlpno-ccsd(at)"] == "CCSD(AT) TOTAL ENERGY"
    assert VARH["dlpno-bccd(at)"]["dlpno-bccd(at)"] == "BCCD(AT) TOTAL ENERGY"


def test_dlpno_complete_triples_cbs_hierarchy():
    assert VARH["dlpno-ccsd(ct)"]["dlpno-ccsd(ct)"] == "CCSD(cT) TOTAL ENERGY"
    assert VARH["dlpno-bccd(ct)"]["dlpno-bccd(ct)"] == "BCCD(cT) TOTAL ENERGY"
