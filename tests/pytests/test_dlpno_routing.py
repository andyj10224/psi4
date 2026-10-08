import pytest

from psi4.driver.procrouting import proc
from psi4.driver.procrouting.proc_table import procedures
from psi4.driver.procrouting.proc_data import method_governing_type_keywords
from psi4.driver.driver_cbs import VARH
from psi4.driver.p4util.exceptions import ValidationError


def test_dlpno_energy_routing():
    assert procedures["energy"]["dlpno-ccsd"] is proc.run_dlpno
    assert procedures["energy"]["dlpno-bccd"] is proc.run_dlpno

    triples_names = {
        "dlpno-ccsd(t0)",
        "dlpno-ccsd(t)",
        "dlpno-ccsd(t)_l",
        "dlpno-ccsd(at)",
        "dlpno-bccd(t)",
        "dlpno-bccd(t)_l",
        "dlpno-bccd(at)",
        "dlpno-ccsdt",
        "dlpno-ccsdt(q0)",
        "dlpno-ccsdt(q)",
        "dlpno-ccsdtq",
        "dlpno-bccdt",
        "dlpno-bccdt(q0)",
        "dlpno-bccdt(q)",
        "dlpno-bccdtq",
    }
    for name in triples_names:
        assert procedures["energy"][name] is proc.run_dlpno
        assert method_governing_type_keywords[name] == "cc_type"

    assert "dlpno-ccsd_l" not in procedures["energy"]



def test_dlpno_property_routing():
    assert procedures["properties"]["dlpno-ccsd"] is proc.run_dlpnoccsd_property
    assert procedures["properties"]["dlpno-bccd"] is proc.run_dlpnoccsd_property


def test_dlpno_asymmetric_triples_cbs_aliases():
    assert VARH["dlpno-ccsd(at)"]["dlpno-ccsd(at)"] == "CCSD(AT) TOTAL ENERGY"
    assert VARH["dlpno-bccd(at)"]["dlpno-bccd(at)"] == "BCCD(AT) TOTAL ENERGY"


@pytest.mark.parametrize("name, algorithm, orbitals", [
    ("dlpno-bccd", "CCSD", "BCCD"),
    ("dlpno-bccd(t)", "CCSD(T)", "BCCD"),
    ("dlpno-bccd(t)_l", "CCSD(T)", "BCCD"),
    ("dlpno-bccdt", "CCSDT", "BCCDT"),
    ("dlpno-bccdt(q0)", "CCSDT(Q)", "BCCDT"),
    ("dlpno-bccdt(q)", "CCSDT(Q)", "BCCDT"),
    ("dlpno-bccdtq", "CCSDTQ", "BCCDTQ"),
])
def test_dlpno_brueckner_aliases(name, algorithm, orbitals):
    method = proc._dlpno_method_settings(name)
    assert method["algorithm"] == algorithm
    assert method["reference_orbitals"] == orbitals
    assert method["brueckner"]


@pytest.mark.parametrize("orbitals", ["HF", "BCCD", "BCCDT", "BCCDTQ"])
def test_dlpno_mixed_reference(orbitals):
    method = proc._dlpno_method_settings("dlpno-ccsdtq", orbitals)
    assert method["reference_orbitals"] == orbitals
    proc._validate_dlpno_reference(method, "RHF")


@pytest.mark.parametrize("name, orbitals", [
    ("dlpno-mp2", "BCCD"),
    ("dlpno-ccsd(t)", "BCCDT"),
    ("dlpno-ccsd(t)", "BCCDTQ"),
    ("dlpno-ccsdt(q)", "BCCDTQ"),
    ("dlpno-bccd(t)", "BCCDT"),
    ("dlpno-bccdtq", "BCCD"),
    ("dlpno-ccsd", "UNKNOWN"),
])
def test_dlpno_invalid_orbital_rank(name, orbitals):
    with pytest.raises(ValidationError):
        proc._dlpno_method_settings(name, orbitals)


@pytest.mark.parametrize("name, orbitals, reference, valid", [
    ("dlpno-bccd", "HF", "ROHF", True),
    ("dlpno-bccd(t)", "HF", "ROHF", True),
    ("dlpno-ccsd(t)", "HF", "ROHF", True),
    ("dlpno-ccsdtq", "BCCDT", "ROHF", False),
    ("dlpno-bccdt", "HF", "ROHF", False),
    ("dlpno-ccsd(t)_l", "HF", "ROHF", False),
    ("dlpno-ccsd", "HF", "UHF", True),
    ("dlpno-bccd", "HF", "UHF", False),
    ("dlpno-ccsd(t)", "HF", "UHF", False),
])
def test_dlpno_reference_compatibility(name, orbitals, reference, valid):
    method = proc._dlpno_method_settings(name, orbitals)
    if valid:
        proc._validate_dlpno_reference(method, reference, method.get("use_lambda", False))
    else:
        with pytest.raises(ValidationError):
            proc._validate_dlpno_reference(method, reference, method.get("use_lambda", False))


def test_dlpno_legacy_brueckner_switch():
    assert proc._dlpno_method_settings("dlpno-ccsd", legacy_brueckner=True)["reference_orbitals"] == "BCCD"
    assert proc._dlpno_method_settings("dlpno-ccsdtq", "BCCDT", True)["reference_orbitals"] == "BCCDT"


@pytest.mark.parametrize("name", ["bccdt", "bccdt(q0)", "bccdt(q)", "bccdtq"])
def test_dlpno_higher_brueckner_cbs(name):
    method = "dlpno-" + name
    assert VARH[method][method] == name.upper() + " TOTAL ENERGY"
    # A lower-rank solve in these orbitals is not the independently optimized
    # lower-rank method and must not be harvested as such by CBS.
    assert "dlpno-bccd" not in VARH[method]


@pytest.mark.parametrize("name", ["bccd", "bccd(t)", "bccd(t)_l", "bccd(at)"])
def test_dlpno_brueckner_cbs_does_not_harvest_rotated_mp2(name):
    assert "dlpno-mp2" not in VARH["dlpno-" + name]
