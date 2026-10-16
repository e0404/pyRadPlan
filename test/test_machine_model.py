import pytest
from datetime import datetime
import numpy as np
from numpydantic import NDArray, Shape
import pymatreader
import scipy.io

from pyRadPlan.machines import (
    Machine,
    PhotonLINAC,
    IonAccelerator,
    find_machine_file,
    load_from_name,
    load_machine_from_mat,
    get_available_machines,
    validate_machine,
    ParticlePencilBeamKernel,
    PhotonSVDKernel,
)
from pyRadPlan.biology.tissues import LEMTissueProperties


def test_basic_machine_model():
    machine = Machine(
        radiation_mode="photons",
        description="A dummy machine",
        name="Generic",
        key="generic_key",
        created_on="2021-01-01T00:00:00",
        last_modified="2021-01-01T00:00:00",
        created_by="Jay Doe",
        last_modified_by="Jay Doe",
        data_type="-",
        version="1",
    )

    assert isinstance(machine.created_on, datetime)
    assert isinstance(machine.last_modified, datetime)

    assert machine.radiation_mode == "photons"
    assert machine.description == "A dummy machine"
    assert machine.name == "Generic"
    assert machine.created_on == datetime.strptime("2021-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    assert machine.last_modified == datetime.strptime("2021-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    assert machine.created_by == "Jay Doe"
    assert machine.last_modified_by == "Jay Doe"
    assert machine.data_type == "-"
    assert machine.version == 1


def test_machine_model_only_required():
    machine = Machine(radiation_mode="photons", name="Generic", key="generic_key")
    assert machine.radiation_mode == "photons"
    assert machine.name == "Generic"
    assert machine.description == ""
    assert machine.created_on is None
    assert machine.last_modified is None
    assert machine.created_by == ""
    assert machine.last_modified_by == ""
    assert machine.data_type == "-"
    assert machine.version == 1


def test_machine_model_invalid_version():
    with pytest.raises(ValueError):
        machine = Machine(radiation_mode="photons", name="Generic_name", version="abc")


def test_machine_model_invalid_machine():
    with pytest.raises(ValueError):
        machine = Machine(radiation_mode="protons", name="")


def test_machine_model_missing_required():
    with pytest.raises(ValueError):
        machine = Machine()


def test_photon_machine_model_from_name():
    machine = load_from_name(radiation_mode="photons", machine_name="Generic")
    assert isinstance(machine, Machine)
    assert isinstance(machine, PhotonLINAC)
    assert machine.radiation_mode == "photons"
    assert machine.name == "Generic"


def test_photon_machine_model_svdpb_get_kernel():
    machine = load_from_name(radiation_mode="photons", machine_name="Generic")
    assert isinstance(machine, PhotonLINAC)
    kernel = machine.get_kernel_by_index(0)

    assert isinstance(kernel, PhotonSVDKernel)
    nk = kernel.num_kernel_components
    assert nk == 3
    nssds = kernel.kernel_ssds.shape[0]
    npos = kernel.kernel_pos.shape[0]
    assert isinstance(kernel.kernel_data, NDArray[Shape[f"{nssds}, {nk}, {npos}"], np.float64])

    kernel1 = kernel.get_kernels_at_ssd(750)
    assert isinstance(kernel1, NDArray[Shape[f"{nk}, {npos}"], np.float64])
    kernel2 = kernel.get_kernels_at_ssd(750.2)
    assert isinstance(kernel2, NDArray[Shape[f"{nk}, {npos}"], np.float64])
    assert np.isclose(kernel1, kernel2).all()

    # check ssd warnings
    with pytest.warns(UserWarning):
        kernel_low = kernel.get_kernels_at_ssd(250)
        assert isinstance(kernel_low, NDArray[Shape[f"{nk}, {npos}"], np.float64])
        assert np.isclose(kernel.get_kernels_at_ssd(kernel.kernel_ssds.min()), kernel_low).all()

    with pytest.warns(UserWarning):
        kernel_high = kernel.get_kernels_at_ssd(5000)
        assert isinstance(kernel_high, NDArray[Shape[f"{nk}, {npos}"], np.float64])
        assert np.isclose(kernel.get_kernels_at_ssd(kernel.kernel_ssds.max()), kernel_high).all()


# TODO: model dump and alias tests


def test_proton_machine_model_from_name():
    machine = load_from_name(radiation_mode="protons", machine_name="Generic")
    assert isinstance(machine, Machine)
    assert isinstance(machine, IonAccelerator)
    assert machine.radiation_mode == "protons"
    assert machine.name == "Generic"

    assert isinstance(machine.energies, np.ndarray)
    assert isinstance(machine.peak_positions, np.ndarray)
    assert machine.pb_kernels is not None
    assert isinstance(machine.sad, float)


def test_proton_machine_model_get_kernel():
    machine = load_from_name(radiation_mode="protons", machine_name="Generic")
    assert isinstance(machine, IonAccelerator)
    kernel = machine.get_kernel_by_index(0)
    assert isinstance(kernel, ParticlePencilBeamKernel)


def test_proton_machine_model_phase_space():
    machine = load_from_name(radiation_mode="protons", machine_name="Generic")
    assert isinstance(machine, IonAccelerator)
    for _, spectrum in machine.spectra.items():
        assert isinstance(spectrum.mean, float)
        assert isinstance(spectrum.sigma, float)
        assert spectrum.mean * spectrum.sigma / 100 == pytest.approx(spectrum.sigma_absolute)
        assert spectrum.sigma / 100 == pytest.approx(spectrum.sigma_relative)

    for _, focus_list in machine.foci.items():
        for focus in focus_list:
            assert isinstance(focus.emittance.sigma_x, float)
            assert isinstance(focus.emittance.sigma_y, float)
            assert isinstance(focus.emittance.div_x, float)
            assert isinstance(focus.emittance.div_y, float)
            assert isinstance(focus.emittance.corr_x, float)
            assert isinstance(focus.emittance.corr_y, float)


def test_carbon_machine_model_from_name():
    machine = load_from_name(radiation_mode="carbon", machine_name="Generic")
    assert isinstance(machine, Machine)
    assert isinstance(machine, IonAccelerator)
    assert machine.radiation_mode == "carbon"
    assert machine.name == "Generic"

    assert isinstance(machine.energies, np.ndarray)
    assert isinstance(machine.peak_positions, np.ndarray)
    assert machine.pb_kernels is not None
    assert isinstance(machine.sad, float)


def test_carbon_machine_model_get_kernel():
    machine = load_from_name(radiation_mode="carbon", machine_name="Generic")
    assert isinstance(machine, IonAccelerator)
    kernel = machine.get_kernel_by_index(0)
    assert isinstance(kernel, ParticlePencilBeamKernel)

    abratio = kernel.alpha_beta_ratio
    assert np.isclose(abratio, kernel.alpha_x / kernel.beta_x).all()

    assert kernel.alpha.shape == (abratio.size, kernel.depths.size)
    assert kernel.beta.shape == (abratio.size, kernel.depths.size)


def test_get_available_machines_lists_shipped_machines():
    machines = get_available_machines()
    for radiation_mode in ("photons", "protons", "helium", "carbon", "oxygen", "VHEE"):
        assert "Generic" in machines[radiation_mode]
        assert machines[radiation_mode] == sorted(machines[radiation_mode], key=str.casefold)
        for machine_name in machines[radiation_mode]:
            assert isinstance(load_from_name(radiation_mode, machine_name), Machine)


def test_get_available_machines_for_single_mode():
    assert get_available_machines("VHEE") == {"VHEE": get_available_machines()["VHEE"]}
    assert get_available_machines("not_a_radiation_mode") == {"not_a_radiation_mode": []}


def _tissue_parameters_as_read_mat_delivers(tmp_path, tissue_parameters):
    """Round trip a matRad ``tissueParameters`` struct through a .mat file and read_mat."""
    path = tmp_path / "tissue_parameters.mat"
    scipy.io.savemat(path, {"tissueParameters": tissue_parameters})
    return pymatreader.read_mat(str(path))["tissueParameters"]


def _carbon_machine_with_tissue_parameters(tissue_parameters):
    raw = load_machine_from_mat(find_machine_file("carbon", "Generic"))
    raw["meta"]["tissueParameters"] = tissue_parameters
    return validate_machine(raw)


@pytest.mark.parametrize("first_field", ["cell_Line", "alphaX"])
def test_machine_with_a_single_tissue_class(tmp_path, first_field):
    """read_mat squeezes a one-class struct to scalars and strings."""
    fields = {
        "cell_Line": "Generic_LEMax05",
        "alphaX": 0.5,
        "betaX": 0.05,
        "Dt": 30.0,
        "rNucleus": 5.0,
        "model": "LEM",
    }
    ordered = {first_field: fields[first_field], **fields}
    tissue_parameters = _tissue_parameters_as_read_mat_delivers(tmp_path, ordered)
    assert isinstance(tissue_parameters["alphaX"], float)
    assert isinstance(tissue_parameters["cell_Line"], str)

    machine = _carbon_machine_with_tissue_parameters(tissue_parameters)

    assert len(machine.tissue_properties) == 1
    tissue = machine.tissue_properties[0]
    assert isinstance(tissue, LEMTissueProperties)
    assert (tissue.cell_line, tissue.alpha_x, tissue.beta_x) == ("Generic_LEMax05", 0.5, 0.05)


def test_machine_with_several_tissue_classes(tmp_path):
    tissue_parameters = _tissue_parameters_as_read_mat_delivers(
        tmp_path,
        {
            "cell_Line": np.array(["Generic_LEMax01", "Generic_LEMax05"], dtype=object),
            "alphaX": np.array([0.1, 0.5]),
            "betaX": np.array([0.05, 0.05]),
            "Dt": np.array([30.0, 30.0]),
            "rNucleus": np.array([5.0, 5.0]),
            "model": np.array(["LEM", "LEM"], dtype=object),
        },
    )

    machine = _carbon_machine_with_tissue_parameters(tissue_parameters)

    assert [(t.cell_line, t.alpha_x) for t in machine.tissue_properties] == [
        ("Generic_LEMax01", 0.1),
        ("Generic_LEMax05", 0.5),
    ]
    assert all(isinstance(t, LEMTissueProperties) for t in machine.tissue_properties)


def test_machine_tissue_parameters_with_differing_class_counts_are_rejected():
    with pytest.raises(ValueError, match="differing numbers of tissue classes"):
        _carbon_machine_with_tissue_parameters(
            {"alphaX": [0.1, 0.5], "betaX": 0.05, "model": "LEM"}
        )
