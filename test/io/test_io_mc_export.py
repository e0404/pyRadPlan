"""Tests for the TOPAS and FRED simulation input export (no Monte Carlo code is run)."""

import re

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan import IonPlan, PhotonPlan, generate_stf, load_tg119
from pyRadPlan.io import TopasExporter, FredExporter, save_data
from pyRadPlan.io._factory import format_for_path, get_exporter

HISTORIES = 1e5
RESOLUTION = 10.0


@pytest.fixture(scope="module")
def plan():
    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=5)
    pln.prop_stf = {"gantry_angles": [0, 90], "couch_angles": [0, 0], "bixel_width": 30}
    pln.prop_dose_calc = {
        "dose_grid": {"resolution": {"x": RESOLUTION, "y": RESOLUTION, "z": RESOLUTION}},
        "num_histories_direct": HISTORIES,
    }
    stf = generate_stf(ct, cst, pln)
    weights = np.random.default_rng(1).random(stf.total_number_of_bixels)
    weights[::4] = 0.0
    return pln, ct, cst, stf, weights


def _dose_grid_dim(ct):
    return [
        int(np.ceil(n * ct.resolution[axis] / RESOLUTION)) for n, axis in zip(ct.cube_dim, "xyz")
    ]


def _topas_histories(directory, beam):
    text = (directory / f"beamSetup_pyRadPlan_plan_field{beam}.txt").read_text()
    match = re.search(r"Tf/Beam/Current/Values = (\d+) ([^\n]*)", text)
    values = [int(v) for v in match.group(2).split()]
    assert len(values) == int(match.group(1))
    return values


def _fred_spots(directory):
    text = (directory / "inp" / "plan" / "plan.inp").read_text()
    nprim = float(re.search(r"nprim = ([\d.e+]+)", text).group(1))
    spot_weights = [float(w) for w in re.findall(r"'beamletID': \d+, .*?'w':([\d.e+-]+)", text)]
    return nprim, spot_weights


class TestTopas:
    def test_files(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        out = tmp_path / "out"
        TopasExporter(out).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        runs = 5
        expected = {"pyRadPlan_cube.dat", "pyRadPlan_cube.txt"}
        for beam in range(len(stf.beams)):
            expected.add(f"beamSetup_pyRadPlan_plan_field{beam}.txt")
            expected.update(f"pyRadPlan_plan_field{beam}_run{run}.txt" for run in range(runs))
        assert {p.name for p in out.iterdir()} == expected

    def test_patient_files(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        TopasExporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        dims = _dose_grid_dim(ct)
        text = (tmp_path / "pyRadPlan_cube.txt").read_text()
        assert f"NumberOfVoxelsX = {dims[0]}" in text
        assert f"NumberOfVoxelsY = {dims[1]}" in text
        assert f"NumberOfVoxelsZ = 1 {dims[2]}" in text
        assert (tmp_path / "pyRadPlan_cube.dat").stat().st_size == 2 * int(np.prod(dims))

    def test_weights_become_histories(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        TopasExporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        per_beam = [_topas_histories(tmp_path, b) for b in range(len(stf.beams))]
        assert [len(v) for v in per_beam] == [b.total_number_of_bixels for b in stf.beams]
        # histories of one run: num_histories_direct / num_runs, drawn over all spots
        assert sum(map(sum, per_beam)) == HISTORIES / 5
        # spots without weight get no histories
        assert sum(v == 0 for values in per_beam for v in values) >= np.count_nonzero(w == 0)

    def test_options_override_plan(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        TopasExporter(tmp_path, num_histories=4e4, num_runs=2).save(
            ct=ct, cst=cst, pln=pln, stf=stf, weights=w
        )

        assert sum(sum(_topas_histories(tmp_path, b)) for b in range(2)) == 2e4
        assert (tmp_path / "pyRadPlan_plan_field1_run1.txt").exists()
        assert not (tmp_path / "pyRadPlan_plan_field1_run2.txt").exists()

    def test_without_cst(self, plan, tmp_path):
        pln, ct, _, stf, w = plan
        TopasExporter(tmp_path).save(ct=ct, pln=pln, stf=stf, weights=w)
        assert (tmp_path / "pyRadPlan_cube.txt").exists()


class TestFred:
    def test_files(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        FredExporter(tmp_path / "out").save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        files = {
            p.relative_to(tmp_path / "out").as_posix()
            for p in (tmp_path / "out").rglob("*")
            if p.is_file()
        }
        assert files == {
            "inp/fred.inp",
            "inp/plan/plan.inp",
            "inp/plan/planDelivery.inp",
            "inp/regions/regions.inp",
            "inp/regions/hLut.inp",
            "inp/regions/CTpatient.mhd",
            "inp/regions/CTpatient.raw",
        }

    def test_patient_file(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        FredExporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        image = sitk.ReadImage(str(tmp_path / "inp" / "regions" / "CTpatient.mhd"))
        assert list(image.GetSize()) == _dose_grid_dim(ct)
        assert image.GetSpacing() == (RESOLUTION,) * 3

    def test_weights_in_plan(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        FredExporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        nprim, spot_weights = _fred_spots(tmp_path)
        assert len(spot_weights) == stf.total_number_of_bixels
        assert nprim == np.floor(HISTORIES / stf.total_number_of_bixels)
        assert sum(spot_weights) == pytest.approx(w.sum() * 1e6, rel=1e-6)
        assert sum(x == 0 for x in spot_weights) == np.count_nonzero(w == 0)

    def test_num_histories_option(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        FredExporter(tmp_path, num_histories=1e6).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)
        assert _fred_spots(tmp_path)[0] == np.floor(1e6 / stf.total_number_of_bixels)

    def test_without_cst(self, plan, tmp_path):
        pln, ct, _, stf, w = plan
        FredExporter(tmp_path).save(ct=ct, pln=pln, stf=stf, weights=w)
        assert (tmp_path / "inp" / "plan" / "plan.inp").exists()


@pytest.mark.parametrize("exporter", [TopasExporter, FredExporter])
class TestCommon:
    def test_weights_required(self, plan, tmp_path, exporter):
        pln, ct, cst, stf, _ = plan
        with pytest.raises(ValueError, match="spot weights"):
            exporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf)
        assert not any(tmp_path.iterdir())

    def test_weights_from_result(self, plan, tmp_path, exporter):
        pln, ct, cst, stf, w = plan
        exporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, result={"w": w})
        assert any(tmp_path.iterdir())

    @pytest.mark.parametrize("missing", ["ct", "pln", "stf"])
    def test_required_objects(self, plan, tmp_path, exporter, missing):
        pln, ct, cst, stf, w = plan
        objects = {"ct": ct, "pln": pln, "stf": stf}
        del objects[missing]
        with pytest.raises(ValueError, match="needs ct, pln and stf"):
            exporter(tmp_path).save(cst=cst, weights=w, **objects)

    def test_wrong_number_of_weights(self, plan, tmp_path, exporter):
        pln, ct, cst, stf, w = plan
        with pytest.raises(ValueError, match="Number of weights"):
            exporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w[:-1])

    def test_overwrite_policy(self, plan, tmp_path, exporter):
        pln, ct, cst, stf, w = plan
        objects = {"ct": ct, "cst": cst, "pln": pln, "stf": stf, "weights": w}
        exporter(tmp_path).save(**objects)
        keep = tmp_path / "notes.txt"
        keep.write_text("mine")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            exporter(tmp_path).save(**objects)

        stale = next(p for p in sorted(tmp_path.rglob("*")) if p.is_file() and p != keep)
        stale.write_text("stale")
        exporter(tmp_path, overwrite=True).save(**objects)
        assert stale.read_text() != "stale"
        assert keep.read_text() == "mine"

    def test_inputs_unchanged(self, plan, tmp_path, exporter):
        pln, ct, cst, stf, w = plan
        pln_before = pln.model_dump_json()
        stf_before = stf.model_dump_json()
        cube_before = sitk.GetArrayFromImage(ct.cube_hu).copy()
        w_before = w.copy()

        exporter(tmp_path, num_histories=3e4).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

        assert pln.model_dump_json() == pln_before
        assert stf.model_dump_json() == stf_before
        np.testing.assert_array_equal(sitk.GetArrayFromImage(ct.cube_hu), cube_before)
        np.testing.assert_array_equal(w, w_before)

    def test_photon_plan_is_refused(self, tmp_path, exporter):
        ct, cst = load_tg119()
        pln = PhotonPlan(machine="Generic")
        pln.prop_stf = {"gantry_angles": [0], "couch_angles": [0], "bixel_width": 30}
        stf = generate_stf(ct, cst, pln)
        w = np.ones(stf.total_number_of_bixels)
        with pytest.raises(ValueError, match="supports ion plans"):
            exporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)

    def test_settings_of_other_engines_are_ignored(self, plan, tmp_path, exporter, recwarn):
        pln, ct, cst, stf, w = plan
        pln = pln.model_copy(deep=True)
        pln.prop_dose_calc = {**pln.prop_dose_calc, "engine": "HongPB", "lateral_model": "single"}
        exporter(tmp_path).save(ct=ct, cst=cst, pln=pln, stf=stf, weights=w)
        assert not [x for x in recwarn if "not found" in str(x.message)]


class TestSaveData:
    @pytest.mark.parametrize("fmt", ["topas", "fred"])
    def test_format_argument(self, plan, tmp_path, fmt):
        pln, ct, cst, stf, w = plan
        target = tmp_path / "plan"
        written = save_data(
            ct=ct, cst=cst, pln=pln, stf=stf, weights=w, file_name=target, format=fmt
        )
        assert written == str(target)
        assert any(target.iterdir())

    @pytest.mark.parametrize("fmt", ["topas", "fred"])
    def test_extension_selects_format(self, plan, tmp_path, fmt):
        pln, ct, cst, stf, w = plan
        target = tmp_path / f"plan.{fmt}"
        save_data(ct=ct, cst=cst, pln=pln, stf=stf, weights=w, file_name=target)
        assert any(target.iterdir())

    def test_exporter_options(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        objects = {"ct": ct, "cst": cst, "pln": pln, "stf": stf, "weights": w}
        save_data(file_name=tmp_path, format="topas", **objects)
        with pytest.raises(FileExistsError):
            save_data(file_name=tmp_path, format="topas", **objects)
        save_data(
            file_name=tmp_path,
            format="topas",
            exporter_options={"overwrite": True, "num_histories": 2e4},
            **objects,
        )
        assert sum(sum(_topas_histories(tmp_path, b)) for b in range(2)) == 4e3

    def test_unsupported_objects_warn(self, plan, tmp_path):
        pln, ct, cst, stf, w = plan
        with pytest.warns(UserWarning, match="topas format cannot store dose"):
            save_data(
                ct=ct,
                cst=cst,
                pln=pln,
                stf=stf,
                weights=w,
                dose=ct.cube_hu,
                file_name=tmp_path,
                format="topas",
            )

    @pytest.mark.parametrize("fmt", ["topas", "fred"])
    def test_needs_file_name(self, plan, fmt):
        pln, ct, cst, stf, w = plan
        with pytest.raises(ValueError, match="needs 'file_name'"):
            save_data(ct=ct, cst=cst, pln=pln, stf=stf, weights=w, format=fmt)


def test_registry():
    for fmt, cls in (("topas", TopasExporter), ("fred", FredExporter)):
        assert get_exporter(fmt) is cls
        assert not cls.container
        assert {"ct", "cst", "pln", "stf", "weights", "result"} == set(cls.supports)
    assert format_for_path("plan.topas") == "topas"
    assert format_for_path("plan.fred") == "fred"
    for name, fmt in (("x.mat", "mat"), ("x.txt", None), ("x.inp", None), ("x.dcm", "dcm")):
        assert format_for_path(name) == fmt
