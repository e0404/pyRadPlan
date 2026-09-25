import csv

import numpy as np
import SimpleITK as sitk

from pyRadPlan.analysis import QICollection
from pyRadPlan.gui.widgets.analysis._qi import QITableWidget


def _masks(cst):
    return {
        voi.name: sitk.GetArrayFromImage(voi.mask)
        if isinstance(voi.mask, sitk.Image)
        else np.asarray(voi.mask)
        for voi in cst.vois
    }


def _collection(cst, dose, **kwargs):
    return QICollection.from_masks(_masks(cst), dose, **kwargs)


def _dose(result):
    return np.swapaxes(result["physicalDose"], 0, 1)


def test_qi_table_widget_init(qapp):
    widget = QITableWidget()
    assert widget is not None
    assert widget.table is not None
    assert widget.to_rows() == []
    # Nothing to export yet
    assert not widget.copy_btn.isEnabled()
    assert not widget.export_btn.isEnabled()


def test_qi_table_populates(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)

    rows = widget.to_rows()
    header, body = rows[0], rows[1:]

    assert header[0] == "Structure"
    assert "Quantity" not in header
    assert len(body) == len(cst.vois)
    assert {r[0] for r in body} == {voi.name for voi in cst.vois}

    # Columns follow the metrics actually computed
    assert [h.split(" [")[0] for h in header[1:]] == [
        "mean",
        "std",
        "max",
        "min",
        "D50",
        "V1Gy",
    ]

    assert widget.table.rowCount() == len(body)
    assert widget.table.columnCount() == len(header)
    assert widget.copy_btn.isEnabled()


def test_qi_table_respects_structure_selection(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])
    selected = [cst.vois[0].name]

    widget = QITableWidget()
    widget.set_qis(qis, structures=selected)

    body = widget.to_rows()[1:]
    assert [r[0] for r in body] == selected

    # Unknown names are skipped rather than raising
    widget.set_qis(qis, structures=[*selected, "not-a-voi"])
    assert [r[0] for r in widget.to_rows()[1:]] == selected


def test_qi_table_header_is_structure_and_unit_labelled_metrics(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])
    selected = [voi.name for voi in cst.vois[:2]]

    widget = QITableWidget()
    widget.set_qis(qis, structures=selected)

    header, body = widget.to_rows()[0], widget.to_rows()[1:]
    assert header == [
        "Structure",
        "mean [Gy]",
        "std [Gy]",
        "max [Gy]",
        "min [Gy]",
        "D50 [Gy]",
        "V1Gy [%]",
    ]
    assert [r[0] for r in body] == selected
    assert all(len(r) == len(header) for r in body)


def test_qi_table_clears(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)
    assert widget.to_rows()

    widget.set_qis(None)
    assert widget.to_rows() == []
    assert widget.table.rowCount() == 0
    assert not widget.export_btn.isEnabled()

    widget.set_qis(qis)
    widget.set_qis(QICollection(), structures=[])
    assert widget.to_rows() == []


def test_qi_table_formats_nan_as_dash(qapp, test_data_photons):
    """An empty VOI yields NaN metrics, which must render as '-', not 'nan'."""
    ct, cst, result = test_data_photons
    dose = _dose(result)

    masks = _masks(cst)
    empty_name = "empty"
    masks[empty_name] = np.zeros_like(next(iter(masks.values())))
    qis = QICollection.from_masks(masks, dose, ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis, structures=[empty_name])

    row = widget.to_rows()[1]
    assert row[0] == empty_name
    assert all(cell == "-" for cell in row[1:])
    assert "nan" not in " ".join(row).lower()


def test_qi_table_csv_export_matches_table(qapp, test_data_photons, tmp_path):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)

    target = tmp_path / "qi.csv"
    assert widget.export_csv(str(target)) == str(target)

    with open(target, newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == widget.to_rows()


def test_qi_table_export_noop_when_empty(qapp, tmp_path):
    widget = QITableWidget()
    target = tmp_path / "qi.csv"

    assert widget.export_csv(str(target)) is None
    assert not target.exists()


def test_qi_table_metric_filter(qapp, test_data_photons):
    """Only the requested metrics become columns."""
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis, metrics=["mean", "D50"])

    assert [h.split(" [")[0] for h in widget.to_rows()[0][1:]] == ["mean", "D50"]

    # An empty selection clears the table rather than showing bare row labels
    widget.set_qis(qis, metrics=[])
    assert widget.to_rows() == []


def test_qi_table_export_button_opens_dialog(qapp, test_data_photons, tmp_path, monkeypatch):
    """Regression: clicked() emits a bool that must not land in export_csv(path=...).

    PySide6 binds the ``checked`` argument to the optional *path* parameter, so
    the button used to call ``open(False)`` - file descriptor 0 - instead of
    opening the save dialog.
    """
    from PySide6.QtWidgets import QFileDialog

    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)

    target = tmp_path / "from_button.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )

    widget.export_btn.click()

    assert target.exists()
    with open(target, newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == widget.to_rows()


def test_qi_table_copy_button(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)
    widget.copy_btn.click()

    text = qapp.clipboard().text()
    assert text.split("\n")[0].split("\t") == widget.to_rows()[0]


def test_qi_table_export_cancelled_dialog(qapp, test_data_photons, monkeypatch):
    """Cancelling the save dialog returns None and writes nothing."""
    from PySide6.QtWidgets import QFileDialog

    ct, cst, result = test_data_photons
    qis = _collection(cst, _dose(result), ref_vols=[50], ref_doses=[1.0])

    widget = QITableWidget()
    widget.set_qis(qis)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    assert widget.export_csv() is None
