from pyRadPlan.gui.widgets._analysis_widget import _NONE_LABEL, AnalysisWidget
import numpy as np
import SimpleITK as sitk


def _masks(cst):
    return {
        voi.name: sitk.GetArrayFromImage(voi.mask)
        if isinstance(voi.mask, sitk.Image)
        else np.asarray(voi.mask)
        for voi in cst.vois
    }


def test_analysis_widget_init(qapp):
    widget = AnalysisWidget()
    assert widget is not None
    assert widget.tabs.count() == 2


def test_analysis_widget_plot(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    quantities = {"Dose": dose}
    masks = _masks(cst)

    widget = AnalysisWidget()
    widget.set_data(quantities=quantities, masks=masks)

    assert widget.dvh_widget is not None
    assert widget.qi_widget is not None


def test_analysis_widget_populates_qi_table(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)
    masks = _masks(cst)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=masks, overlay_units={"Dose": "Gy"})

    rows = widget.qi_widget.to_rows()
    assert rows, "QI table was not populated"

    header, body = rows[0], rows[1:]
    assert header[0] == "Structure"
    assert "Quantity" not in header
    assert len(body) == len(masks)
    assert any(h.startswith("mean") for h in header)


def test_analysis_widget_qi_follows_structure_selection(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst))

    widget._deselect_all_vois()
    assert widget.qi_widget.to_rows() == []

    widget._select_all_vois()
    assert len(widget.qi_widget.to_rows()) - 1 == len(cst.vois)


def test_analysis_widget_qi_custom_parameters(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    widget.ref_vols_edit.setText("50")
    widget.ref_doses_edit.setText("1")
    widget._replot()

    metrics = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert metrics == ["mean", "std", "max", "min", "D50", "V1Gy"]


def test_analysis_widget_qi_invalid_parameters_fall_back(qapp, test_data_photons):
    """Garbage in the D_x/V_x fields must not raise; defaults are used instead."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    widget.ref_vols_edit.setText("not a number")
    widget._replot()

    metrics = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "D50" in metrics  # the default ref_vols were used


def test_analysis_widget_qi_unparseable_unit(qapp, test_data_photons):
    """A unit string pint cannot parse degrades to unitless QIs instead of raising."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(
        quantities={"Other": dose},
        masks=_masks(cst),
        overlay_units={"Other": "bogus unit"},
    )

    header = widget.qi_widget.to_rows()[0]
    assert "mean" in header  # no unit suffix, and no empty "[]"
    assert "mean []" not in header


def test_analysis_widget_qi_rbe_unit_alias(qapp, test_data_photons):
    """'Gy (RBE)' is a display label pint cannot parse; it is aliased to Gy."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(
        quantities={"RBExDose": dose},
        masks=_masks(cst),
        overlay_units={"RBExDose": "Gy (RBE)"},
    )

    header = widget.qi_widget.to_rows()[0]
    assert "mean [Gy]" in header


def test_analysis_widget_metric_checkboxes(qapp, test_data_photons):
    """Only the fixed reductions get a checkbox; D_x/V_x are driven by the inputs."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(
        quantities={"Dose": dose},
        masks=_masks(cst),
        overlay_units={"Dose": "Gy"},
    )

    assert list(widget._metric_checkboxes) == ["mean", "std", "max", "min"]
    assert all(cb.isChecked() for cb in widget._metric_checkboxes.values())

    # Listing them twice in the same box would be redundant
    assert not [n for n in widget._metric_checkboxes if n.startswith(("D", "V"))]

    # ... but they are still columns
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "D50" in columns
    assert columns[:4] == ["mean", "std", "max", "min"]


def test_analysis_widget_param_fields_show_actual_values(qapp, test_data_photons):
    """The D_x/V_x inputs display the reference values actually used, never 'auto'."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    assert widget.ref_vols_edit.text() == "2 5 50 95 98"

    # The auto-derived V_x doses are written back rather than left blank
    ref_doses = widget.ref_doses_edit.text().split()
    assert len(ref_doses) == 5
    columns = widget.qi_widget.to_rows()[0]
    for value in ref_doses:
        assert any(h.startswith(f"V{value}") for h in columns)

    # Clearing re-derives them
    widget.ref_doses_edit.setText("")
    widget._replot()
    assert widget.ref_doses_edit.text().split() == ref_doses

    # Invalid input falls back to the defaults, and that is made visible
    widget.ref_vols_edit.setText("not a number")
    widget._replot()
    assert widget.ref_vols_edit.text() == "2 5 50 95 98"


def test_analysis_widget_metric_selection_filters_columns(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    widget._metric_checkboxes["std"].setChecked(False)
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "std" not in columns
    assert "mean" in columns

    # Unchecking every reduction leaves the parameter-driven columns in place
    widget._deselect_all_metrics()
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert columns
    assert not any(c in columns for c in ("mean", "std", "max", "min"))
    assert "D50" in columns

    widget._select_all_metrics()
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert columns[:4] == ["mean", "std", "max", "min"]


def test_analysis_widget_metric_selection_survives_param_change(qapp, test_data_photons):
    """Changing D_x/V_x keeps the reduction check states."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    widget._metric_checkboxes["std"].setChecked(False)
    widget.ref_doses_edit.setText("1 2")
    widget._replot()

    assert not widget._metric_checkboxes["std"].isChecked()
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "std" not in columns
    assert "V1Gy" in columns and "V2Gy" in columns


def test_analysis_widget_qi_out_of_range_parameters(qapp, test_data_photons):
    """An out-of-range D_x falls back to its defaults without wiping the V_x field."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})

    widget.ref_vols_edit.setText("150")
    widget.ref_doses_edit.setText("1 2")
    widget._replot()

    assert widget.ref_vols_edit.text() == "2 5 50 95 98"
    assert widget.ref_doses_edit.text() == "1 2"
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "D50" in columns and "V1Gy" in columns

    # A negative V_x dose likewise falls back to the defaults
    widget.ref_doses_edit.setText("-5")
    widget._replot()
    assert len(widget.ref_doses_edit.text().split()) == 5


def test_analysis_widget_qi_sync_keeps_fields_without_results(qapp):
    """With nothing computed, the D_x/V_x fields keep the user's input."""
    widget = AnalysisWidget()
    widget.ref_vols_edit.setText("10")
    widget.ref_doses_edit.setText("3")
    widget._sync_param_fields(None)
    assert widget.ref_vols_edit.text() == "10"
    assert widget.ref_doses_edit.text() == "3"


def test_analysis_widget_qi_enter_corrects_invalid_text(qapp, test_data_photons):
    """Pressing Enter (focus stays in the field) still shows the defaults used."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(quantities={"Dose": dose}, masks=_masks(cst), overlay_units={"Dose": "Gy"})
    widget.show()
    widget.activateWindow()
    widget.ref_vols_edit.setFocus()
    qapp.processEvents()
    assert widget.ref_vols_edit.hasFocus()

    widget.ref_vols_edit.setText("abc")
    widget.ref_vols_edit.editingFinished.emit()

    assert widget.ref_vols_edit.text() == "2 5 50 95 98"
    widget.close()


def _dose_and_let(cst, result):
    dose = np.swapaxes(result["physicalDose"], 0, 1)
    widget = AnalysisWidget()
    widget.set_data(
        quantities={"Dose": dose, "LET": dose * 3.0},
        masks=_masks(cst),
        overlay_units={"Dose": "Gy", "LET": "keV/µm"},
        initial_quantity="Dose",
    )
    return widget


def test_analysis_widget_quantity_switch_updates_units(qapp, test_data_photons):
    """Switching the quantity relabels the DVH axis, QI headers and V_x label."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    assert widget.quantity_combo.currentText() == "Dose"
    assert "Gy" in widget.dvh_widget.figure.axes[0].get_xlabel()

    widget.quantity_combo.setCurrentText("LET")

    assert "keV" in widget.dvh_widget.figure.axes[0].get_xlabel()
    header = widget.qi_widget.to_rows()[0]
    assert "mean [keV / \u03bcm]" in header
    assert not any("Gy" in h for h in header)
    assert "keV" in widget._ref_doses_label.text()


def test_analysis_widget_vx_label_unit(qapp, test_data_photons):
    """The V_x label carries the shown quantity's unit, or none when unparsable."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(
        quantities={"Dose": dose, "Other": dose},
        masks=_masks(cst),
        overlay_units={"Dose": "Gy", "Other": "bogus unit"},
        initial_quantity="Dose",
    )
    assert widget._ref_doses_label.text() == "V_x [Gy]:"

    widget.quantity_combo.setCurrentText("Other")
    assert widget._ref_doses_label.text() == "V_x:"


def test_analysis_widget_vx_remembered_per_quantity(qapp, test_data_photons):
    """Typed V_x thresholds belong to their quantity; others keep derived defaults."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    widget.ref_doses_edit.setText("1 2")
    widget.ref_doses_edit.editingFinished.emit()
    assert widget.ref_doses_edit.text() == "1 2"

    widget.quantity_combo.setCurrentText("LET")
    let_doses = widget.ref_doses_edit.text().split()
    assert len(let_doses) == 5
    assert let_doses != ["1", "2"]

    widget.quantity_combo.setCurrentText("Dose")
    assert widget.ref_doses_edit.text() == "1 2"
    columns = [h.split(" [")[0] for h in widget.qi_widget.to_rows()[0][1:]]
    assert "V1Gy" in columns and "V2Gy" in columns

    widget.quantity_combo.setCurrentText("LET")
    assert widget.ref_doses_edit.text().split() == let_doses


def test_analysis_widget_dvhs_computed_lazily(qapp, test_data_photons):
    """DVHs are only computed for the shown quantity."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    assert widget._dvh_cache
    assert {qty for qty, _ in widget._dvh_cache} == {"Dose"}

    widget.quantity_combo.setCurrentText("LET")
    assert {qty for qty, _ in widget._dvh_cache} == {"Dose", "LET"}


def test_analysis_widget_voi_toggle_reuses_qi_cache(qapp, test_data_photons):
    """Toggling a VOI re-renders from the QI cache without adding entries."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    n_cached = len(widget._qi_cache)
    name = next(iter(widget._voi_checkboxes))
    widget._voi_checkboxes[name].setChecked(False)
    widget._voi_checkboxes[name].setChecked(True)
    widget._replot()
    assert len(widget._qi_cache) == n_cached


def _linestyles(widget):
    return {line.get_linestyle() for line in widget.dvh_widget.figure.axes[0].get_lines()}


def test_analysis_widget_secondary_defaults_to_none(qapp, test_data_photons):
    """The compare dropdown lists every quantity after the None entry, which is preselected."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    items = [widget.secondary_combo.itemText(i) for i in range(widget.secondary_combo.count())]
    assert items == [_NONE_LABEL, "Dose", "LET"]
    assert widget.secondary_combo.currentText() == _NONE_LABEL
    assert all(
        line.get_linestyle() == "-" for line in widget.dvh_widget.figure.axes[0].get_lines()
    )


def test_analysis_widget_secondary_compatible_unit_shares_axis(qapp, test_data_photons):
    """A Gy-compatible secondary is dashed on the same axis; the QI table stays on the primary."""
    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)

    widget = AnalysisWidget()
    widget.set_data(
        quantities={"Dose": dose, "RBExDose": dose * 1.1},
        masks=_masks(cst),
        overlay_units={"Dose": "Gy", "RBExDose": "Gy (RBE)"},
        initial_quantity="Dose",
    )
    vx_label = widget._ref_doses_label.text()

    widget.secondary_combo.setCurrentText("RBExDose")

    assert len(widget.dvh_widget.figure.axes) == 1
    assert widget.dvh_widget.secondary_axes is None
    assert {"-", "--"} <= _linestyles(widget)
    assert "mean [Gy]" in widget.qi_widget.to_rows()[0]
    assert widget._ref_doses_label.text() == vx_label


def test_analysis_widget_secondary_incompatible_unit_gets_twin_axis(qapp, test_data_photons):
    """An incompatible secondary unit gets its own x axis; the QI table stays on the primary."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    widget.secondary_combo.setCurrentText("LET")

    assert widget.dvh_widget.secondary_axes is not None
    assert "keV" in widget.dvh_widget.secondary_axes.get_xlabel()
    assert "Gy" in widget.dvh_widget.figure.axes[0].get_xlabel()
    assert not any("keV" in h for h in widget.qi_widget.to_rows()[0])


def test_analysis_widget_secondary_equal_to_primary_is_ignored(qapp, test_data_photons):
    """Comparing the primary quantity against itself draws no dashed curves."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)

    widget.secondary_combo.setCurrentText("Dose")

    assert widget._secondary_quantity() == ""
    assert "--" not in _linestyles(widget)
    assert widget.dvh_widget.secondary_axes is None


def test_analysis_widget_secondary_caches_dvhs_only(qapp, test_data_photons):
    """The secondary quantity adds DVHs to the cache but never computes QIs."""
    ct, cst, result = test_data_photons
    widget = _dose_and_let(cst, result)
    n_qi_cached = len(widget._qi_cache)

    widget.secondary_combo.setCurrentText("LET")

    assert {qty for qty, _ in widget._dvh_cache} == {"Dose", "LET"}
    assert len(widget._qi_cache) == n_qi_cached
