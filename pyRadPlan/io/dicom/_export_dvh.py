"""RT DVH module (C.8.8.4) of an exported RTDOSE: one cumulative DVH per exported ROI."""

import logging
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.valuerep import format_number_as_ds

from pyRadPlan.analysis import DVH
from pyRadPlan.cst import StructureSet

logger = logging.getLogger(__name__)

#: Number of dose bins of each exported DVH.
DVH_NUM_BINS = 1000


@dataclass
class DvhExport:
    """Structures an RTDOSE gets its DVH sequence for.

    Attributes
    ----------
    cst : StructureSet
        The structures, exported as RTSTRUCT with ROI numbers ``1..N`` in VOI order.
    structure_set : tuple of str
        SOP class UID and SOP instance UID of the exported RTSTRUCT.
    """

    cst: StructureSet
    structure_set: tuple[str, str]


def _ds_list(values) -> list:
    return [float(format_number_as_ds(float(v))) for v in values]


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def _mask_on_grid(mask: sitk.Image, grid: sitk.Image) -> sitk.Image:
    if mask.GetDimension() == 4:
        mask = mask[:, :, :, 0]
    if _same_grid(mask, grid):
        return mask
    return sitk.Resample(mask, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def _referenced_structure_set(structure_set: tuple[str, str]) -> Sequence:
    item = Dataset()
    item.ReferencedSOPClassUID = structure_set[0]
    item.ReferencedSOPInstanceUID = structure_set[1]
    return Sequence([item])


def add_dvh_module(ds: Dataset, dose: sitk.Image, dose_type: str, export: DvhExport) -> None:
    """Add the RT DVH module to an RTDOSE dataset.

    Cumulative DVHs are computed on the dose grid, with the masks of the structures
    resampled (nearest neighbor) to it if their grid differs. Every DVH has
    :data:`DVH_NUM_BINS` bins of equal width from zero to the maximum dose of the
    distribution; bin widths and volumes (cm3) are written as (D_i, V_i) pairs, V_i being the
    volume receiving at least the lower edge of bin i. Structures without a voxel on the
    dose grid are skipped with a warning.

    Parameters
    ----------
    ds : Dataset
        The RTDOSE dataset, modified in place.
    dose : sitk.Image
        The dose in Gy as stored in the file.
    dose_type : str
        DICOM Dose Type of the RTDOSE.
    export : DvhExport
        The structures and the RTSTRUCT to reference.
    """
    max_dose = float(sitk.GetArrayViewFromImage(dose).max())
    # Rounding up keeps the maximum dose inside the last bin.
    width = (
        float(format_number_as_ds(max_dose / DVH_NUM_BINS * (1 + 1e-9))) if max_dose > 0 else 1.0
    )
    values = sitk.GetArrayFromImage(dose)

    sequence = Sequence()
    skipped = []
    for roi_number, voi in enumerate(export.cst.vois, start=1):
        mask = _mask_on_grid(voi.mask, dose)
        selected = values[sitk.GetArrayViewFromImage(mask).astype(bool)]
        if selected.size == 0:
            skipped.append(voi.name)
            continue

        dvh = DVH.compute(
            dose,
            mask=mask,
            num_points=DVH_NUM_BINS,
            max_value=width * DVH_NUM_BINS,
            name=voi.name,
        )
        volume = dvh.cum_volume / 100.0 * dvh.total_volume

        item = Dataset()
        item.DVHType = "CUMULATIVE"
        item.DoseUnits = "GY"
        item.DoseType = dose_type
        item.DVHDoseScaling = 1
        item.DVHVolumeUnits = "CM3"
        item.DVHNumberOfBins = DVH_NUM_BINS
        data = np.empty(2 * DVH_NUM_BINS)
        data[0::2] = width
        data[1::2] = volume
        item.DVHData = _ds_list(data)

        roi = Dataset()
        roi.ReferencedROINumber = roi_number
        roi.DVHROIContributionType = "INCLUDED"
        item.DVHReferencedROISequence = Sequence([roi])

        item.DVHMinimumDose = _ds_list([selected.min()])[0]
        item.DVHMaximumDose = _ds_list([selected.max()])[0]
        item.DVHMeanDose = _ds_list([selected.mean()])[0]
        sequence.append(item)

    if skipped:
        logger.warning("No DVH for structures without voxels on the dose grid: %s", skipped)

    ds.ReferencedStructureSetSequence = _referenced_structure_set(export.structure_set)
    if sequence:
        ds.DVHSequence = sequence
