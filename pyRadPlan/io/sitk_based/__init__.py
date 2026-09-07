"""SimpleITK-based import/export backends (NIfTI, NRRD, MetaImage).

These single-image formats share the base classes in :mod:`pyRadPlan.io.sitk_based.base`
and map a patient onto a folder of images (``ct``, ``dose``, and a label-map
``segmentation`` + JSON sidecar). Each subpackage self-registers its importer/exporter.
"""

from .base import SitkImporterBase, SitkExporterBase
from .nifti import NiftiImporter, NiftiExporter, NiftiHandler
from .nrrd import NrrdImporter, NrrdExporter, NrrdHandler
from .meta_image import MetaImageImporter, MetaImageExporter, MetaImageHandler
from ._binary_import import (
    load_binary_patient,
    masks_to_cst,
    mask_file_to_voi,
    image_file_to_vois,
    infer_image_kind,
    read_ct_image,
    list_image_files,
)

__all__ = [
    "SitkImporterBase",
    "SitkExporterBase",
    "NiftiImporter",
    "NiftiExporter",
    "NiftiHandler",
    "NrrdImporter",
    "NrrdExporter",
    "NrrdHandler",
    "MetaImageImporter",
    "MetaImageExporter",
    "MetaImageHandler",
    "load_binary_patient",
    "masks_to_cst",
    "mask_file_to_voi",
    "image_file_to_vois",
    "infer_image_kind",
    "read_ct_image",
    "list_image_files",
]
