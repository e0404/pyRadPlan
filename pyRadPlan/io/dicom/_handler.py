"""Combined import/export handler for DICOM data."""

from ._importer import DicomImporter
from ._exporter import DicomExporter


class DicomHandler(DicomImporter, DicomExporter):
    """Low-level handler bundling :class:`DicomImporter` and :class:`DicomExporter`.

    The two bases do not cooperate through ``super().__init__`` (``BaseImporter``
    does not chain to ``BaseExporter``), so both are initialized explicitly here to
    keep the exporter's ``structure_format`` and the importer's state consistent.
    """

    name = "DICOM Handler"

    def __init__(  # noqa: PLR0913 - mirrors the exporter's options
        self,
        path,
        structure_format: str = "rtstruct",
        metadata=None,
        anonymize: bool = False,
        overwrite: bool = False,
        roi_interpreted_types=None,
        per_beam_doses: bool = False,
        export_plan: bool = True,
    ):
        DicomImporter.__init__(self, path)
        DicomExporter.__init__(
            self,
            path,
            structure_format=structure_format,
            metadata=metadata,
            anonymize=anonymize,
            overwrite=overwrite,
            roi_interpreted_types=roi_interpreted_types,
            per_beam_doses=per_beam_doses,
            export_plan=export_plan,
        )
