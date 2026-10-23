"""Where tissue parameter values come from: citations and the sentinels for unknown ones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

#: Source of values whose origin has not been confirmed; the default.
UNVERIFIED_SOURCE = "Unverified"

#: Source of values that were never published, e.g. in-house measurements.
UNPUBLISHED_SOURCE = "Unpublished"

#: Source strings that name no confirmed origin.
SOURCE_SENTINELS: tuple[str, ...] = (UNVERIFIED_SOURCE, UNPUBLISHED_SOURCE)


def is_verified_source(source: Optional[str]) -> bool:
    """Whether a source string names a confirmed origin, i.e. is set and no sentinel."""
    return bool(source) and source.strip().casefold() not in {
        sentinel.casefold() for sentinel in SOURCE_SENTINELS
    }


@dataclass(frozen=True)
class ParameterSource:
    """
    Provenance of a set of parameter values, e.g. of a cell line preset.

    Any string naming the origin will do: a paper, a toolkit, a dataset, a personal
    communication. :data:`UNVERIFIED_SOURCE` and :data:`UNPUBLISHED_SOURCE` stand for values
    without a confirmed origin.

    Attributes
    ----------
    citation : str
        The full source, recorded on tissues built from the values
        (`TissueProperties.cell_line_source`).
    short : str
        Compact form for labels, e.g. ``"Manganaro 2018"``; defaults to `citation`.
    note : str
        Remarks on the values, e.g. which of them the source does not cover.
    """

    citation: str = UNVERIFIED_SOURCE
    short: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        """Compact form for labels; a sentinel in lower case, e.g. ``"unverified"``."""
        if not self.verified:
            return self.citation.strip().lower() or UNVERIFIED_SOURCE.lower()
        return self.short or self.citation

    @property
    def verified(self) -> bool:
        """Whether the source names a confirmed origin, see :func:`is_verified_source`."""
        return is_verified_source(self.citation)

    @classmethod
    def coerce(cls, source: Union[str, "ParameterSource", None]) -> "ParameterSource":
        """Return a source given as a string (None: :data:`UNVERIFIED_SOURCE`) as a structure."""
        if isinstance(source, ParameterSource):
            return source
        return cls(citation=source.strip() if source and source.strip() else UNVERIFIED_SOURCE)
