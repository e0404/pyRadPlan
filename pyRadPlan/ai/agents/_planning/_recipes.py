"""Recipes for the optimization helper structures the planning agent creates."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class _Recipe(BaseModel):
    name: str = Field(description="Helper name; 'opt_' is prefixed if missing.")
    min_cc: float = Field(0.1, description="Refuse results smaller than this.")


class ExcludeSpec(BaseModel):
    voi: str
    margin_mm: float = 0.0


class RingRecipe(_Recipe):
    """Shell around a target following its shape, cropped to the patient, targets removed."""

    kind: Literal["ring"] = "ring"
    target: str
    inner_mm: float = 0.0
    outer_mm: float = 20.0
    exclude: list[ExcludeSpec] = Field(
        default_factory=list, description="OARs (+margin) to cut out of the ring."
    )
    skip_concavity: bool = Field(
        False, description="Drop ring voxels inside the target's convex hull (concave targets)."
    )


class MarginRecipe(_Recipe):
    """Isotropic expansion (PRV) or shrink (negative margin) of a VOI."""

    kind: Literal["margin"] = "margin"
    source: str
    margin_mm: float


class BooleanRecipe(_Recipe):
    """a minus / intersect / union b, with an optional margin on b."""

    kind: Literal["subtract", "intersect", "union"]
    a: str
    b: str
    b_margin_mm: float = 0.0


class IsodoseRecipe(_Recipe):
    """Region where the current dose >= threshold, by default outside targets (+margin)."""

    kind: Literal["isodose"] = "isodose"
    threshold: float
    outside_targets: bool = True
    target_margin_mm: float = 3.0


HelperRecipe = Annotated[
    Union[RingRecipe, MarginRecipe, BooleanRecipe, IsodoseRecipe], Field(discriminator="kind")
]
