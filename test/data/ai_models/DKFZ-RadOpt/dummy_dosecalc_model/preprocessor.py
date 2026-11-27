"""Preprocessor for the dummy dose-calc model (dose + physical_dose_var + LET).

Builds a ``(batch, 2, depth, lateral, lateral)`` network input from a CT cuboid
and per-bixel energy, runs a single deterministic forward pass and reports dose,
LET-weighted dose and a dose variance derived from the dose itself. Being
deterministic, repeated calls -- and a forward calculation versus an influence
matrix -- give bit-identical results.
"""

import array_api_compat
import torch

from pyRadPlan.ai.modelhub import BasePreprocessor
from pyRadPlan.core.xp_utils import to_namespace


class DummyDoseCalcPreprocessor(BasePreprocessor):
    """CT+energy -> network input; forward pass -> {physical_dose, physical_dose_var, let_dose}."""

    def preprocess(self, inputs: dict) -> torch.Tensor:
        """Assemble the (ct, energy-map) network input from CT cuboid + energy."""
        cfg = self.config
        ct_cuboid = inputs["ct_cuboid"]
        energy = inputs["energy"]

        clip_lo, clip_hi = cfg["ct_clip_range"]
        xp = array_api_compat.array_namespace(ct_cuboid)
        # the clip range is normalized onto [0, 1]
        ct_norm = (xp.clip(ct_cuboid, clip_lo, clip_hi) - clip_lo) / (clip_hi - clip_lo)
        energy_norm = energy / cfg["max_energy"]

        ct_t = to_namespace("torch", ct_norm).float().detach()
        e_t = to_namespace("torch", energy_norm).float().detach()
        if e_t.ndim == 0:
            e_t = e_t.view(1)

        batch_size = e_t.shape[0]
        depth, height, width = ct_t.shape[-3:]
        ct_map = ct_t.view(1, 1, depth, height, width).expand(batch_size, 1, depth, height, width)
        energy_map = e_t.view(-1, 1, 1, 1, 1).expand(-1, 1, depth, height, width)
        return torch.cat((ct_map, energy_map), dim=1)

    def postprocess(self, outputs: torch.Tensor) -> dict:
        """Rescale the network output to physical dose/LET and derive the dose variance."""
        max_dose = self.config["max_dose"]
        physical_dose = outputs[:, 0, ...] * max_dose
        return {
            "physical_dose": physical_dose,
            "physical_dose_var": (self.config["dose_var_rel"] * physical_dose) ** 2,
            "let_dose": outputs[:, 1, ...] * self.config["max_let"],
        }

    def predict(self, model: torch.nn.Module, model_input: torch.Tensor) -> dict:
        """Single deterministic forward pass.

        Assumes ``model_input`` is already preprocessed (i.e. :meth:`preprocess` was
        called beforehand).
        """
        model_input = model_input.to(next(model.parameters()).device)
        with torch.inference_mode():
            return self.postprocess(model(model_input))
