"""Preprocessor for the dummy dose-calc model (dose + physical_dose_var + LET).

Builds a ``(batch, 2, depth, lateral, lateral)`` network input from a CT
cuboid and per-bixel energy, runs an ensemble of dropout-active forward
passes (MC sampling), and reports the ensemble mean dose/LET plus the
predictive variance of the dose channel across the ensemble.
"""

import array_api_compat
import torch

from pyRadPlan.ml import BasePreprocessor
from pyRadPlan.core.xp_utils import to_namespace


class DummyDoseCalcPreprocessor(BasePreprocessor):
    """CT+energy -> network input; MC-dropout ensemble -> {physical_dose, physical_dose_var, let_dose}."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._ensemble_seeds = None

    def _get_ensemble_seeds(self) -> list:
        """Per-member seeds for the ensemble, generated once and cached."""
        if self._ensemble_seeds is None:
            torch.manual_seed(self.config.get("ensemble_seed", 42))
            ensemble_size = self.config.get("ensemble_size", 8)
            self._ensemble_seeds = torch.randint(0, 2**31 - 1, (ensemble_size,)).tolist()
        return self._ensemble_seeds

    def preprocess(self, inputs: dict) -> torch.Tensor:
        """Assemble the (ct, energy-map) network input from CT cuboid + energy."""
        cfg = self.config
        ct_cuboid = inputs["ct_cuboid"]
        energy = inputs["energy"]

        clip_lo, clip_hi = cfg["ct_clip_range"]
        xp = array_api_compat.array_namespace(ct_cuboid)
        ct_norm = (xp.clip(ct_cuboid, clip_lo, clip_hi) + cfg["ct_offset"]) / cfg["ct_scale"]
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

    def postprocess(self, outputs: dict) -> dict:
        """Rescale ensemble statistics back to physical dose/LET/physical_dose_var."""
        max_dose = self.config["max_dose"]
        max_let = self.config["max_let"]
        mean = outputs["mean"]
        dose_variance = outputs["dose_variance"]
        return {
            "physical_dose": mean[:, 0, ...] * max_dose,
            "physical_dose_var": dose_variance * (max_dose**2),
            "let_dose": mean[:, 1, ...] * max_let,
        }

    def predict(self, model: torch.nn.Module, model_input: torch.Tensor) -> dict:
        """Run an MC-dropout ensemble and return {physical_dose, physical_dose_var, let_dose}.

        Assumes ``model_input`` is already preprocessed (i.e. :meth:`preprocess` was
        called beforehand). Dropout is enabled only for the duration of the
        ensemble so the model's own eval/train state is left unchanged.
        """
        model_input = model_input.to(next(model.parameters()).device)

        ensemble_seeds = self._get_ensemble_seeds()
        cuda_available = torch.cuda.is_available()

        was_training = model.training
        model.train()
        try:
            preds = []
            with torch.inference_mode():
                for member_seed in ensemble_seeds:
                    torch.manual_seed(member_seed)
                    if cuda_available:
                        torch.cuda.manual_seed_all(member_seed)
                    preds.append(model(model_input))
        finally:
            model.train(was_training)

        preds_stack = torch.stack(preds, dim=0)
        mean = preds_stack.mean(dim=0)
        dose_variance = preds_stack[:, :, 0, ...].var(dim=0)

        return self.postprocess({"mean": mean, "dose_variance": dose_variance})
