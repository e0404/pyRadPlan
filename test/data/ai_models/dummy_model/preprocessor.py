"""Dummy preprocessor for pyRadPlan.ai.modelhub local-load tests."""

import numpy as np
import torch

from pyRadPlan.ai.modelhub import BasePreprocessor


class DummyPreprocessor(BasePreprocessor):
    """Stack modality volumes into a batched, channel-first tensor."""

    def preprocess(self, inputs: dict) -> torch.Tensor:
        type_order = self.config.get("type_order") or list(inputs.keys())
        channels = [np.asarray(inputs[name], dtype=np.float32) for name in type_order]
        stacked = np.stack(channels, axis=0)[np.newaxis, ...]
        return torch.from_numpy(stacked)
