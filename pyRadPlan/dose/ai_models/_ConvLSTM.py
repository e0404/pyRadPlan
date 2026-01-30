"""ConvLSTM-based dose prediction model (torch)."""

try:
    import torch
except ImportError:
    torch = None
import json
from pathlib import Path
from ._base_ai_models import AIDoseModelBase
import array_api_compat
from typing import Any, Union
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.core.xp_utils import to_namespace

# from pyRadPlan.core.xp_utils.helpers import to_namespace
from pyRadPlan.data.ai_models.ConvLSTM_pyRadPlan import ConvBayes_new


class predict_ConvLSTM(AIDoseModelBase):
    """ConvBayes LSTM-based dose predictor (torch).

    Overview
    - Loads a pretrained ConvBayes model and runs inference on a
      (depth, lateral, lateral) input grid augmented with an energy channel.
    - Handles array-API interop via DLPack for minimal copies.
    - Applies simple CT and energy normalizations and reverses output scaling.
    """

    model_name = "ConvLSTM"
    possible_radiation_modes = ["protons"]
    HAS_BATCHED_FORWARD = True

    def __init__(self, pln: Any | None = None) -> None:
        """Initialize model, device, and load pretrained weights.

        Parameters
        ----------
        pln : Any, optional
            Plan-like object forwarded to the base class.
        """
        super().__init__(pln)

        # Load configuration from JSON
        config_path = (
            Path(__file__).parent.parent.parent / "data" / "ai_models" / "ConvLSTM_config.json"
        )
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Extract parameters from config
        arch_config = self.config["model_architecture"]
        norm_config = self.config["normalization"]
        model_path = self.config["model_path"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ConvBayes_new(
            arch_config["batch_size"],
            arch_config["n_steps"],
            arch_config["imsize"],
            projection_dim=arch_config["projection_dim"],
            energy_encoding=arch_config["energy_encoding"],
            reduced_model=arch_config["reduced_model"],
        ).to(self.device)
        self.network.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=self.device)
        )
        self.network.eval()

        self.imsize = self.network.imsize
        self.n_step = self.network.n_step
        self.N = 1
        self.max_dose = norm_config["max_dose"]
        self.max_energy = norm_config["max_energy"]
        self.USE_BATCH = self.HAS_BATCHED_FORWARD

    def _forward_pass(
        self,
        ct_cuboid: Union[torch.Tensor, Array],
        energy: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """Run a single forward pass on the torch model.

        Parameters
        ----------
        ct_cuboid : torch.Tensor or DLPack-capable Array
            Depth-first 3D grid (depth, lateral, lateral) after normalization.
        energy : torch.Tensor or DLPack-capable Array
            Normalized energy value(s) to encode as an additional channel.

        Returns
        -------
        pred_t : torch.Tensor
            Predicted dose cuboid with shape (depth, lateral, lateral) on `self.device`.
        """
        # Move ct_cuboid to torch
        xp = array_api_compat.array_namespace(ct_cuboid)

        inp_img = to_namespace("torch", ct_cuboid).float().detach()  # type: ignore[arg-type]
        if inp_img.device.type != "cuda" and torch.cuda.is_available():
            inp_img = inp_img.to(self.device, dtype=torch.float32)

        energy_t = torch.tensor(energy).float().detach()
        if energy_t.device.type != "cuda" and torch.cuda.is_available():
            energy_t = energy_t.to(self.device, dtype=torch.float32)

        # Build energy channel and concatenate on channel dimension
        energy_map = energy_t.view(1, 1, 1).repeat(1, inp_img.shape[1], inp_img.shape[2])
        inp_img = torch.cat((energy_map, inp_img)).float()

        # Forward pass without grad, then convert back to original namespace
        with torch.inference_mode():
            out = self.network(inp_img)
            pred_t = out[0, :, :, :].detach()

        return to_namespace(xp, pred_t)

    def _forward_pass_batch(
        self,
        ct_cuboid: Union[torch.Tensor, Array],
        energy: Union[torch.Tensor, Any],
    ) -> torch.Tensor:
        """Run a single forward pass on the torch model.

        Parameters
        ----------
        ct_cuboid : torch.Tensor or DLPack-capable Array
            Depth-first 3D grid (depth, lateral, lateral) after normalization.
        energy : torch.Tensor or DLPack-capable Array
            Normalized energy value(s) to encode as an additional channel.

        Returns
        -------
        pred_t : torch.Tensor
            Predicted dose cuboid with shape (depth, lateral, lateral) on `self.device`.
        """

        xp = array_api_compat.array_namespace(ct_cuboid)
        B = int(energy.shape[0])
        t_single = to_namespace("torch", ct_cuboid).float().detach()

        if t_single.device.type != "cuda" and torch.cuda.is_available():
            t_single = t_single.to(self.device, dtype=torch.float32)

        t = t_single.expand(B, -1, -1, -1) if B > 1 else t_single

        e_t = to_namespace("torch", energy).float().detach()
        if e_t.device.type != "cuda" and torch.cuda.is_available():
            e_t = e_t.to(self.device, dtype=torch.float32)

        if e_t.ndim == 0:
            e_t = e_t.view(1)

        # If batch is singleton, duplicate to avoid 0-D energy token inside network
        duplicated_singleton = False
        if B == 1:
            duplicated_singleton = True
            t = t.repeat(2, 1, 1, 1)
            e_t = e_t.expand(2)
            B = 2
        # Build energy channel of shape (B,1,H,W) and concatenate to channel dim
        # Current `t` shape: (B, D, H, W) -> treat D as channel dim
        H = t.shape[-2]
        W = t.shape[-1]
        energy_map = e_t.view(-1, 1, 1, 1).expand(-1, 1, H, W)

        # Concatenate energy channel as additional channel at dim=1
        inp = torch.cat((energy_map, t), dim=1)

        # Forward pass in inference mode
        with torch.inference_mode():
            pred_t = self.network(inp)

        if duplicated_singleton:
            pred_t = pred_t[:1]

        result = to_namespace(xp, pred_t)

        del t, t_single, H, W, e_t, energy_map, inp, pred_t

        return result

    def _undo_dose_normalization(self, dose: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """Scale model output back to physical dose range.

        Parameters
        ----------
        dose : torch.Tensor or Array
            Model prediction in normalized units.

        Returns
        -------
        dose_norm : same type as input when possible
            De-normalized dose values.
        """
        dose_norm = dose * self.max_dose
        return dose_norm

    def _ct_normalization(self, ct_cuboid: Array) -> Array:
        """Normalize CT values to [0, 1] after clipping.

        Parameters
        ----------
        ct_cuboid : Array
            CT HU values on (depth, lateral, lateral) grid.

        Returns
        -------
        ct_norm : Array
            Normalized CT values with the same array namespace as input.
        """
        xp = array_api_compat.array_namespace(ct_cuboid)
        inp_data_to_normalize = xp.clip(
            ct_cuboid, -1000.0, 3000.0
        )  # Clip values to the range [-1000, 3000]

        inp_data_norm = (inp_data_to_normalize + 1000) / 4000
        return inp_data_norm

    def _energy_normalization(self, energy: Union[float, Any]) -> Union[float, Any]:
        """Normalize energy to [0, 1] range using `self.max_energy`.

        Parameters
        ----------
        energy : float or Array
            Nominal beamlet energy.

        Returns
        -------
        energy_norm : same type as input when possible
            Normalized energy values.
        """
        energy_norm = energy / self.max_energy
        return energy_norm

    def _predict_dose(
        self,
        ct_cuboid: Array,
        energy: Union[float, Any],
        grid_information: dict,
    ) -> Array:
        """Predict a dose cuboid on the provided regular local grid.

        Parameters
        ----------
        ct_cuboid : Array
            CT values on (depth, lateral, lateral) grid in the engine's array API.
        energy : float or Array
            Beamlet energy value used for energy-channel conditioning.
        grid_information : dict
            Grid metadata; not directly required here but kept for interface consistency.

        Returns
        -------
        dose : Array
            Predicted physical dose cuboid with shape (depth, lateral, lateral),
            converted back to the input array namespace and dtype when feasible.
        """
        # Get original array namespace to convert back after inference
        xp = array_api_compat.get_namespace(ct_cuboid)

        ct_norm = self._ct_normalization(ct_cuboid)
        energy_norm = self._energy_normalization(energy)

        if self.USE_BATCH:
            prediction = self._forward_pass_batch(ct_norm, energy_norm)
        else:
            prediction = self._forward_pass(ct_norm, energy_norm)

        del ct_norm, energy_norm
        # Match dtype of input if possible
        try:
            prediction = prediction.astype(ct_cuboid.dtype)
        except Exception:
            pass

        dose = self._undo_dose_normalization(prediction)

        del prediction

        return dose
