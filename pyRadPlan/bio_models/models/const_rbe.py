"""Constant relative biological effectiveness (RBE) model."""

from pyRadPlan.bio_models._base import BiologicalModelBase


class ConstantRBEModel(BiologicalModelBase):
    """
    Biological model applying a single, user-configurable RBE to all bixels.

    The relative biological effectiveness (RBE) is assumed to be spatially
    uniform and independent of dose, LET, or tissue type. Alpha and beta
    values are scaled from their reference (photon) counterparts by the
    constant RBE, making the model compatible with RBE-weighted optimisation
    frameworks that operate on alpha/beta directly.

    Parameters
    ----------
    rbe : float, optional
        Constant RBE factor applied to all bixels. Defaults to ``1.1``,
        the clinically adopted value for proton therapy.

    """

    model = "constant_rbe"
    required_quantities = ["physical_dose"]  # Requires physical dose information
    possible_radiation_modes = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "oxygen",
        "VHEE",
    ]  # Compatible with all common modalities
    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning

    def __init__(self):
        self.rbe = 1.1  # Default RBE value, can be overridden by user input
        super().__init__()

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """
        Calculate the RBE-weighted dose for a given bixel using the constant RBE value.

        Parameters
        ----------
        bixel:
            A data structure representing a bixel, which should contain at least
            the physical dose information.
        kernels:
            A dictionary containing the kernel values, including LET.

        Returns
        -------
        dict[str, float]
            A dictionary containing the RBE-weighted dose calculated as:
            RBE-weighted dose = physical dose * constant RBE value.
        """
        bixel["alpha"] = self.rbe * bixel["v_alpha_x"]
        bixel["beta"] = (
            self.rbe**2 * bixel["v_beta_x"]
        )  # do i want this, not necessary for the constant RBE model, but would then work with the current RBE optimization
        return bixel
