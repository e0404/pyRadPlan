from pyRadPlan.bio_models._base import BiologicalModelBase


class ConstantRBEModel(BiologicalModelBase):
    """
    A simple model that assumes a constant RBE value.

    """

    model = "constant_rbe"
    required_quantities = ["physical_dose"]  # Requires physical dose information
    possible_radiation_modes = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "VHEE",
    ]  # Compatible with all common modalities
    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning

    def __init__(self, pln):
        self.rbe = 1.1  # Default RBE value, can be overridden by user input
        super().__init__(pln)

    def calc_biological_quantities_for_bixel(self, bixel: dict) -> dict:
        """
        Calculate the RBE-weighted dose for a given bixel using the constant RBE value.

        Parameters
        ----------
        bixel:
            A data structure representing a bixel, which should contain at least
            the physical dose information.

        Returns
        -------
        dict[str, float]
            A dictionary containing the RBE-weighted dose calculated as:
            RBE-weighted dose = physical dose * constant RBE value.
        """
        bixel["alpha"] = self.rbe * bixel["_v_alpha_x"]
        bixel["beta"] = (
            self.rbe**2 * bixel["_v_beta_x"]
        )  # do i want this, not necessary for the constant RBE model, but would then work with the current RBE optimization
        return bixel
