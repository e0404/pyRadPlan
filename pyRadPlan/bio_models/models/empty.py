from ._base import BiologicalModelBase


class EmptyModel(BiologicalModelBase):
    """
    A model that implements the None model

    """

    model = "none"
    required_quantities = []  # Requires physical dose information
    possible_radiation_modes = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "VHEE",
    ]  # Compatible with all common modalities
    default_report_quantity = "physical_dose"  # Suggested quantity for display and planning

    def __init__(self, pln):
        super().__init__(pln)
