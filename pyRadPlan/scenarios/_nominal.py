"""
Nominal Scenario Model.

This module contains the NominalScenario class, which represents an example scenario.

Example
-------
    $ python -m pyRadPlan.scenarios._nominal

    # Usage Example
    # scenario = NominalScenario()
    # scenario.list_all_scenarios()
"""

from typing import ClassVar

import warnings
import numpy as np
from pyRadPlan.scenarios import ScenarioModel


# Example subclass
class NominalScenario(ScenarioModel):
    """
    Class representing a nominal (example) scenario.

    Attributes
    ----------
    name : str
        The name of the scenario.
    short_name : str
        The short name of the scenario.

    Methods
    -------
    update_scenarios()
        Updates the scenarios.
    extract_single_scenario(scen_num: int) -> ScenarioModel
        Extracts a single scenario.
    """

    name: ClassVar[str] = "Nominal Scenario"
    short_name: ClassVar[str] = "nomScen"

    @property
    def ndim(self) -> int:
        """Dimensionality of the scenario model."""
        return 1

    def update_scenarios(self) -> np.ndarray[float]:
        num_of_ct_scen = len(self.ct_scen_prob)
        self._num_of_available_ct_scen = num_of_ct_scen

        # Scenario weight
        self._scen_weight = np.ones(num_of_ct_scen).astype(float) / float(num_of_ct_scen)
        self._scen_weight = [prob[1] for prob in self.ct_scen_prob]
        self._ct_scen_ix = [prob[0] for prob in self.ct_scen_prob]

        # set variables
        self._tot_num_shift_scen = 1
        self._tot_num_range_scen = 1
        self._tot_num_scen = num_of_ct_scen

        # Individual shifts
        self._rel_range_shift = np.zeros(num_of_ct_scen)
        self._abs_range_shift = np.zeros(num_of_ct_scen)
        self._iso_shift = np.zeros((num_of_ct_scen, 3))

        # Probability matrices
        self._scen_for_prob = np.hstack(
            (np.array(self.ct_scen_prob)[:, 0].reshape(-1, 1), np.zeros((num_of_ct_scen, 5)))
        )  # Realization matrix
        self._scen_prob = np.array(self.ct_scen_prob)[:, 1]  # Probabilities for each scenario

        # Mask for scenario selection
        self._scen_mask = np.zeros(
            (self._num_of_available_ct_scen, self._tot_num_shift_scen, self._tot_num_range_scen),
            dtype=bool,
        )
        self._scen_mask[self._ct_scen_ix, :, :] = True

        # generic code
        x = np.where(self._scen_mask)
        self._linear_mask = np.column_stack(x)
        tot_num_scen = np.sum(self._scen_mask)

        # Get Scenario probability
        covariance = np.diag(
            np.hstack((np.array(self.shift_sd), self.range_abs_sd, self.range_rel_sd / 100.0)) ** 2
        )
        d = float(covariance.shape[0])
        cs = np.linalg.cholesky(covariance)
        tmp_standardized = np.linalg.solve(cs, self._scen_for_prob[:, 1:].T).T
        tmp_scen_prob = (
            (2 * np.pi) ** (-d / 2)
            * np.exp(-0.5 * np.sum(tmp_standardized**2, axis=1))
            / np.prod(np.diag(cs))
        )

        # Multiply with 4D phase probability
        self._scen_prob = self._scen_weight * tmp_scen_prob

        # Get relative (normalized) weight of the scenario
        self._scen_weight = self._scen_prob / np.sum(self._scen_prob)

        # Return variable
        scenarios = self._scen_for_prob

        if int(tot_num_scen) != self._tot_num_scen:
            warnings.warn(
                (
                    "Check Implementation of Total Scenario computation - "
                    f"given {self._tot_num_scen} but found {tot_num_scen}!"
                )
            )
            self._tot_num_scen = tot_num_scen

        return scenarios

    # TODO: Finalize implementation to extract according to ct number
    def extract_single_scenario(self, scen_num: int) -> "ScenarioModel":
        # Example logic
        if scen_num != 0:
            raise NotImplementedError(
                "Currently the Nominal Scenario Model is only implemented for single ct scenarios!"
            )
        return self
