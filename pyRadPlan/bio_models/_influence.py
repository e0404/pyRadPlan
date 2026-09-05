"""Biological influence matrices derived from already computed influence matrices."""

from typing import Any

import numpy as np
from scipy import sparse

from ._evaluator import BioEvaluationContext, BioModelEvaluator, BioModelResult


def bio_influence_from_let(
    evaluator: BioModelEvaluator,
    physical_dose: Any,
    let_dose: Any,
    alpha_x: Any,
    beta_x: Any,
) -> BioModelResult:
    """
    Compute biological influence matrices from dose and LET·dose matrices.

    Evaluates an LET-based model entry-wise on the sparsity pattern of ``physical_dose``:
    the LET of an entry is ``let_dose / physical_dose``. The evaluator converts its model
    outputs into the additive quantities it declares through ``influence_quantity_names``.

    Parameters
    ----------
    evaluator : BioModelEvaluator
        Evaluator of a model requiring the ``"let"`` input.
    physical_dose, let_dose : sparse matrix, shape (n_voxels, n_columns)
        Physical dose and LET-weighted dose influence matrices of one scenario.
    alpha_x, beta_x : array, shape (n_voxels,)
        Reference photon LQ parameters of the voxels (one CT scenario).

    Returns
    -------
    BioModelResult
        One CSC influence matrix per declared influence quantity.
    """
    dose = sparse.coo_array(physical_dose)
    keep = dose.data > 0
    rows, cols, d = dose.row[keep], dose.col[keep], dose.data[keep]

    let = np.asarray(sparse.csr_array(let_dose)[rows, cols]).ravel() / d
    result = evaluator.evaluate_influence(
        BioEvaluationContext(
            {
                "alpha_x": np.asarray(alpha_x).ravel()[rows],
                "beta_x": np.asarray(beta_x).ravel()[rows],
                "physical_dose": d,
                "let": let,
            }
        )
    )
    shape = physical_dose.shape
    return BioModelResult(
        {
            name: sparse.csc_array(
                (np.broadcast_to(np.asarray(values, dtype=d.dtype), d.shape), (rows, cols)),
                shape=shape,
            )
            for name, values in result.items()
        }
    )
