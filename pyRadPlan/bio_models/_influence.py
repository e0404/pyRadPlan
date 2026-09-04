"""Alpha/beta influence matrices derived from already computed influence matrices."""

from typing import Any

import numpy as np
from scipy import sparse

from ._evaluator import BioEvaluationContext, BioModelEvaluator


def alpha_beta_influence_from_let(
    evaluator: BioModelEvaluator,
    physical_dose: Any,
    let_dose: Any,
    alpha_x: Any,
    beta_x: Any,
) -> tuple[sparse.csc_array, sparse.csc_array]:
    """
    Compute ``alpha_dose`` / ``sqrt_beta_dose`` matrices from dose and LET·dose matrices.

    Evaluates an LET-based model entry-wise on the sparsity pattern of ``physical_dose``:
    the LET of an entry is ``let_dose / physical_dose``, the LQ parameters follow from the
    evaluator and the reference photon parameters of the entry's voxel.

    Parameters
    ----------
    evaluator : BioModelEvaluator
        Evaluator of a model with ``requires_let`` (e.g. Wedenberg, McNamara).
    physical_dose, let_dose : sparse matrix, shape (n_voxels, n_columns)
        Physical dose and LET-weighted dose influence matrices of one scenario.
    alpha_x, beta_x : array, shape (n_voxels,)
        Reference photon LQ parameters of the voxels (one CT scenario).

    Returns
    -------
    (alpha_dose, sqrt_beta_dose) : tuple of csc_array
    """
    dose = sparse.coo_array(physical_dose)
    keep = dose.data > 0
    rows, cols, d = dose.row[keep], dose.col[keep], dose.data[keep]

    let = np.asarray(sparse.csr_array(let_dose)[rows, cols]).ravel() / d
    result = evaluator.evaluate(
        BioEvaluationContext(
            {
                "alpha_x": np.asarray(alpha_x).ravel()[rows],
                "beta_x": np.asarray(beta_x).ravel()[rows],
                "physical_dose": d,
                "let": let,
            }
        )
    )
    alpha = result.require("alpha")
    beta = result.require("beta")
    alpha = np.broadcast_to(np.asarray(alpha, dtype=d.dtype), d.shape)
    beta = np.broadcast_to(np.asarray(beta, dtype=d.dtype), d.shape)

    shape = physical_dose.shape
    alpha_dose = sparse.csc_array((d * alpha, (rows, cols)), shape=shape)
    sqrt_beta_dose = sparse.csc_array((d * np.sqrt(beta), (rows, cols)), shape=shape)
    return alpha_dose, sqrt_beta_dose
