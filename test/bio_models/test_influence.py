import numpy as np
from scipy import sparse

from pyRadPlan.bio_models import Wedenberg, alpha_beta_influence_from_let


def test_alpha_beta_influence_from_let_matches_model():
    rng = np.random.default_rng(1)
    n_vox, n_bix = 40, 7
    dose = sparse.random_array((n_vox, n_bix), density=0.3, rng=rng, dtype=np.float64)
    let = sparse.csc_array(dose.multiply(sparse.random_array(dose.shape, density=1.0, rng=rng)))
    let_dose = sparse.csc_array(dose.multiply(2.0 + 5.0 * let.toarray() / let.max()))
    alpha_x = rng.uniform(0.05, 0.5, n_vox)
    beta_x = rng.uniform(0.01, 0.1, n_vox)

    model = Wedenberg()
    alpha_dose, sqrt_beta_dose = alpha_beta_influence_from_let(
        model.evaluator(None, {}), dose, let_dose, alpha_x, beta_x
    )
    assert sparse.issparse(alpha_dose) and alpha_dose.shape == dose.shape

    d = dose.toarray()
    nz = d > 0
    let_entries = np.zeros_like(d)
    let_entries[nz] = let_dose.toarray()[nz] / d[nz]
    rbe_min, rbe_max = model.rbe_min_max(let_entries, alpha_x[:, None], beta_x[:, None])
    assert np.allclose(alpha_dose.toarray(), np.where(nz, d * rbe_max * alpha_x[:, None], 0))
    assert np.allclose(
        sqrt_beta_dose.toarray(), np.where(nz, d * np.sqrt(rbe_min**2 * beta_x[:, None]), 0)
    )
