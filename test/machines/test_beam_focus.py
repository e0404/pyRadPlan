import numpy as np

from pyRadPlan.machines.particles._beam_focus import ChargedBeamFocus


def _emittance(n=None):
    scalar = {
        "type": "bigaussian",
        "sigmaX": 3.0,
        "sigmaY": 4.0,
        "divX": 1e-3,
        "divY": 2e-3,
        "corrX": 0.1,
        "corrY": -0.1,
    }
    if n is None:
        return scalar
    return {
        k: ([v] * n if k == "type" else np.full(n, v) * np.arange(1, n + 1))
        for k, v in scalar.items()
    }


def test_from_dict_without_emittance():
    focus = ChargedBeamFocus.from_dict(
        {"dist": [0.0, 100.0], "sigma": [2.0, 3.0], "SisFWHMAtIso": 5.0}
    )
    assert isinstance(focus, ChargedBeamFocus)
    assert focus.fwhm_iso == 5.0
    assert focus.emittance is None
    assert not focus.has_emittance


def test_from_dict_with_emittance():
    focus = ChargedBeamFocus.from_dict(
        {"dist": np.linspace(0, 100, 5), "sigma": np.ones(5), "emittance": _emittance()}
    )
    assert isinstance(focus, ChargedBeamFocus)
    assert focus.has_emittance
    assert focus.emittance.sigma_x == 3.0 and focus.emittance.corr_y == -0.1


def test_from_dict_multiple_foci():
    n = 3
    foci = ChargedBeamFocus.from_dict(
        {
            "dist": np.tile(np.linspace(0, 100, 5), (n, 1)),
            "sigma": np.ones((n, 5)) * np.arange(1, n + 1)[:, None],
            "SisFWHMAtIso": [5.0, 6.0, 7.0],
            "emittance": _emittance(n),
        }
    )
    assert isinstance(foci, list) and len(foci) == n
    for i, focus in enumerate(foci):
        assert focus.sigma[0] == i + 1
        assert focus.fwhm_iso == 5.0 + i
        assert focus.emittance.sigma_x == 3.0 * (i + 1)
