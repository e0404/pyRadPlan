import numpy as np

from scipy.sparse import hstack
from ._dij import Dij


def compose_beam_dijs(dijs: list[Dij]) -> Dij:
    """Compose multiple Dij objects into a single Dij object.

    Parameters
    ----------
    dijs : list[Dij]
        List of Dij objects to be composed.

    Returns
    -------
    Dij
        Composed Dij object.
    """

    qs = dijs[0].quantities

    num_of_beams = 0
    beam_num = np.array([], dtype=int)
    ray_num = np.array([], dtype=int)
    bixel_num = np.array([], dtype=int)

    for i, dij in enumerate(dijs):
        beam_num = np.append(beam_num, dij.beam_num + num_of_beams)
        ray_num = np.append(ray_num, dij.ray_num)
        bixel_num = np.append(bixel_num, dij.bixel_num)
        num_of_beams = num_of_beams + dij.num_of_beams

    matrices = {}
    for q in qs:
        tmp_matrices = [getattr(dij, q) for dij in dijs]

        new_matrix = np.empty_like(tmp_matrices[0], dtype=object)

        # Slots sharing their matrices (alias scenarios) keep sharing the composed one.
        composed = {}
        for i, scen_matrix in enumerate(tmp_matrices[0].flat):
            if scen_matrix is not None:
                key = tuple(id(tmp_matrix.flat[i]) for tmp_matrix in tmp_matrices)
                if key not in composed:
                    composed[key] = hstack([tmp_matrix.flat[i] for tmp_matrix in tmp_matrices])
                new_matrix.flat[i] = composed[key]

        matrices.update({q: new_matrix})

    return Dij(
        ct_grid=dijs[0].ct_grid,
        dose_grid=dijs[0].dose_grid,
        matrices=matrices,
        num_of_beams=num_of_beams,
        beam_num=beam_num,
        ray_num=ray_num,
        bixel_num=bixel_num,
        scenarios=dijs[0].scenarios,
        **matrices,
    )
