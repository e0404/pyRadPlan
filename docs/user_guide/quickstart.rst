.. _quickstart:

Quickstart: The first minimal treatment plan
============================================

Once pyRadPlan is installed (see :ref:`installation`), a basic treatment plan can be generated
with just a few lines of code — mirroring the experience of matRad's ``matRad.m`` example script.

.. code-block:: python

    from pyRadPlan import (
        load_tg119,
        IonPlan,
        generate_stf,
        calc_dose_influence,
        fluence_optimization,
        plot_slice,
    )

    # 1. Load patient data from the bundled TG119 phantom
    ct, cst = load_tg119()

    # 2. Create a plan: proton therapy on the Generic machine
    pln = IonPlan(radiation_mode="protons", machine="Generic")

    # 3. Generate beam geometry (steering information)
    stf = generate_stf(ct, cst, pln)

    # 4. Compute the dose influence matrix
    dij = calc_dose_influence(ct, cst, stf, pln)

    # 5. Optimize beamlet fluences
    fluence = fluence_optimization(ct, cst, stf, dij, pln)

    # 6. Reconstruct dose on the CT grid
    result = dij.compute_result_ct_grid(fluence)

    # 7. Visualize
    plot_slice(image_volume=ct, cst=cst, overlay=result["physical_dose"], overlay_unit="Gy")

Each step corresponds to a major concept described in :ref:`concepts`. The following sections
explain every object and function in detail.
