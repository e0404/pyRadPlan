.. _concept_quantities:

Quantity System
===============

Radiotherapy optimization requires more than just physical dose. RBE-weighted dose, biological
effect, and LET-weighted dose all depend on combinations of raw dose matrices stored in the
:class:`~pyRadPlan.dij.Dij` object. pyRadPlan's *quantity system* manages these dependencies
automatically, resolving which matrices need to be combined and caching intermediate results.

Motivation
----------

Consider RBE-weighted dose in the LQ model:

.. math::

    d_\text{RBE} = \frac{-\alpha_x + \sqrt{\alpha_x^2 + 4\beta_x \cdot E}}{2\beta_x}

where :math:`E = \alpha \cdot w + \beta \cdot w^2` is the biological effect and
:math:`\alpha`, :math:`\beta` are dose-influence matrices stored inside ``dij``. Computing
this by hand for every optimizer iteration — including correct gradient propagation — is
error-prone and backend-specific. The quantity system handles all of this transparently.

Available quantities
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Class
     - Identifier
     - Description
   * - :class:`~pyRadPlan.quantities.Dose`
     - ``"physical_dose"``
     - Physical absorbed dose in Gy.
   * - :class:`~pyRadPlan.quantities.DoseWeightedLET`
     - ``"let"``
     - Dose-averaged LET, derived from LET-weighted dose and physical dose.
   * - :class:`~pyRadPlan.quantities.LETxDose`
     - ``"let_dose"``
     - LET-weighted dose matrix.
   * - :class:`~pyRadPlan.quantities.AlphaDose`
     - ``"alpha_dose"``
     - Alpha-weighted dose for LQ model.
   * - :class:`~pyRadPlan.quantities.SqrtBetaDose`
     - ``"sqrt_beta_dose"``
     - Square-root-beta-weighted dose for LQ model.
   * - :class:`~pyRadPlan.quantities.Effect`
     - ``"effect"``
     - Biological effect :math:`E = \alpha D + \beta D^2` (LQ model).
   * - :class:`~pyRadPlan.quantities.RBExDose`
     - ``"rbe_x_dose"``
     - RBE-weighted dose in Gy(RBE). Derived from ``effect``, ``alpha_dose``,
       ``sqrt_beta_dose``.

Selecting a quantity for optimization
--------------------------------------

Objectives reference a quantity by identifier. The identifier is resolved against the
quantity registry and the available matrices in ``dij``:

.. code-block:: python

    from pyRadPlan.optimization.objectives import SquaredDeviation

    # Optimize on physical dose (default)
    obj = SquaredDeviation(priority=1000, d_ref=60.0, quantity="physical_dose")

    # Optimize on RBE-weighted dose (requires alpha_dose and sqrt_beta_dose in dij)
    obj = SquaredDeviation(priority=1000, d_ref=60.0, quantity="rbe_x_dose")

Resolution modes
----------------

Each :class:`~pyRadPlan.quantities.FluenceDependentQuantity` checks whether its corresponding
matrix is already present in ``dij`` (``"direct"`` mode) or must be derived from other
quantities (``"indirect"`` mode):

**Direct mode**
    The quantity matrix (e.g. ``dij.physical_dose``) exists and can be multiplied by the
    fluence vector directly. Fastest path.

**Indirect mode**
    The quantity is not pre-computed but can be derived from other available matrices
    (e.g. ``rbe_x_dose`` is computed from ``alpha_dose``, ``sqrt_beta_dose``, and reference
    tissue parameters α\ :sub:`x` / β\ :sub:`x`). The resolver chains quantity computations
    in the correct dependency order.

The QuantityResolver
--------------------

:class:`~pyRadPlan.quantities.QuantityResolver` is the component that binds a set of
quantities to a specific ``dij``. During optimization, the planning problem instantiates a
resolver and calls it by identifier:

.. code-block:: python

    from pyRadPlan.quantities import QuantityResolver

    resolver = QuantityResolver(dij)

    # Compute physical dose for a given fluence vector
    dose = resolver.get("physical_dose")(fluence)

    # Compute RBE-weighted dose (resolver handles the dependency chain automatically)
    rbe_dose = resolver.get("rbe_x_dose")(fluence)

Gradients and the Array API
-----------------------------

Quantity objects also expose ``compute_chain_derivative()`` for propagating derivatives back
to the fluence vector. Both forward evaluation and derivative propagation follow the Python
Array API standard, so operations can run on NumPy, CuPy, or PyTorch arrays depending on the
configured backend and installed optional dependencies.

Adding custom quantities
------------------------

New quantities can be added by subclassing
:class:`~pyRadPlan.quantities.FluenceDependentQuantity` and registering the class in the
``pyRadPlan.quantities.QUANTITIES`` dictionary. A custom quantity must implement the
single-scenario forward calculation and chain derivative methods used by the optimization
problem; use the existing quantity classes in ``pyRadPlan/quantities`` as templates.
