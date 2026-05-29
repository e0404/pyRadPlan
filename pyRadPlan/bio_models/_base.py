from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional
import warnings
from pyRadPlan.plan import Plan, validate_pln

from .empty import EmptyModel


class BiologicalModelBase(ABC):
    """
    Abstract base class for biological models used in dose calculation and
    plan optimisation.

    """

    model: ClassVar[str]
    required_quantities: ClassVar[
        list[str]
    ]  # kernels in base data needed for the alpha/beta calculation
    possible_radiation_modes: ClassVar[
        list[str]
    ]  # radiation modalitites compatible with the model
    default_report_quantity: ClassVar[
        str
    ]  # default suggested quantity to use for display and planning

    _quantity_opt: Optional[Any] = None
    _quantity_vis: Optional[Any] = None

    def __init__(self, pln: Plan):
        if pln is not None:
            pln = validate_pln(pln)
            # crerate bio model based on what is save din the pln.bioModel

    @abstractmethod
    def calc_biological_quantities_for_bixel(self, bixel: Any) -> Any:
        raise NotImplementedError(
            "Method '_calc_biological_quantities_for_bixel' must be implemented."
        )

    def is_available(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """
        Check whether this model is compatible with the given radiation mode
        and the quantities supplied by the dose engine / machine dataset.

        Parameters
        ----------
        radiation_mode:
            Radiation modality string (e.g. ``"photons"``, ``"protons"``).
        provided_quantities:
            List of quantity names available from the dose engine or machine.

        Returns
        -------
        (available, message)
            ``available`` is ``True`` only when both the radiation mode and
            all required quantities are satisfied.
        """
        messages: list[str] = []

        valid_rad_mode = radiation_mode in self.possible_radiation_modes
        if not valid_rad_mode:
            messages.append(
                f"Radiation mode '{radiation_mode}' is invalid for model '{self.model}'."
            )

        valid_quantities = True
        if provided_quantities is not None and self.required_quantities:
            missing = [q for q in self.required_quantities if q not in provided_quantities]
            if missing:
                valid_quantities = False
                messages.append(
                    "Required quantities not provided by Dose Engine or selected Machine Dataset."
                )

        return (valid_rad_mode and valid_quantities), ", ".join(messages)

    @staticmethod
    def get_available_models(
        radiation_mode: Optional[str] = None,
        provided_quantities: Optional[list[str]] = None,
    ) -> list[BiologicalModelBase]:
        """
        Discover all concrete subclasses of :class:`BiologicalModel` and,
        optionally, filter them by radiation mode and available quantities.

        Parameters
        ----------
        radiation_mode:
            When given, only models that declare this mode as compatible
            are returned.
        provided_quantities:
            When given together with *radiation_mode*, models whose
            ``required_quantities`` are not satisfied are also excluded.

        Returns
        -------
        list[BiologicalModelBase]
            List of available model instances.

        Raises
        ------
        RuntimeError
            When no models are found at all.
        """
        subclasses = BiologicalModelBase._get_all_subclasses(BiologicalModelBase)

        class_list: list[BiologicalModelBase] = []
        for cls in subclasses:
            if not hasattr(cls, "model"):
                continue
            class_list.append(cls())

        if not class_list:
            class_list.append(BiologicalModelBase(model="none"))

        if radiation_mode is not None:
            filtered: list[BiologicalModelBase] = []
            for info in class_list:
                instance = info.handle()
                call_kwargs: dict[str, Any] = {"radiation_mode": radiation_mode}
                if provided_quantities is not None:
                    call_kwargs["provided_quantities"] = provided_quantities
                available, _ = instance.is_available(**call_kwargs)
                if available:
                    filtered.append(info)
            class_list = filtered

        if not class_list:
            raise RuntimeError(
                "No BiologicalModelBase subclasses found that satisfy the radiation mode"
            )

        return class_list

    @staticmethod
    def create(model_metadata: Any) -> BiologicalModelBase:
        """
        Factory: build a BiologicalModelBase instance from a model name,
        a dict / Pydantic model, or an existing instance.

        Parameters
        ----------
        model_metadata:
            * An existing :class:`BiologicalModelBase` instance – returned as-is.
            * A plain string – treated as the model name.
            * A ``dict`` or Pydantic model with at least a ``"model"`` key.

        Returns
        -------
        BiologicalModelBase
            A freshly constructed (or forwarded) model instance.
        """
        if isinstance(model_metadata, BiologicalModelBase):
            return model_metadata

        if isinstance(model_metadata, str):
            model_metadata = {"model": model_metadata}
        elif hasattr(model_metadata, "model_dump"):
            model_metadata = model_metadata.model_dump()
        elif not isinstance(model_metadata, dict):
            model_metadata = dict(model_metadata)

        model_metadata = dict(model_metadata)  # ensure mutable copy

        try:
            class_list = BiologicalModelBase.get_available_models()
        except RuntimeError:
            warnings.warn(
                "Biological Model not found, creating Empty Model!",
                UserWarning,
                stacklevel=2,
            )
            return EmptyModel()

        model_names = [info.model for info in class_list]
        model_name = model_metadata.get("model", "")

        if model_name not in model_names:
            warnings.warn(
                "Biological Model not found, creating Empty Model!",
                UserWarning,
                stacklevel=2,
            )
            return EmptyModel()

        idx = model_names.index(model_name)
        cls = class_list[idx].handle

        model_metadata.pop("model", None)

        instance = cls()
        for field, value in model_metadata.items():
            if hasattr(instance, field):
                setattr(instance, field, value)
            else:
                warnings.warn(
                    f"Not able to assign property '{field}' from metadata to Biological Model.",
                    UserWarning,
                    stacklevel=2,
                )

        return instance

    @staticmethod
    def validate_model(
        model: Any,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> BiologicalModelBase:
        """
        Create and validate a biological model.

        Parameters
        ----------
        model:
            Model name, metadata dict, or existing instance.
        radiation_mode:
            Must be in the model's ``possible_radiation_modes``.
        provided_quantities:
            Must cover all entries in ``required_quantities``.

        Returns
        -------
        BiologicalModelBase
            A validated model instance.

        Raises
        ------
        ValueError
            When the model is not valid for the given configuration.
        """
        instance = BiologicalModelBase.create(model)

        if provided_quantities is not None:
            valid, msg = instance.is_available(radiation_mode, provided_quantities)
        else:
            valid, msg = instance.is_available(radiation_mode)

        if not valid:
            raise ValueError(f"Biological Model not valid: {msg}")

        return instance
