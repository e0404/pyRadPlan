"""Loading of pyanno4rt outcome models.

pyanno4rt snapshots a trained outcome model as a bare pickled scikit-learn
estimator (``<model_path>/predictor.save``). Everything needed to *use* that
estimator -- which features it consumes, from which segments, and in which
order -- lives in the ``input.json`` of the treatment plan it was trained with.
:func:`load_pyanno4rt_model` reads that pair and returns the estimator together
with a declarative :class:`OutcomeModelSpec`.

Feature order is the subtle part: pyanno4rt re-sorts the feature columns listed
in ``input.json`` into the column order of the training table before fitting, so
the estimator's coefficients are bound to *that* order and not to the order the
columns appear in the JSON. This is reproduced by reading the header of
``dataset.data_path`` when it is reachable; when it is not, the JSON order is
used and a warning is emitted, because a wrong order binds the coefficients to
the wrong features without any other symptom.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


logger = logging.getLogger(__name__)

#: Predictor file names understood by this loader, in order of preference.
_PREDICTOR_SUFFIXES = (".save", ".save", ".pkl", ".pickle", ".joblib")


@dataclass(frozen=True)
class OutcomeFeature:
    """A single input feature of a feature-based outcome model.

    Attributes
    ----------
    name : str
        Column name in the training table; used to recover the feature order.
    segment : str
        Name of the segment (VOI) the feature is computed on.
    function : str
        Feature function, e.g. ``"Dose Mean"`` or ``"Segment Volume"``.
    argument : Any
        Optional argument of the feature function (e.g. the level of a Dx).
    """

    name: str
    segment: str
    function: str
    argument: Any = None


@dataclass
class OutcomeModelSpec:
    """Everything needed to evaluate a pyanno4rt outcome model in pyRadPlan.

    Attributes
    ----------
    label : str
        Model label as given in the pyanno4rt configuration.
    outcome_type : str
        ``"NTCP"`` or ``"TCP"``.
    features : list[OutcomeFeature]
        Model inputs, in the order the estimator expects them.
    transform : bool
        Whether pyanno4rt embedded the model with its SALU objective transform.
    model_dir : Path or None
        Directory the predictor was loaded from.
    num_of_fractions : int
        Number of fractions pyanno4rt multiplied the dose with before computing
        the features (the model expects total-course dose).
    """

    label: str
    outcome_type: str = "NTCP"
    features: list[OutcomeFeature] = field(default_factory=list)
    transform: bool = False
    model_dir: Optional[Path] = None
    num_of_fractions: int = 30


def _outcome_components(config: dict) -> list[tuple[str, dict]]:
    """Return the ``(name, parameters)`` pairs of all outcome components."""
    components = config.get("optimization", {}).get("components", [])
    found = []
    for entry in components:
        if not isinstance(entry, dict):
            continue
        for name, params in entry.items():
            if isinstance(params, dict) and isinstance(params.get("model"), dict):
                found.append((name, params))
    return found


def _select_component(config: dict, component: Optional[str]) -> tuple[str, dict]:
    """Pick the requested outcome component, or the only one present."""
    found = _outcome_components(config)
    if not found:
        raise ValueError("The pyanno4rt configuration contains no outcome model component.")

    if component is not None:
        matches = [
            (name, params)
            for name, params in found
            if component in (name, params.get("identifier"), params["model"].get("label"))
        ]
        if not matches:
            available = [f"{name} ({params['model'].get('label')})" for name, params in found]
            raise ValueError(
                f"No outcome component matching '{component}'. Available: {available}"
            )
        return matches[0]

    if len(found) > 1:
        available = [f"{name} ({params['model'].get('label')})" for name, params in found]
        raise ValueError(
            f"The configuration holds {len(found)} outcome components; pass 'component' to "
            f"select one of {available}."
        )
    return found[0]


def _parse_features(model_cfg: dict) -> list[OutcomeFeature]:
    """Read the dynamic feature columns of a pyanno4rt dataset definition."""
    features = []
    for column in model_cfg.get("dataset", {}).get("columns", []):
        if not isinstance(column, dict):
            continue
        for kind, params in column.items():
            if kind != "Dynamic Feature":
                continue
            features.append(
                OutcomeFeature(
                    name=params["column"],
                    segment=params["segment"],
                    function=params["function"],
                    argument=params.get("argument"),
                )
            )
    if not features:
        raise ValueError("The outcome component defines no 'Dynamic Feature' columns.")
    return features


def _order_features(
    features: list[OutcomeFeature], data_path: Optional[str]
) -> list[OutcomeFeature]:
    """Sort features into training-table column order (see the module docstring)."""
    warning = (
        "Cannot verify the feature order of the outcome model (%s); falling back to the order "
        "given in the configuration. If it differs from the training table, the model's "
        "coefficients bind to the wrong features. Pass an explicit feature order or make the "
        "training table available to remove this ambiguity."
    )

    if not data_path:
        logger.warning(warning, "the configuration records no dataset path")
        return features

    path = Path(data_path)
    if not path.is_file():
        logger.warning(warning, f"'{path}' is not readable")
        return features

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration) as exc:
        logger.warning(warning, f"'{path}' could not be read ({exc})")
        return features

    order = {name: index for index, name in enumerate(header)}
    missing = [feature.name for feature in features if feature.name not in order]
    if missing:
        logger.warning(warning, f"columns {missing} are absent from '{path}'")
        return features

    ordered = sorted(features, key=lambda feature: order[feature.name])
    if ordered != features:
        logger.info(
            "Reordered outcome model features to training-table order: %s",
            [feature.name for feature in ordered],
        )
    return ordered


def _resolve_model_dir(model_cfg: dict, snapshot_dir: Path) -> Path:
    """Locate the directory holding the pickled predictor inside the snapshot."""
    label = model_cfg.get("label")
    raw = model_cfg.get("model_path")

    # 'model_path' is relative to the working directory pyanno4rt ran in, so only its last
    # part is reliable; the snapshot stores the predictor in '<snapshot>/<label>'.
    candidates = []
    if raw:
        candidates.append(snapshot_dir / Path(raw).name)
    if label:
        candidates.append(snapshot_dir / label)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not locate the model directory for '{label}'. Tried {[str(c) for c in candidates]}."
    )


def _find_predictor(model_dir: Path) -> Path:
    """Return the predictor file inside ``model_dir``."""
    for suffix in _PREDICTOR_SUFFIXES:
        candidate = model_dir / f"predictor{suffix}"
        if candidate.is_file():
            return candidate

    others = sorted(p.name for p in model_dir.glob("predictor.*"))
    if others:
        raise NotImplementedError(
            f"Model directory '{model_dir}' holds {others}, which is not a pickled scikit-learn "
            f"estimator. Only {list(_PREDICTOR_SUFFIXES)} predictors are supported."
        )
    raise FileNotFoundError(f"No predictor file found in '{model_dir}'.")


def _load_predictor(path: Path) -> Any:
    """Unpickle the estimator."""
    if path.suffix == ".joblib":
        import joblib  # noqa: PLC0415

        return joblib.load(path)

    with path.open("rb") as handle:
        return pickle.load(handle)


def load_pyanno4rt_model(
    snapshot_dir: Union[str, Path],
    component: Optional[str] = None,
) -> tuple[Any, OutcomeModelSpec]:
    """Load a pyanno4rt outcome model from its snapshot folder.

    Parameters
    ----------
    snapshot_dir : str or Path
        pyanno4rt snapshot folder, holding ``input.json`` and the model's
        ``<label>/predictor.*``.
    component : str, optional
        Name, identifier or model label of the outcome component to load.
        Required only when the configuration holds more than one.

    Returns
    -------
    tuple[Any, OutcomeModelSpec]
        The scikit-learn estimator and its feature specification.

    Raises
    ------
    NotImplementedError
        If the snapshot carries a fitted pyanno4rt preprocessor (feature
        scaling), which this loader does not reproduce.

    Notes
    -----
    Loading unpickles the estimator, which runs code from the file. The path
    is given directly by the caller, so it is trusted like any local file; only
    load models you trust.
    """
    snapshot_dir = Path(snapshot_dir)
    config_path = snapshot_dir / "input.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No pyanno4rt configuration '{config_path}' found.")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    name, params = _select_component(cfg, component)
    model_cfg = params["model"]

    resolved_dir = _resolve_model_dir(model_cfg, snapshot_dir)
    preprocessors = [
        resolved_dir / f"preprocessor{suffix}"
        for suffix in _PREDICTOR_SUFFIXES
        if (resolved_dir / f"preprocessor{suffix}").is_file()
    ]
    if preprocessors:
        raise NotImplementedError(
            f"Model '{model_cfg.get('label')}' was trained with a feature preprocessor "
            f"({preprocessors[0]}), which pyRadPlan does not apply. Re-export "
            "the model without preprocessing or fold the scaling into the estimator."
        )

    estimator = _load_predictor(_find_predictor(resolved_dir))

    features = _order_features(
        _parse_features(model_cfg), model_cfg.get("dataset", {}).get("data_path")
    )

    n_expected = getattr(estimator, "n_features_in_", None)
    if n_expected is not None and int(n_expected) != len(features):
        raise ValueError(
            f"Model '{model_cfg.get('label')}' expects {int(n_expected)} features, but the "
            f"configuration defines {len(features)}."
        )

    spec = OutcomeModelSpec(
        label=model_cfg.get("label", name),
        outcome_type=params.get("outcome_type", "NTCP"),
        features=features,
        transform=bool(params.get("transform", False)),
        model_dir=resolved_dir,
        num_of_fractions=int(cfg.get("configuration", {}).get("number_of_fractions", 30)),
    )

    logger.info(
        "Loaded pyanno4rt outcome model '%s' (%s) from %s with features %s.",
        spec.label,
        spec.outcome_type,
        resolved_dir,
        [feature.name for feature in spec.features],
    )
    return estimator, spec
