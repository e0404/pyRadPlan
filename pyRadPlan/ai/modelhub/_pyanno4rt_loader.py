"""Loader for pyanno4rt outcome models.

A pyanno4rt snapshot folder holds the plan's ``input.json``, which defines the model's
features, and one folder per outcome model. Two layouts of the model folder are read:

- the one written by the public pyanno4rt: ``<label>/model.sav`` (a pickled scikit-learn
  estimator) next to ``<label>/configuration.json``, which records ``feature_names`` (the
  column order of the training matrix) and ``preprocessing_steps``;
- ``<label>/predictor.*`` without a configuration file.

``model.*`` is preferred over ``predictor.*`` if a folder holds both.

pyanno4rt fits on the column order of the training table, not the JSON order. The feature
order is therefore taken from the first source available: ``feature_names`` of
``configuration.json``, the estimator's ``feature_names_in_``, the header of the training
table ``dataset.data_path``, and last the JSON order with a warning.

pyRadPlan does not scale features. pyanno4rt refits its preprocessor from the training data
instead of saving it, so a model is refused if ``configuration.json`` or ``input.json`` name
any preprocessing step other than ``Identity``, or if a ``preprocessor.*`` file is present.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional, Union


logger = logging.getLogger(__name__)

#: Estimator file names (without suffix) understood by this loader, in order of preference.
_PREDICTOR_STEMS = ("model", "predictor")

#: Estimator file suffixes understood by this loader, in order of preference.
_PREDICTOR_SUFFIXES = (".save", ".sav", ".pkl", ".pickle", ".joblib")

#: pyanno4rt preprocessing steps that leave the features unchanged.
_IDENTITY_STEPS = frozenset({"Identity"})


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
    model_dir : Path or None
        Directory the predictor was loaded from.
    """

    label: str
    outcome_type: str = "NTCP"
    features: list[OutcomeFeature] = field(default_factory=list)
    model_dir: Optional[Path] = None


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


def _select_features(
    features: list[OutcomeFeature], names: list[str], origin: str
) -> list[OutcomeFeature]:
    """Return the features listed in ``names``, in that order.

    Configured features that ``names`` does not list were removed by pyanno4rt's feature
    filter before the fit and are dropped. Names without a configured feature are an error.
    """
    by_name = {feature.name: feature for feature in features}
    unknown = [name for name in names if name not in by_name]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    unlisted = [feature.name for feature in features if feature.name not in names]
    if unknown or duplicates:
        raise ValueError(
            f"The feature names of the outcome model given by {origin} do not match the "
            f"'Dynamic Feature' columns of the configuration: without a configured feature "
            f"{unknown}, listed more than once {duplicates}, configured but not listed "
            f"{unlisted}."
        )

    if unlisted:
        logger.warning(
            "The outcome model was not trained on the configured features %s (%s does not "
            "list them); they are ignored.",
            unlisted,
            origin,
        )

    selected = [by_name[name] for name in names]
    if selected != features:
        logger.info("Outcome model features in the order given by %s: %s", origin, names)
    return selected


def _order_features(
    features: list[OutcomeFeature],
    configuration: Optional[dict],
    estimator: Any,
    data_path: Optional[str],
) -> list[OutcomeFeature]:
    """Sort features into the order the estimator was fitted on (see the module docstring)."""
    recorded = {}
    if configuration is not None and configuration.get("feature_names") is not None:
        recorded["'configuration.json'"] = [str(n) for n in configuration["feature_names"]]
    if getattr(estimator, "feature_names_in_", None) is not None:
        recorded["the estimator's 'feature_names_in_'"] = [
            str(n) for n in estimator.feature_names_in_
        ]

    if recorded:
        if len({tuple(names) for names in recorded.values()}) > 1:
            raise ValueError(f"The outcome model's feature names are contradictory: {recorded}.")
        origin, names = next(iter(recorded.items()))
        return _select_features(features, names, origin)

    warning = (
        "Cannot verify the feature order of the outcome model (%s); falling back to the order "
        "given in the configuration. If it differs from the training table, the model's "
        "coefficients bind to the wrong features. Make the training table available to "
        "remove this ambiguity."
    )

    if not data_path:
        logger.warning(warning, "the configuration records no dataset path")
        return features

    path = Path(data_path)
    if not path.is_file():
        logger.warning(warning, f"'{path}' is not readable")
        return features

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, UnicodeDecodeError, StopIteration) as exc:
        logger.warning(warning, f"'{path}' could not be read ({exc})")
        return features

    order = {name: index for index, name in enumerate(header)}
    missing = [feature.name for feature in features if feature.name not in order]
    if missing:
        raise ValueError(
            f"The configured feature columns {missing} are absent from the training table "
            f"'{path}' (columns {header}), so the feature order of the outcome model cannot "
            "be determined."
        )

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
    """Return the estimator file inside ``model_dir``; ``model.*`` wins over ``predictor.*``."""
    for stem in _PREDICTOR_STEMS:
        for suffix in _PREDICTOR_SUFFIXES:
            candidate = model_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate

    others = sorted(p.name for stem in _PREDICTOR_STEMS for p in model_dir.glob(f"{stem}.*"))
    if others:
        raise NotImplementedError(
            f"Model directory '{model_dir}' holds {others}, which is not a pickled scikit-learn "
            f"estimator. Only {list(_PREDICTOR_SUFFIXES)} files are supported."
        )
    raise FileNotFoundError(
        f"No model file found in '{model_dir}' (expected 'model.*' or 'predictor.*')."
    )


def _read_model_configuration(model_dir: Path) -> Optional[dict]:
    """Read the ``configuration.json`` pyanno4rt writes next to the model, if present."""
    path = model_dir / "configuration.json"
    if not path.is_file():
        return None
    configuration = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError(f"'{path}' is not a pyanno4rt model configuration.")
    return configuration


def _check_preprocessing(label: Optional[str], model_dir: Path, stated: dict[str, Any]) -> None:
    """Refuse models whose features were transformed before the fit."""
    for suffix in _PREDICTOR_SUFFIXES:
        preprocessor = model_dir / f"preprocessor{suffix}"
        if preprocessor.is_file():
            raise NotImplementedError(
                f"Model '{label}' was trained with a feature preprocessor "
                f"({preprocessor}), which pyRadPlan does not apply. Re-export "
                "the model without preprocessing or fold the scaling into the estimator."
            )

    stated = {origin: steps for origin, steps in stated.items() if steps is not None}
    for origin, steps in stated.items():
        names = [steps] if isinstance(steps, str) else list(steps)
        applied = [step for step in names if step not in _IDENTITY_STEPS]
        if applied:
            raise NotImplementedError(
                f"Model '{label}' was trained with the preprocessing steps {applied} "
                f"(according to {origin}), which pyRadPlan does not apply: the model would be "
                "evaluated on unscaled features. Retrain it with the 'Identity' step only or "
                "fold the scaling into the estimator."
            )

    if not stated:
        logger.warning(
            "Neither 'configuration.json' nor 'input.json' state the preprocessing steps of "
            "outcome model '%s', so it could not be verified that the model was trained "
            "without feature scaling. pyRadPlan evaluates it on unscaled features.",
            label,
        )


def _load_predictor(path: Path) -> Any:
    """Unpickle the estimator."""
    if find_spec("sklearn") is None:
        raise ImportError(
            "Loading a pyanno4rt model needs scikit-learn, which pyRadPlan does not install: "
            "pip install scikit-learn"
        )
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
        pyanno4rt snapshot folder, holding ``input.json`` and the model folder
        ``<label>`` with the pickled estimator ``model.*`` (as written by the
        public pyanno4rt: ``model.sav`` next to ``configuration.json``) or
        ``predictor.*``. ``model.*`` is preferred if both exist.
    component : str, optional
        Name, identifier or model label of the outcome component to load.
        Required only when the configuration holds more than one.

    Returns
    -------
    tuple[Any, OutcomeModelSpec]
        The scikit-learn estimator and its feature specification. The features
        are in the order the estimator was fitted on, taken from the first
        source available: ``feature_names`` in ``<label>/configuration.json``,
        the estimator's ``feature_names_in_``, the header of the training
        table ``dataset.data_path`` (comma-separated, as pyanno4rt reads it),
        and last the order in ``input.json`` (with a warning, because it
        cannot be verified). Configured features that the first two sources
        do not list were filtered out before the fit and are dropped.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    NotImplementedError
        If the model was trained on preprocessed features, which pyRadPlan
        does not reproduce: ``preprocessing_steps`` in
        ``<label>/configuration.json`` or in the model's entry of
        ``input.json`` name a step other than ``Identity`` (an empty list
        means ``Identity``), or a ``<label>/preprocessor.*`` file exists. If
        neither file states the steps, a warning is logged instead. Also
        raised for model files that are not pickled scikit-learn estimators
        (e.g. ``model.h5``).
    ValueError
        If the feature names recorded with the model are not among the
        ``Dynamic Feature`` columns of ``input.json``, if these columns are
        missing from the training table, or if the number of features does
        not match the estimator.

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
    configuration = _read_model_configuration(resolved_dir)
    _check_preprocessing(
        model_cfg.get("label"),
        resolved_dir,
        {
            "'configuration.json'": (configuration or {}).get("preprocessing_steps"),
            "'input.json'": model_cfg.get("preprocessing_steps"),
        },
    )

    estimator = _load_predictor(_find_predictor(resolved_dir))

    features = _order_features(
        _parse_features(model_cfg),
        configuration,
        estimator,
        model_cfg.get("dataset", {}).get("data_path"),
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
        model_dir=resolved_dir,
    )

    logger.info(
        "Loaded pyanno4rt outcome model '%s' (%s) from %s with features %s.",
        spec.label,
        spec.outcome_type,
        resolved_dir,
        [feature.name for feature in spec.features],
    )
    return estimator, spec
