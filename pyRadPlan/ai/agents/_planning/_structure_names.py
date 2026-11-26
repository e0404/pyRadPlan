"""Clinical structure names: normalization, synonyms and fuzzy matching."""

import difflib
import re
from typing import Optional

# Small multilingual synonym table -> TG-263-like canonical names. Extend per clinic.
_SYNONYMS = {
    "rectum": "Rectum",
    "rektum": "Rectum",
    "rect": "Rectum",
    "bladder": "Bladder",
    "blase": "Bladder",
    "harnblase": "Bladder",
    "vesica": "Bladder",
    "brainstem": "Brainstem",
    "hirnstamm": "Brainstem",
    "spinalcord": "SpinalCord",
    "myelon": "SpinalCord",
    "rueckenmark": "SpinalCord",
    "cord": "SpinalCord",
    "chiasm": "OpticChiasm",
    "chiasma": "OpticChiasm",
    "opticchiasm": "OpticChiasm",
    "parotidl": "Parotid_L",
    "parotisl": "Parotid_L",
    "lparotid": "Parotid_L",
    "parotidr": "Parotid_R",
    "parotisr": "Parotid_R",
    "rparotid": "Parotid_R",
    "femurl": "Femur_Head_L",
    "femoralheadl": "Femur_Head_L",
    "femurheadl": "Femur_Head_L",
    "hueftkopfl": "Femur_Head_L",
    "femurr": "Femur_Head_R",
    "femoralheadr": "Femur_Head_R",
    "femurheadr": "Femur_Head_R",
    "hueftkopfr": "Femur_Head_R",
    "lungl": "Lung_L",
    "lunger": "Lung_R",
    "lungr": "Lung_R",
    "heart": "Heart",
    "herz": "Heart",
    "body": "BODY",
    "external": "BODY",
    "aussenkontur": "BODY",
    "skin": "BODY",
    "lungs": "Lungs",
    "lungtotal": "Lungs",
    "lungbilat": "Lungs",
    "lunge": "Lungs",
    "lungen": "Lungs",
    "esophagus": "Esophagus",
    "oesophagus": "Esophagus",
    "speiseroehre": "Esophagus",
    "larynx": "Larynx",
    "kehlkopf": "Larynx",
    "mandible": "Mandible",
    "mandibula": "Mandible",
    "unterkiefer": "Mandible",
    "opticnervel": "OpticNerve_L",
    "nervusopticusl": "OpticNerve_L",
    "sehnervl": "OpticNerve_L",
    "opticnerver": "OpticNerve_R",
    "nervusopticusr": "OpticNerve_R",
    "sehnervr": "OpticNerve_R",
    "cochleal": "Cochlea_L",
    "cochlear": "Cochlea_R",
    "penilebulb": "PenileBulb",
    "bulbuspenis": "PenileBulb",
}


def normalize_structure_name(name: str) -> str:
    """Lower-case, strip separators and laterality words: 'Parotid Left ' -> 'parotidl'."""
    n = name.lower().replace("_", " ").strip()
    n = re.sub(r"\b(links|left|lt)\b", "l", n)
    n = re.sub(r"\b(rechts|right|rt)\b", "r", n)
    n = n.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")  # codespell:ignore ue
    return re.sub(r"[^a-z0-9]", "", n)


def canonical_structure_name(name: str) -> Optional[str]:
    """Return the canonical name of *name*, None if unknown."""
    return _SYNONYMS.get(normalize_structure_name(name))


def name_candidates(query: str, names: list[str], n: int = 3) -> list[dict]:
    """Rank *names* by similarity to *query* (canonical match first, then fuzzy)."""
    q_norm, q_can = normalize_structure_name(query), canonical_structure_name(query) or query
    scored = []
    for name in names:
        score = difflib.SequenceMatcher(None, q_norm, normalize_structure_name(name)).ratio()
        if canonical_structure_name(name) == q_can:
            score = max(score, 0.95)
        scored.append((round(score, 2), name))
    scored.sort(reverse=True)
    return [{"voi": nm, "score": sc} for sc, nm in scored[:n] if sc >= 0.4]
