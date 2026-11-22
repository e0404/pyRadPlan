"""The About texts shared by the Qt and the web GUI (``pyRadPlan/app/about.json``)."""

import json
from pathlib import Path

import pyRadPlan.app._about as about


def test_about_texts_come_from_the_json_file():
    path = Path(about.__file__).with_name("about.json")
    texts = json.loads(path.read_text(encoding="utf-8"))
    assert set(texts) == {"description", "license_note", "disclaimer_title", "disclaimer"}
    assert about.DISCLAIMER == texts["disclaimer"]
    assert about.DISCLAIMER_TITLE == texts["disclaimer_title"]
    assert about.LICENSE_NOTE == texts["license_note"]
    assert about.DESCRIPTION == texts["description"]


def test_disclaimer_states_research_use_and_no_medical_device():
    assert "NOT FOR CLINICAL USE" in about.DISCLAIMER_TITLE
    assert "research and education" in about.DISCLAIMER
    assert "not a medical device" in about.DISCLAIMER
