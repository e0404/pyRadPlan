"""The About texts shared by the Qt and the web GUI (``pyRadPlan/app/about.json``)."""

import json
from pathlib import Path

import pyRadPlan.app._about as about


def test_about_texts_come_from_the_json_file():
    path = Path(about.__file__).with_name("about.json")
    texts = json.loads(path.read_text(encoding="utf-8"))
    assert set(texts) == {
        "description",
        "license_note",
        "disclaimer_title",
        "disclaimer",
        "imprint_url",
        "privacy_url",
    }
    assert about.DISCLAIMER == texts["disclaimer"]
    assert about.DISCLAIMER_TITLE == texts["disclaimer_title"]
    assert about.LICENSE_NOTE == texts["license_note"]
    assert about.DESCRIPTION == texts["description"]
    assert about.IMPRINT_URL == texts["imprint_url"]
    assert about.PRIVACY_URL == texts["privacy_url"]


def test_disclaimer_states_research_use_and_no_medical_device():
    assert "NOT FOR CLINICAL USE" in about.DISCLAIMER_TITLE
    assert "research and education" in about.DISCLAIMER
    assert "not a medical device" in about.DISCLAIMER


def test_legal_pages_are_in_the_documentation():
    docs = Path(__file__).parents[2] / "docs" / "legal"
    for url, page in ((about.IMPRINT_URL, "imprint"), (about.PRIVACY_URL, "privacy")):
        assert url.startswith("https://docs.pyradplan.org/")
        assert url.endswith(f"/legal/{page}.html")
        assert (docs / f"{page}.rst").is_file()
