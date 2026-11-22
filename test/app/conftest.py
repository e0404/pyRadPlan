import time

import pytest

from pyRadPlan.app import SessionManager, dispatch

#: Coarse TG119 photon plan: three beams, 10 mm bixels and a 6 mm dose grid keep the whole
#: pipeline at a few seconds.
COARSE_PLAN = {
    "gantry_angles": [0.0, 120.0, 240.0],
    "bixel_width": 10.0,
    "dose_grid_resolution": [6.0, 6.0, 6.0],
}


class Client:
    """Calls :func:`dispatch` like a frontend would, collecting the events."""

    def __init__(self) -> None:
        self.manager = SessionManager()
        self.events: list[dict] = []
        self.session = None
        self._next_id = 0

    def call(self, method: str, params: dict | None = None, buffers=None, *, session=True):
        self._next_id += 1
        request = {"id": self._next_id, "method": method, "params": params or {}}
        if buffers is not None:
            request["buffers"] = buffers
        if session:
            request["session"] = self.session
        return dispatch(request, self.events.append, manager=self.manager)

    def ok(self, method: str, params: dict | None = None, **kwargs):
        response = self.call(method, params, **kwargs)
        assert response["ok"], response.get("error")
        return response

    def events_of(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["event"] == kind]


@pytest.fixture(scope="session")
def client_factory():
    """The :class:`Client` class, for fixtures wider than one test."""
    return Client


@pytest.fixture
def coarse_plan():
    return dict(COARSE_PLAN)


@pytest.fixture
def client():
    client = Client()
    client.session = client.ok("session.create", session=False)["result"]["session"]
    yield client
    client.manager.close_all()


@pytest.fixture(scope="session")
def tg119_pipeline():
    """A TG119 session run through the full photon pipeline once, via dispatch."""
    client = Client()
    client.session = client.ok("session.create", session=False)["result"]["session"]
    timings = {}
    steps = [
        ("data.load_phantom", {"name": "TG119"}),
        ("plan.set", COARSE_PLAN),
        ("workflow.generate_stf", {}),
        ("workflow.calc_dose_influence", {}),
        ("workflow.optimize", {"solver": "scipy", "max_iter": 20}),
    ]
    states = {}
    for method, params in steps:
        start = time.perf_counter()
        states[method] = client.ok(method, params)["result"]
        timings[method] = time.perf_counter() - start
    yield client, states, timings
    client.manager.close_all()
