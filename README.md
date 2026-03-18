# pyRadPlan
[![Tests](https://github.com/e0404/pyRadPlan/actions/workflows/tests.yml/badge.svg)](https://github.com/e0404/pyRadPlan/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/e0404/pyRadPlan/graph/badge.svg?token=S1KCYDU17G)](https://codecov.io/gh/e0404/pyRadPlan)
[![pypi version](https://img.shields.io/pypi/v/pyRadPlan)](https://pypi.org/project/pyRadPlan/)
![pyversion](https://img.shields.io/pypi/pyversions/pyRadPlan)
![contributors](https://img.shields.io/github/contributors-anon/e0404/pyRadPlan)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/github/license/e0404/pyRadPlan)



pyRadPlan is an open-source radiotherapy treatment planning toolkit designed for interoperability with [matRad](http://www.matRad.org).

Development is lead by the [Radiotherapy Optimization group](https://www.dkfz.de/radopt) at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de)

## Concept and Goals
pyRadPlan is a multi-modality treatment planning toolkit in python born from the established Matlab-based toolkit [matRad](http://www.matRad.org). As such, pyRadPlan aims to provide a framework as well as tools for combining dose calculation with optimization with focus on ion planning.

One of pyRadPlan's main goals is for its algorithms to be [Python Array API](https://data-apis.org/array-api/) conform as much as possible, to allow free choice of compute backends and devices (i.e., [`numpy`](https://numpy.org/), [`cupy`](https://cupy.dev/), [`pytorch`](https://pytorch.org/)). Further, its data structures (more precisely, their metadata) should be easily serializable for exchange with novel LLMs. This facilitate performant treatment planning research while enabling the integration of AI - such as deep neural networks implemented in [`pytorch`](https://pytorch.org/) or interaction with an LLM Agent via [`pydantic-ai`](https://ai.pydantic.dev/) -  at every step of the treatment planning workflow.

### Data Structures
pyRadPlan uses a similar datastructure and workflow concept as in matRad, while trying to ensure that the corresponding datastructures can be easily imported and exported from/to matRad. This facilitates the application of either algortihms from matRad or native pyRadPlan at any stage of the treatment planning workflow.

To enforce valid datastructures, we perform validation and serialization with [`pydantic`](https://docs.pydantic.dev/latest/).
Datastructures and algorithms rely mostly on [`SimpleITK`](https://simpleitk.readthedocs.io), [`numpy`](https://numpy.org/), and [`scipy`](https://scipy.org/) for internal data representation and processing.

### Scripting and API

#### Minimal Script using the Top-Level API
A minimal script is very similar to matRad's example [`matRad.m`](https://github.com/e0404/matRad) script:
```python
from importlib import resources
import pymatreader
from pyRadPlan import load_patient, IonPlan, generate_stf, calc_dose_influence, fluence_optimization

#  Read patient from provided TG119.mat file and validate data
tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
ct, cst = load_patient(tg119_path)

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Fluence Optimization (objectives loaded from "cst")
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Result
result = dij.compute_result_ct_grid(fluence)
```

This script uses the top-level API exposed when importing pyRadPlan. The top-level functions are designed to take the main data structures as input and configure the corresponding workflow step via the `Plan` using the attribute dictionaries `pln.prop*`:

| Plan property | API function | Description | ID |
| -------- | ------- | ------ | ------- |
| `prop_stf`  | `generate_stf` | Create beam Geometry |  generator |
| `prop_dose_calc`  | `calc_dose_influence`, `calc_dose_forward` |  Calculate dose matrix / distribution | engine |
| `prop_opt` | `fluence_optimization` | Optimization of beam fluences | problem

The Plan properties are dictionaries. Based on their content, the top-level-api will choose the correct implementation and configure the corresponding settings. The ID can be used in the corresponding `pln.prop*` dictionary to identify a specific implementation. E.g., `pln.prop_stf = {"generator": "IMPT"}` will select the IMPT Geometry Generator if compatible to the plan. pyRadPlan will try to set all provided configuration parameters in the dictionary and use default parameters and implementations if no keys are provided.

The top level api is designed to require minimal programming experience and to run the same planning workflows with different configurations by just changing the corresponding plan object.

#### Low-level API
Instead of using above top-level workflow functions and a central plan configuration, one can also built custom workflows instantiating the necessary algorithm objects directly, for example:

```python
...
from pyRadPlan.stf import StfGeneratorIMPT
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")

# Equivalent Top Level API configuration
# pln.prop_stf = {"gantry_angles": [90, 270], "couch_angles": [0, 0], "generator": "IMPT"}
# stf = generate_stf(ct,cst,pln)

# Low level:
stf_gen = StfGeneratorIMPT()
stf_gen.gantry_angles = [90, 270]
stf_gen.couch_angles = [0, 0]
stf = stf_gen.generate(ct,cst)
```

If you are interested in helping with development, get in touch, read the contributing guidelines, and the developer note below.

## Contributing & Notes for Developers
pyRadPlan development uses unit-testing and code formatting via pre-commit hooks to ensure clean code.
If you are a developer or want to contribute, make sure to clone the latest state via git. Then, we strongly suggest to create a virtual python environment with a suitable version and do an editable installation of pyRadPlan in dev mode:  `pip install -e .[dev]`

> **Note**
> If you are using venv to create a virtual environment in the project's root folder, we suggest to name it `.venv` as this folder will be automatically excluded by all formatters and linters

This will install an editable pyRadPlan module including `pytest` and `pre-commit` in addition to the standard modules.
- **pytest** is used to run unit tests before publishing the code. Run on the test folder via `pytest test`, or choose any test file following the pytest syntax. We encourage writing at least fundamental unit tests for new code. Will also install `coverage` and the pytest extension to monitor coverage.
- **pre-commit** allows for automatic code formatting to ensure following of PEPs (mainly PEP8 and PEP257).

After successfully running `pip install -e .[dev]` check in the console that `pre-commit --version` provides a correct response.
Afterwards run `pre-commit install`. This reads the `.pre-commit_config.yaml` and adds a hook to your git repository, which whenever a commit is made the changed files are reformatted to ensure the PEP standards.

### Matlab / Octave Files & Engine
If you want to interface via matRad, the simplest way is to write mat files using scipy's `savemat` and load them in matRad. To save a datastructure such that it can be read into Matlab and interpreted by matRad, you can call `to_matrad` on the structure and then pass the resulting dictionary to `savemat`.
You can also install the matlab engine by appending `[matlab]` to the pip install command. However, as there is some compatibility matrix for what matlabengine version is suitable for your Matlab installation, it might be better to not install the latest version but instead install the correct matlabengine manually before.
To use Octave via oct2py, analogously append `[octave]`.
