"""Monte Carlo engine for TOPAS.

TOPAS is a Monte Carlo simulation toolkit for medical physics, based on Geant4

References
----------
[1] SFaddegon B, Ramos-Mendez J, Schuemann J, McNamara A, Shin J, Perl J, Paganetti H,
    The TOPAS Tool for Particle Simulation, a Monte Carlo Simulation Tool for Physics,
    Biology and Clinical Research. Physica Medica. 2020; 72:114-121.
[2] Perl J, Shin J, Schumann J, Faddegon B, Paganetti H.
    TOPAS: an innovative proton Monte Carlo platform for research and clinical applications.
    Med Phys. 2012; 39(11):6818-37.

Notes
-----
This engine provides an interface to the TOPAS Monte Carlo system
for dose calculations in radiotherapy treatment planning.
For installation instructions and more information, please visit: https://opentopas.readthedocs.io/en/latest/index.html/

Developed by the matRad development team.

Attributes
----------
available_source_models : list[str]
    Supported beam source models (e.g., biGaussian).
    could be extended to for example phasespace

radiation_mode : str
    selected radiation type for the simulation.
source_model : str
    Beam source model used for particle generation.
room_material : str
    Geant4 material used for surrounding environment (e.g., G4_AIR).
rsp_material : str
    Reference stopping power material (e.g., G4_WATER).

scaling_factor : float
    Scaling factor applied to particle statistics or dose normalization.
num_threads : int
    Number of CPU threads used for simulation execution.
num_runs : int
    Number of Monte Carlo runs for statistical averaging.

external_calculation : bool | str
    Flag or mode indicating whether external TOPAS execution is used.

physics_modules : dict
    Dictionary defining Geant4 physics lists used in the simulation.

scorer : dict
    Configuration of scoring options (dose to medium, water, LET, output type).

material_converter : dict
    Configuration for material conversion strategy (RSP or Schneider method).

input_filenames : dict
    Paths to TOPAS input template files for beam setup, scorers, and converters.

base_topas_folder : pathlib.Path
    Base directory containing TOPAS template and configuration files.

simu_dir : str
    Output directory for simulation runs, typically date-stamped.


"""

import logging
import os
from datetime import datetime
import sys
from typing import Any, Union
import numpy as np
import SimpleITK as sitk
from jinja2 import Template

from pyRadPlan.ct import CT, resample_ct, default_hlut
from pyRadPlan.cst import StructureSet
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from pyRadPlan.machines._load import load_from_name
from pyRadPlan.stf import SteeringInformation
from ._base_montecarlo import MonteCarloEngineAbstract

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+


logger = logging.getLogger(__name__)


class ParticleTOPASMCEngine(MonteCarloEngineAbstract):
    # constants

    short_name = "TOPAS"
    name = "TOPAS"
    possible_radiation_modes = [
        "electron",
        "protons",
        "helium",
        "carbon",
        "oxygen",
    ]  # add more if needed

    available_source_models = ["biGaussian"]

    calc_bio_dose: bool
    radiation_mode: str

    external_calculation: Union[str, bool]

    radiation_mode: str
    source_model: str
    room_material: str
    rsp_material: str
    scaling_factor: float
    num_threads: int
    num_runs: int

    physics_modules: list[str]
    base_topas_folder = resources.files("pyRadPlan.data.TOPAS")
    input_filenames = {
        "beam_biGaussian": base_topas_folder
        / "input"
        / "beam_setup"
        / "TOPAS_beam_setup_biGaussian.txt.in",
        "matConv_Schneider_loadFromFile": base_topas_folder
        / "input"
        / "material_converter"
        / "TOPAS_SchneiderConverter.txt.in",
        "scorer_doseToMedium": base_topas_folder
        / "input"
        / "scorer"
        / "TOPAS_scorer_doseToMedium.txt.in",
        "scorer_LET": base_topas_folder / "input" / "scorer" / "TOPAS_subscorer_LET.txt.in",
        "scorer_doseToWater": base_topas_folder
        / "input"
        / "scorer"
        / "TOPAS_scorer_doseToWater.txt.in",
    }

    # Scoring
    scorer = {
        "dose_to_medium": True,
        "dose_to_water": False,
        "LET": False,
        "output_type": "binary",  #'csv'; 'binary';}
    }

    material_converter = {
        "mode": "RSP",  # "Schneider", "RSP"
        "load_from_file": False,  # set true if you want to use your own SchneiderConverter written in "TOPAS_SchneiderConverter"
    }

    _template_filenames = {
        "beamsetup": base_topas_folder / "input" / "template" / "beamSetup_template.txt",
        "patientfile": base_topas_folder / "input" / "template" / "patient_template.txt",
        "current_run": base_topas_folder / "input" / "template" / "pyRadPlan_run_template.txt",
    }
    _label: str = "pyRadPlan_plan"
    _patient_filename = "pyRadPlan_cube"

    def __init__(self, pln: Union[Plan, dict] = None):
        ### Public attributes ###
        # properties with setters and getters
        self.radiation_mode = "protons"
        self.source_model = "biGaussian"

        self.num_threads = 0
        self.num_runs = 5

        self.room_material = "G4_AIR"
        self.rsp_material = "G4_WATER"

        self.scaling_factor = 1e6

        self.external_calculation = True

        self.physics_modules = {
            "default": [
                "g4em-standard_opt4",
                "g4h-phy_QGSP_BIC_HP",
                "g4decay",
                "g4h-elastic_HP",
                "g4stopping",
                "g4ion-QMD",
                "g4radioactivedecay",
            ]
        }

        self.simu_dir: str = self.base_topas_folder.joinpath(datetime.now().strftime("%Y-%m-%d"))

        ### Private attributes ###
        self._computed_quantities = []

        super().__init__(pln)

    def _write_plan_file(self, beams_topas: list[dict], beam_ix: int) -> None:
        """Write  TOPAS files for external simulation."""
        file_path = os.path.join(self.simu_dir, f"beamSetup_pyRadPlan_plan_field{beam_ix}.txt")
        particle_map = {
            "protons": (1, "proton"),
            "helium": (4, "GenericIon(2,4)"),
            "carbon": (12, "GenericIon(6,12)"),
            "oxygen": (16, "GenericIon(8,16)"),
            "electron": (1, "e-"),
        }

        particle_a, particle_str = particle_map[self.radiation_mode]

        input_beam_definition = self._write_beam_scanning(beams_topas, beam_ix)
        input_beam_definition.extend(self._write_beam_parameters(beams_topas, beam_ix, particle_a))

        template_variables = {
            "world_material": self.room_material,
            "particle_name": particle_str,
            "BAMS_to_iso": beams_topas[beam_ix]["BAMS_to_iso"],
            "gantry_angle": beams_topas[beam_ix]["GA"],
            "couch_angle": beams_topas[beam_ix]["CA"],
            "timeline_start": 0,
            "timeline_end": 10 * beams_topas[beam_ix]["total_number_of_bixels"],
            "num_times": beams_topas[beam_ix]["total_number_of_bixels"],
            "trans_x": np.round(beams_topas[beam_ix]["iso_center"][0], 6),
            "trans_y": np.round(beams_topas[beam_ix]["iso_center"][1], 6),
            "trans_z": np.round(beams_topas[beam_ix]["iso_center"][2], 6),
            "input_beam_definition": "".join(input_beam_definition),
            "physics_modules": self.physics_modules.get(
                self.radiation_mode, self.physics_modules["default"]
            ),
        }

        try:
            with open(self._template_filenames["beamsetup"], "r") as template_file:
                template_beam = Template(template_file.read())
            with open(file_path, "w") as f:
                f.write(template_beam.render(**template_variables))

            logger.info(f"File written: {file_path}")
        except PermissionError:
            logger.warning(f"Permission denied when trying to write to {file_path}.")
        except IOError as e:
            logger.warning(f"An error occurred while writing to {file_path}. Error: {e}")
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _write_beam_parameters(
        self, beams_topas: list[dict], beam_ix: int, particle_a: int
    ) -> list[str]:
        # write beam Profile
        beam_path = self.input_filenames["beam_" + self.source_model]
        lines = []
        if not os.path.exists(beam_path):
            print(f"Warning: beam file does not exist: {beam_path}")
        else:
            with open(beam_path, "r") as beam_file:
                for line in beam_file:
                    lines.extend(line)
        # Write Beam Parameters, Energy, Energy Spread, Emittance, Position, Angle, Current
        # Go through the available source models and handle the selected source model
        if self.source_model == "biGaussian":
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas,
                    beam_ix,
                    "Energy",
                    "Emean",
                    "MeV",
                    "dv",
                    transform=lambda x: x * particle_a,
                )
            )
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas, beam_ix, "EnergySpread", "ESpread", "", "uv"
                )
            )
            lines.extend(
                self.helper_write_beam_params(beams_topas, beam_ix, "SigmaX", "sigma_x", "mm")
            )
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas, beam_ix, "SigmaXPrime", "div_x", "", "uv"
                )
            )
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas, beam_ix, "CorrelationX", "corr_x", "", "uv"
                )
            )
            lines.extend(
                self.helper_write_beam_params(beams_topas, beam_ix, "SigmaY", "sigma_y", "mm")
            )
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas, beam_ix, "SigmaYPrime", "div_y", "", "uv"
                )
            )
            lines.extend(
                self.helper_write_beam_params(
                    beams_topas, beam_ix, "CorrelationY", "corr_y", "", "uv"
                )
            )
        else:
            logger.error(f"# Unknown source model: {self.source_model}\n")
        return lines

    def _write_beam_scanning(self, beams_topas: list[dict], beam_ix: int) -> list[str]:
        lines = []
        lines.extend(
            self.helper_write_scanning(
                beams_topas, beam_ix, "AngleX", lambda pb: f"{float(pb['angle'][0]):.6f}", "rad"
            )
        )
        lines.extend(
            self.helper_write_scanning(
                beams_topas, beam_ix, "AngleY", lambda pb: f"{float(pb['angle'][1]):.6f}", "rad"
            )
        )
        lines.extend(
            self.helper_write_scanning(
                beams_topas,
                beam_ix,
                "PosX",
                lambda pb: f"{float(pb['position_bev'][0]):.6f}",
                "cm",
            )
        )
        lines.extend(
            self.helper_write_scanning(
                beams_topas,
                beam_ix,
                "PosY",
                lambda pb: f"{float(pb['position_bev'][1]):.6f}",
                "cm",
            )
        )
        lines.extend(
            self.helper_write_scanning(
                beams_topas, beam_ix, "Current", lambda pb: f"{int(pb['weight'])}", "", "uv"
            )
        )
        return lines

    def _write_patient_file(self, ct: CT) -> None:
        """Write  TOPAS files for external simulation."""
        file_path = os.path.join(self.simu_dir, self._patient_filename)

        image_cube = sitk.GetArrayFromImage(sitk.Cast(ct.cube_hu, sitk.sitkInt16))
        # write ct cube
        try:
            image_cube.tofile(file_path + ".dat")
            logger.info(f"Image File written: {file_path}")
        except PermissionError:
            logger.warning(f"Permission denied when trying to write to {file_path}.")
        except IOError as e:
            logger.warning(f"An error occurred while writing to {file_path}. Error: {e}")
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

        # Material conversion
        input_material_definition = []
        if self.material_converter["mode"] == "RSP":
            material_converter = "MaterialTagNumber"
            hlut = default_hlut()
            unique_materials = np.unique(image_cube)
            unique_rsp = np.interp(unique_materials, hlut[:, 0], hlut[:, 1])
            for idx, value in enumerate(unique_rsp):
                material_name = str(unique_materials[idx]).replace("-", "m")
                input_material_definition.append(
                    's:Ma/Material_HU_{}/BaseMaterial = "{}" \n'.format(
                        material_name, self.rsp_material
                    )
                )
                input_material_definition.append(
                    "d:Ma/Material_HU_{}/Density = {} g/cm3\n".format(material_name, value)
                )

            input_material_definition.append(
                f"iv:Ge/Patient/MaterialTagNumbers = {len(unique_materials)} "
            )
            input_material_definition.append(" ".join(str(int(val)) for val in unique_materials))
            input_material_definition.append("\n")
            input_material_definition.append(
                f"sv:Ge/Patient/MaterialNames = {len(unique_materials)} "
            )
            input_material_definition.append(
                " ".join(
                    f'"Material_HU_{str(name).replace("-", "m")}"' for name in unique_materials
                )
            )
            input_material_definition.append("\n")
        elif self.material_converter["mode"] == "Schneider":
            material_converter = "SchneiderConverter"
            if self.material_converter["load_from_file"]:
                mat_conv_path = self.input_filenames["matConv_Schneider_loadFromFile"]
                if not os.path.exists(mat_conv_path):
                    logger.warning(f"Material converter file does not exist: {mat_conv_path}")
                else:
                    with open(mat_conv_path, "r") as mat_conv_file:
                        for line in mat_conv_file:
                            input_material_definition.append(line)
            else:
                logger.warning(
                    "Schneider converter without load_from_file is not implemented yet."
                )
        else:
            logger.error(f"Unknown material converter mode: {self.material_converter['mode']}")
        template_variables = {
            "world_material": self.room_material,
            "material_converter": material_converter,
            "ct_dim0": ct.cube_dim[0],
            "ct_dim1": ct.cube_dim[1],
            "ct_dim2": ct.cube_dim[2],
            "resx": np.round(ct.resolution["x"], 3),
            "resy": np.round(ct.resolution["y"], 3),
            "resz": np.round(ct.resolution["z"], 3),
            "input_material_definition": "".join(input_material_definition),
        }
        try:
            with open(self._template_filenames["patientfile"], "r") as template_file:
                template_beam = Template(template_file.read())
            with open(file_path + ".txt", "w") as f:
                f.write(template_beam.render(**template_variables))
            logger.info(f"File written: {file_path}")
        except PermissionError:
            logger.warning(f"Permission denied when trying to write to {file_path}.")
        except IOError as e:
            logger.warning(f"An error occurred while writing to {file_path}. Error: {e}")
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _write_run_file(self, beam_ix: int) -> None:
        """Write  TOPAS files for external simulation."""
        for run_ix in range(0, self.num_runs):
            file_path = os.path.join(
                self.simu_dir, f"pyRadPlan_plan_field{beam_ix}_run{run_ix}.txt"
            )
            input_scorer_definition = []
            if self.scorer["dose_to_medium"]:
                with open(self.input_filenames["scorer_doseToMedium"], "r") as scorer_file:
                    for line in scorer_file:
                        input_scorer_definition.append(line)
            if self.scorer["dose_to_water"]:
                with open(self.input_filenames["scorer_doseToWater"], "r") as scorer_file:
                    for line in scorer_file:
                        input_scorer_definition.append(line)
            if self.scorer["LET"]:
                with open(self.input_filenames["scorer_LET"], "r") as scorer_file:
                    for line in scorer_file:
                        input_scorer_definition.append(line)
            try:
                template_variables = {
                    "beam_ix": beam_ix,
                    "run_ix": run_ix,
                    "scorer_output_type": self.scorer["output_type"],
                    "num_threads": self.num_threads,
                    "input_scorer_definition": "".join(input_scorer_definition),
                }

                with open(self._template_filenames["current_run"], "r") as template_file:
                    template_beam = Template(template_file.read())
                with open(file_path, "w") as f:
                    f.write(template_beam.render(**template_variables))
                logger.info(f"File written: {file_path}")
            except PermissionError:
                logger.warning(f"Permission denied when trying to write to {file_path}.")
            except IOError as e:
                logger.warning(f"An error occurred while writing to {file_path}. Error: {e}")
            except Exception as e:
                logger.warning(f"An unexpected error occurred: {e}")

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> Dij:
        """
        Perform the dose calculation using the TOPAS engine.

        Parameters
        ----------
        ct : CT
            The CT object containing the patient data.
        cst : StructureSet
            The structure set containing the contours and VOIs.
        stf : SteeringInformation
            The steering information for the beams.

        Returns
        -------
        Dij
            The dose influence matrix containing the calculated dose.
        """
        dij = self._init_dose_calc(ct, cst, stf)

        w = [beamlet.weight for beam in stf.beams for ray in beam.rays for beamlet in ray.beamlets]
        self.scaling_factor = self.scaling_factor * sum(w)

        ct = resample_ct(
            ct=ct,
            interpolator=sitk.sitkNearestNeighbor,
            target_grid=self.dose_grid,
        )

        if self.external_calculation is True:
            os.makedirs(self.simu_dir, exist_ok=True)
            tmp_machine = load_from_name(stf.beams[0].radiation_mode, stf.beams[0].machine)
            sad = tmp_machine.sad / 10
            beams_topas = []

            # overwrite weights with histories used in simulation
            w = [wi / sum(w) for wi in w]
            rng = np.random.default_rng()
            indices = rng.choice(len(w), size=int(self.num_histories_direct / self.num_runs), p=w)
            w_hist = np.bincount(indices, minlength=len(w))

            counter = 0
            for beam in stf.beams:
                for ray in beam.rays:
                    for beamlet in ray.beamlets:
                        beamlet.weight = w_hist[counter]
                        counter = counter + 1

            for b, beam in enumerate(stf.beams):
                grid_resolution = np.array(
                    [ct.resolution["x"], ct.resolution["y"], ct.resolution["z"]]
                )
                topas_iso = [
                    ct.origin[i] + grid_resolution[i] * (ct.size[i] - 1) / 2.0 for i in range(3)
                ] - beam.iso_center

                beams_dict = {
                    "fieldID": b,
                    "origin": beam.source_point_bev / 10,
                    "GA": beam.gantry_angle,
                    "CA": beam.couch_angle,
                    "BAMS_to_iso": tmp_machine.bams_to_iso_dist / 10,
                    "layers": None,
                    "total_number_of_bixels": beam.total_number_of_bixels,
                    "iso_center": topas_iso,
                }

                layers_topas = []

                for key, value in beam.energy_layers.items():
                    pb_topas = []

                    for i in range(len(value["beamlet_idx"])):
                        ray_position = [
                            beam.rays[value["rays_idx"][i]].ray_pos_bev[2] / 10,
                            beam.rays[value["rays_idx"][i]].ray_pos_bev[0] / 10,
                        ]
                        # angleX corresponds to the rotation around the X axis necessary to move the spot in the Y direction
                        # angleY corresponds to the rotation around the Y' axis necessary to move the spot in the X direction
                        # note that Y' corresponds to the Y axis after the rotation of angleX around X axis
                        # note that Y translates to -Y for TOPAS
                        ray_angle = [0, 0]
                        ray_angle[0] = np.arctan2(ray_position[1], sad)
                        ray_angle[1] = np.arctan2(-ray_position[0], sad / np.cos(ray_angle[0]))
                        # Translate posX and posY to patient coordinates
                        ray_position[0] = (
                            ray_position[0] / sad * (sad - tmp_machine.bams_to_iso_dist / 10)
                        )
                        ray_position[1] = (
                            ray_position[1] / sad * (sad - tmp_machine.bams_to_iso_dist / 10)
                        )

                        pb_dict = {
                            "ID": i,
                            "fieldID": b,
                            "particle": beam.radiation_mode,
                            "T": value["full_energy"],
                            "position_bev": [str(ray_position[0]), str(ray_position[1]), str(0)],
                            "angle": [str(ray_angle[0]), str(ray_angle[1]), str(0)],
                            "weight": beam.rays[value["rays_idx"][i]]
                            .beamlets[value["beamlet_idx"][i]]
                            .weight,
                        }

                        pb_topas.append(pb_dict)

                    emittance = tmp_machine.foci[value["full_energy"]][0].emittance

                    layer_dict = {
                        "T": value["full_energy"],
                        "Emean": tmp_machine.spectra[value["full_energy"]].mean,
                        "ESpread": tmp_machine.spectra[value["full_energy"]].sigma,
                        "sigma_x": emittance.sigma_x,
                        "sigma_y": emittance.sigma_y,
                        "div_x": emittance.div_x,
                        "div_y": emittance.div_y,
                        "corr_x": emittance.corr_x,
                        "corr_y": emittance.corr_y,
                        "pb": pb_topas,
                    }

                    layers_topas.append(layer_dict)

                beams_dict["layers"] = layers_topas
                beams_topas.append(beams_dict)

                self._write_plan_file(beams_topas, b)
                self._write_run_file(b)

                dij["beam_num"][b] = b
                dij["ray_num"][b] = b
                dij["bixel_num"][b] = b

            self._write_patient_file(ct)
        else:
            dij = self._read_external(dij)
            # set dose outside of the patient to 0
            for q_name in self._computed_quantities:
                dij[q_name].flat[0][~self._vdose_grid_mask, :] = 0.0

        dij = self._finalize_dose(dij)
        return dij

    def _read_output_files(self, num_beams: int) -> list[np.ndarray]:
        """
        Read the output files generated by the TOPAS engine for all beams and runs.

        Returns
        -------
        list[np.ndarray]
            A list containing all dose cubes for all beams and runs.
        """
        all_dose_cubes = []

        for beam_ix in range(num_beams):
            for run_ix in range(self.num_runs):
                score_label = f"score_pyRadPlan_plan_field{beam_ix}_run{run_ix}_physicalDose"
                file_path = os.path.join(self.external_calculation, score_label + ".bin")
                try:
                    shape = (
                        self.dose_grid.dimensions[2],
                        self.dose_grid.dimensions[1],
                        self.dose_grid.dimensions[0],
                    )
                    arr = np.fromfile(file_path, dtype=np.float64)
                    arr = arr.reshape(shape)
                    all_dose_cubes.append(arr)
                except Exception as e:
                    logger.warning(f"Could not read dose file {file_path}: {e}")
                    continue

        return all_dose_cubes

    def _read_external(self, dij: dict) -> dict:
        all_dose_cubes = self._read_output_files(dij["num_of_beams"])
        all_dose_cubes = np.array(all_dose_cubes) * self.scaling_factor / self.num_histories_direct
        # Calculate sum and var for all dose cubes
        # Calculate sum and var for each beam individually
        for beam_ix in range(dij["num_of_beams"]):
            # Each beam has self.num_runs dose cubes
            start = beam_ix * self.num_runs
            end = start + self.num_runs
            dij["physical_dose"].flat[0][:, beam_ix] = np.sum(
                all_dose_cubes[start:end], axis=0
            ).flatten()
            dij["physical_dose_var"].flat[0][:, beam_ix] = np.var(
                all_dose_cubes[start:end], axis=0
            ).flatten()
            dij["beam_num"][beam_ix] = beam_ix
            dij["ray_num"][beam_ix] = beam_ix
            dij["bixel_num"][beam_ix] = beam_ix
        return dij

    def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]) -> Dij:
        # Loop over all requested quantities
        for q_name in names:
            # Create dij list for each quantity
            dij[q_name] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)
            dij[q_name + "_var"] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

            # Loop over all scenarios and preallocate quantity containers
            # TODO: write test for this
            for i in range(self.mult_scen.scen_mask.size):
                # Only if there is a scenario we will allocate
                if self.mult_scen.scen_mask.flat[i]:
                    dij[q_name].flat[i] = np.zeros(
                        (self.dose_grid.num_voxels, dij["num_of_beams"]), dtype=np.float32
                    )
                    dij[q_name + "_var"].flat[i] = np.zeros(
                        (self.dose_grid.num_voxels, dij["num_of_beams"]), dtype=np.float32
                    )

            self._computed_quantities.append(q_name)
            self._computed_quantities.append(q_name + "_var")

        return dij

    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> None:
        self.radiation_mode = stf.beams[0].radiation_mode
        dij = super()._init_dose_calc(ct, cst, stf)
        dij = self._allocate_quantity_matrices(dij, ["physical_dose"])

        if dij["num_of_scenarios"] > 1:
            raise NotImplementedError(
                "Multiple scenarios are not supported for TOPAS calculations."
            )

        return dij

    @staticmethod
    def is_available(pln, machine):
        available, msg = True, ""
        return available, msg

    @staticmethod
    def helper_write_scanning(
        beams_topas, beam_ix, field_name, value_getter, unit="", value_prefix="dv"
    ):
        beam = beams_topas[beam_ix]

        values = " ".join(
            value_getter(pb_entry) for layer in beam["layers"] for pb_entry in layer["pb"]
        )
        lines = []
        lines.extend(f's:Tf/Beam/{field_name}/Function = "Step"\n')
        lines.extend(f"dv:Tf/Beam/{field_name}/Times = Tf/Beam/Spot/Times ms\n")
        lines.extend(
            f"{value_prefix}:Tf/Beam/{field_name}/Values = {beam['total_number_of_bixels']:d} "
        )
        lines.extend(values)
        if unit:
            lines.extend(f" {unit}")
        lines.extend("\n")
        return lines

    @staticmethod
    def helper_write_beam_params(
        beams_topas,
        beam_ix,
        param_name,
        layer_key,
        unit="",
        value_prefix="dv",
        transform=None,
    ):
        total = beams_topas[beam_ix]["total_number_of_bixels"]
        layers = beams_topas[beam_ix]["layers"]
        lines = []
        lines.extend(f's:Tf/Beam/{param_name}/Function = "Step"\n')
        lines.extend(f"dv:Tf/Beam/{param_name}/Times = Tf/Beam/Spot/Times ms\n")
        lines.extend(f"{value_prefix}:Tf/Beam/{param_name}/Values = {total:d} ")

        values = (
            transform(layer[layer_key]) if transform else layer[layer_key]
            for layer in layers
            for _ in layer["pb"]
        )

        lines.extend(" ".join(f"{v:.6f}" for v in values))

        if unit:
            lines.extend(f" {unit}")
        lines.extend("\n")
        return lines
