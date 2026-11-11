"""Monte Carlo engine for FRED.

FRED (Fast paRticle dosE calculatoR) is a Monte Carlo simulation toolkit
for particle therapy developed by the Fondazione CNAO in Italy.

References
----------
.. [1] Schiavi, A., Senzacqua, M., Pioli, S., Mairani, A., Magro, G., Molinelli, S.,
       Ciocca, M., Battistoni, G., Patera, V. (2017). Fred: a GPU-accelerated
       fast-Monte Carlo code for rapid treatment plan recalculation in ion beam therapy.
       Physics in Medicine and Biology, 62(18), 7482–7504.
       https://doi.org/10.1088/1361-6560/aa8134

Notes
-----
This engine provides an interface to the FRED Monte Carlo system
for particle dose calculations in radiotherapy treatment planning.
For installation instructions and more information, please visit: https://www.fred-mc.org/

Developed by the matRad development team.
"""

import itertools
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Annotated, Any, ClassVar, Optional, Union, cast, Literal
import time
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import sparse

from ...biology.models import bio_influence_from_let
from ...core import AlgorithmParameterMetadata, Grid, get_data_dir
from ...ct import CT, resample_ct, default_hlut
from ...cst import StructureSet
from ...dij import Dij
from ...plan import Plan
from ...stf import SteeringInformation, Beam
from ...machines.particles import IonAccelerator

from ._base_montecarlo import MonteCarloEngineAbstract, ScoredLETBioMixin

logger = logging.getLogger(__name__)

#: FRED's name of the primary particle of each radiation mode (FRED names ions by nuclide).
FRED_PARTICLES = {"protons": "proton", "helium": "He4", "carbon": "C12", "oxygen": "O16"}


class ParticleFredMCEngine(ScoredLETBioMixin, MonteCarloEngineAbstract):
    # constants

    short_name = "FRED"
    name = "FRED"
    possible_radiation_modes = list(FRED_PARTICLES)

    # A LET-based biological model adds the LETd scorer
    _per_calculation_settings: ClassVar[tuple[str, ...]] = (
        *MonteCarloEngineAbstract._per_calculation_settings,
        "scorers",
        "hlut",
    )

    computes_bio_influence = True
    # FRED scores the dose-averaged LET itself and does not use the machine's kernels.
    bio_kernel_quantities = ()
    bio_scored_quantities = ("let:LETd",)

    available_source_models = ["gaussian", "emittance", "sigmaSqrModel"]

    available_versions = ["3.70.0", "3.76.0", "3.60.0"]

    # As in TOPAS: True only writes FRED's input files (the dose is zero); a folder reads the
    # output of FRED runs made from them
    external_calculation: Union[str, bool]
    calc_bio_dose: Union[Literal["auto"], bool]
    calc_let: Union[Literal["auto"], bool]

    fred_version: str
    fred_cmd: str
    fred_dir: Path
    use_gpu: bool

    # Folder FRED's inp/ and out/ are written to and kept in; None uses a temporary folder
    # that is deleted after each calculation
    simu_dir: Optional[Path]
    save_input: Union[os.PathLike, bool]
    save_output: Union[os.PathLike, bool]
    use_output: Union[os.PathLike, bool]
    print_output: bool
    execution_timeout: Optional[float]

    scorers: list[str]
    source_model: str
    room_material: str
    HU_table_file: str
    material_converter: Literal["hlut", "RSP"]
    # HU to rSP lookup table shared with the analytical engines (default_hlut() if None)
    hlut: Annotated[
        Optional[np.ndarray], AlgorithmParameterMetadata(configurable=False, kind="data")
    ] = None
    scaling_factor: int
    # Zero the dose (and LET / biological quantities) in voxels outside all cst structures
    ignore_outside_patient: bool
    # Forward calculation: one FRED run per beam (inp/beam_<b>, out/beam_<b>), so the result
    # holds the dose of every beam. Influence matrices are per beamlet anyway. A `use_output`
    # folder without beam_<b> subfolders is read as one combined dose.
    dose_per_beam: bool

    def __init__(self, pln: Union[Plan, dict] = None):
        ### Public attributes ###
        # properties with setters and getters
        self.simu_dir = None
        self.save_input = False
        self.save_output = False
        self.use_output = False
        self.source_model = "gaussian"
        self.external_calculation = False

        self.use_gpu = True
        self.calc_let = "auto"
        self.calc_bio_dose = "auto"
        self.scorers = ["Dose"]
        self.source_model = "gaussian"
        self.room_material = "Air"
        self.HU_table_file = "hLut.inp"
        # "hlut": FRED interpolates the hlut itself.
        # "RSP": one water entry per HU value in the CT, interpolated like TOPAS's RSP mode.
        self.material_converter = "hlut"
        self.scaling_factor = 1e6
        self.ignore_outside_patient = False
        self.dose_per_beam = True
        self.fred_version = "3.70.0"
        self.print_output = False
        # Seconds before the FRED subprocess is aborted; None waits indefinitely.
        self.execution_timeout = None

        self.fred_cmd = "fred"
        self.fred_dir = Path(os.environ.get("FREDDIR", ""))

        ### Private attributes ###
        self._computed_quantities = []
        self._bio_evaluator = None
        self._bio_influence_names: tuple[str, ...] = ()
        self._total_number_of_bixels = None
        self._dij_format_version = None
        self._patient_filename = "CTpatient.mhd"
        self._run_input_filename = "fred.inp"
        self._regions_filename = "regions.inp"
        self._plan_filename = "plan.inp"
        self._plan_delivery_filename = "planDelivery.inp"
        self._input_foldername = "inp"
        self._regions_foldername = "regions"
        self._plan_foldername = "plan"
        self._output_folder = "out"
        self._output_image = "Dose.mhd"
        self._output_score = "/score/Phantom.Dose.mhd"
        self._conversion_factor = 1e6
        self._HU_clamping: bool = False
        self._ct_hu_values: Optional[np.ndarray] = None
        self._simulated_particles_per_bixel = 1e2
        self._create_working_dirs()

        super().__init__(pln)

    @property
    def simu_dir(self) -> Optional[Path]:
        return self._simu_dir

    @simu_dir.setter
    def simu_dir(self, value: Optional[os.PathLike]):
        if value is None or value == "":
            self._simu_dir = None
        elif isinstance(value, (str, os.PathLike)):
            # FRED runs inside the input folder, so a relative path must not depend on it
            self._simu_dir = Path(value).resolve()
        else:
            raise ValueError("simu_dir must be a path or None.")

    @property
    def save_output(self) -> Union[os.PathLike, bool]:
        if self._save_output is None:
            return False
        return self._save_output

    @save_output.setter
    def save_output(self, value: Union[str, int, bool]):
        try:
            self._save_output = self._parse_external_calc_property(value)
        except ValueError as e:
            raise ValueError("save_output could not be set") from e

    @property
    def use_output(self) -> Any:
        if self._use_output is None:
            return False
        return self._use_output

    @use_output.setter
    def use_output(self, value: Union[str, int, bool]):
        try:
            self._use_output = self._parse_external_calc_property(value)
        except ValueError as e:
            raise ValueError("use_output could not be set") from e

    @property
    def save_input(self) -> Union[os.PathLike, bool]:
        if self._save_input is None:
            return False
        return self._save_input

    @save_input.setter
    def save_input(self, value: Union[os.PathLike, int, bool]):
        try:
            self._save_input = self._parse_external_calc_property(value)
        except ValueError as e:
            raise ValueError("save_input could not be set") from e

    def _parse_external_calc_property(
        self, value: Union[os.PathLike, bool, int]
    ) -> Union[os.PathLike, None]:
        """
        Parse the external calculation property to ensure it is a valid path or None.

        Parameters
        ----------
        value : Union[os.PathLike,bool]
            The value to be parsed.

        Returns
        -------
        Union[os.PathLike, None]
            The parsed path or None if the value is not set.
        """
        if isinstance(value, (str, bytes, os.PathLike)) and (
            os.path.isdir(value) or os.path.isdir(os.path.dirname(value))
        ):
            return Path(value)
        elif isinstance(value, (int, bool)):
            value = bool(value)
            if value:
                return Path(os.getcwd())
            else:
                return None
        else:
            raise ValueError("Invalid value. Must be a valid path or boolean.")

    def _get_path(self, subfolder: os.PathLike, filename: os.PathLike) -> Path:
        """
        Construct the full path to a file in a specified subfolder.

        Parameters
        ----------
        subfolder : str
            The subfolder name (e.g., 'inp', 'plan', 'regions', 'out').
        filename : str
            The name of the file.

        Returns
        -------
        str
            The full path to the file.

        Raises
        ------
        ValueError
            If the subfolder is invalid.
        """
        if subfolder not in ["inp", "plan", "regions", "out"]:
            raise ValueError("Invalid subfolder. Choose from 'inp', 'plan', or 'regions'.")
        base_path = self._inp_dir if subfolder == "inp" else getattr(self, f"_{subfolder}_dir")
        return Path(os.path.join(base_path, filename))

    def _write_run_file(self) -> None:
        """Write main run file for the FRED engine."""
        file_content = textwrap.dedent("""
            include: regions/regions.inp
            include: plan/planDelivery.inp
            """)
        try:
            file_path = self._get_path(self._input_foldername, self._run_input_filename)
            with open(file_path, "w") as f:
                f.write(file_content)
                # TODO: better way to handle different versions! 3.70.0 does not support dijformatversion
                if (
                    self._dij_format_version is not None
                    and not self._calc_dose_direct
                    and self.fred_version == "3.76.0"
                ):
                    f.write(f"ijFormatVersion= {self._dij_format_version}\n")
            logger.info(f"File written: {file_path}")
        except PermissionError:
            logger.warning(
                f"Permission denied when trying to write to {self._run_input_filename}."
            )
        except IOError as e:
            logger.warning(
                f"An error occurred while writing to {self._run_input_filename}. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _write_regions_file(self) -> None:
        """Write the regions file for the FRED engine, defining the phantom and room regions."""
        try:
            file_path = self._get_path(self._regions_foldername, self._regions_filename)
            with open(file_path, "w") as f:
                f.write("region<\n")
                f.write("\tID=Phantom\n")
                f.write("\tCTscan=regions/{}\n".format(self._patient_filename))
                f.write("\tO=[{:.2f},{:.2f},{:.2f}]\n".format(0, 0, 0))
                f.write("\tpivot=[0.5,0.5,0.5]\n")

                # l=e1 u=e2
                # x in Room coordinates is  x in patient frame
                # y in Romm coordinates is -y in patient frame
                # Voxels in y-direction in matRad grow in -y direction in FRED Room reference
                f.write("\tl=[{:.2f},{:.2f},{:.2f}]\n".format(1, 0, 0))
                f.write("\tu=[{:.2f},{:.2f},{:.2f}]\n".format(0, -1, 0))

                # Syntax changes for scorers according to direct or ij calculation
                if self._calc_dose_direct:
                    f.write("\tscore=[")
                else:
                    f.write("\tscoreij=[")

                f.write("{}]\n".format(",".join(self.scorers)))

                # Write Room parameters
                f.write("region>\n")
                f.write("region<\n")
                f.write("\tID=Room\n")
                f.write("\tmaterial={}\n".format(self.room_material))
                f.write("region>\n")
                f.write("include: regions/{}\n".format(self.HU_table_file))
                if self._HU_clamping:
                    f.write("lAllowHUClamping=t\n")
            logger.info(f"File written: {file_path}")
        except PermissionError:
            logger.warning(
                f"Permission denied when trying to write to {self._run_input_filename}."
            )
        except IOError as e:
            logger.warning(
                f"An error occurred while writing to {self._run_input_filename}. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _set_up_patient_ct(self, ct: CT) -> None:
        """
        Set up the patient CT data for the FRED engine by resampling and saving it.

        Parameters
        ----------
        ct : CT
            The CT object containing the patient data.
        """
        file_path = self._get_path(self._regions_foldername, self._patient_filename)

        resampled_ct = resample_ct(
            ct=ct,
            interpolator=sitk.sitkNearestNeighbor,
            target_grid=self.dose_grid,
        )

        resampled_ct.cube_hu = sitk.Cast(resampled_ct.cube_hu, sitk.sitkInt16)

        # Force Origin to (0,0,0) and Direction to Identity to match matRad behavior
        resampled_ct.cube_hu.SetOrigin((0.0, 0.0, 0.0))
        resampled_ct.cube_hu.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        sitk.WriteImage(resampled_ct.cube_hu, file_path)

        self._ct_hu_values = np.unique(sitk.GetArrayFromImage(resampled_ct.cube_hu))

    def _set_up_hu_table(self) -> None:
        """
        Write the HU table file for the FRED engine, defining the mapping of HU values to material properties.

        The table is ``hlut``, the HU to rSP lookup table the analytical engines use
        (:func:`~pyRadPlan.ct.default_hlut` unless set). Every entry is water with density
        and RSP set to the table value. With ``material_converter="hlut"`` FRED interpolates
        the table itself. With ``"RSP"`` every HU value of the CT gets its own entry with
        the value interpolated from the table, as TOPAS does in its RSP mode.
        """
        if self.hlut is None:
            self.hlut = default_hlut()
        hlut = np.asarray(self.hlut, dtype=float)
        hlut = hlut[np.argsort(hlut[:, 0])]

        if self.material_converter == "RSP":
            hus = self._ct_hu_values
            values = np.interp(hus, hlut[:, 0], hlut[:, 1])
            # Every HU of the CT has its own table entry, so nothing needs clamping
            self._HU_clamping = False
        elif self.material_converter == "hlut":
            hus, values = hlut[:, 0], hlut[:, 1]
            self._HU_clamping = bool(
                self._ct_hu_values.min() < hus[0] or self._ct_hu_values.max() > hus[-1]
            )
            if self._HU_clamping:
                logger.warning(("HU outside of boundaries"))
        else:
            raise ValueError(
                f"Unknown material_converter '{self.material_converter}' (use 'hlut' or 'RSP')."
            )

        lines = ["", "matColumns: HU rho RSP Ipot Lrad C Ca H N O P Ti S"]
        lines += [
            f"mat: {hu:5g} {value:.8g} {value:.8g} 78.0 36.1 0 0 11.189400 0 88.810600 0 0 0"
            for hu, value in zip(hus, values)
        ]
        file_content = "\n".join(lines) + "\n"
        try:
            file_path = self._get_path(self._regions_foldername, self.HU_table_file)
            with open(file_path, "w") as f:
                f.write(file_content)

        except PermissionError:
            logger.warning(
                f"Permission denied when trying to write to {self._run_input_filename}."
            )
        except IOError as e:
            logger.warning(
                f"An error occurred while writing to {self._run_input_filename}. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _write_plan_delivery_file(self) -> None:
        """Write the plan delivery file for the FRED engine, defining the delivery of pencil beams."""
        if self.source_model == "gaussian":
            layer_case_parameters = """def: currFWHM   = layer.get('FWHM')"""

            beamlet_case_parameters = """Xsec = gauss
                        FWHM 	= $currFWHM
                    """

        elif self.source_model == "emittance":
            layer_case_parameters = """def: currEmittanceX = layer.get(emittanceX)
                def: currTwissAlphaX = layer.get(twissAlphaX)
                def: currTwissBetaX = layer.get(twissBetaX)
                def: currReferencePlane = layer.get(emittanceRefPlaneDistance)
                """
            beamlet_case_parameters = """Xsec = emittance
                    emittanceX  = $currEmittanceX
                    twissAlphaX = $currTwissAlphaX
                    twissBetaX  = $currTwissBetaX
                    emittanceRefPlaneDistance = 100
                    """

        elif self.source_model == "sigmaSqrModel":
            layer_case_parameters = """def: currSQr_a = layer.get(sSQr_a)
                def: currSQr_b = layer.get(sSQr_b)
                def: currSQr_c = layer.get(sSQr_c)
                """
            beamlet_case_parameters = """Xsec = emittance
                        sigmaSqrModel = [${plan.get("
                        "SAD"
                        ")},${currSQr_a},${currSQr_b}, ${currSQr_c}]
                        """

        radiation_mode = self._machine.radiation_mode
        if radiation_mode not in FRED_PARTICLES:
            raise ValueError(
                f"FRED cannot simulate radiation mode '{radiation_mode}', only "
                f"{list(FRED_PARTICLES)}."
            )
        particle = FRED_PARTICLES[radiation_mode]
        # The machine tabulates ion energies and their spread per nucleon, which FRED reads
        # from Tu / EuFWHM; T / EFWHM are the kinetic energy of the whole ion.
        energy_key, spread_key = ("T", "EFWHM") if particle == "proton" else ("Tu", "EuFWHM")

        file_content = textwrap.dedent(f"""
        #Include file defining fields and layers geometry
        include: plan/plan.inp

        #Define the fields
        for(currField in plan.get('Fields'))<
            field<
                ID = ${{currField.get('fieldNumber')}}
                O = [0,${{plan.get('SAD')}},0]
                L = ${{currField.get('dim')}}
                pivot = [0.5,0.5,0.5]
                l = [0, 0, -1]
                u = [1, 0 ,0]
            field>

            #Deactivate the fields to avoid geometrical overlap
            deactivate: field_${{currField.get('fieldNumber')}}
        for>

        for(currField in plan.get('Fields'))<

            def: fieldIdx = currField.get('fieldNumber')

            # Activate current field
            activate: field_$fieldIdx

            #Collect Gantry and Couch angles
            def: GA = currField.get('GA')
            def: CA = currField.get('CA')

            #Collect Isocenter
            def: ISO = currField.get('ISO')

            #First move the patient so that the Isocenter is now in the center of the Room coordinate system
            transform: Phantom move_to ${{ISO.item(0)}} ${{ISO.item(1)}} ${{ISO.item(2)}} Room

            #Second rotate the patient according to the gantry and couch angles.
            transform: Phantom rotate y ${{CA}} Room
            transform: Phantom rotate z ${{GA}} Room

            for(layer in currField.get('Layers'))<
                #Recover parameters of the current energy layer
                def: currEnergy  = layer.get('Energy')
                def: currEspread = layer.get('Espread')
                {layer_case_parameters}

                for(beamlet in layer.get('beamlets'))<
                    pb<
                        ID      = ${{beamlet.get('beamletID')}}
                        fieldID = $fieldIdx
                        particle = {particle}
                        {energy_key:<4} = $currEnergy
                        {spread_key:<6} = $currEspread
                        {beamlet_case_parameters}
                        P    = ${{beamlet.get('P')}}
                        v    = ${{beamlet.get('v')}}
                        N    = ${{beamlet.get('w')}}
                    pb>
                for>
            for>

            #Deliver all the pencil beams in this field
            deliver: field_$fieldIdx

            #Deactivate the current field
            deactivate: field_$fieldIdx

            #Restore the patient to original position
            transform: Phantom rotate z ${{-1*GA}} Room
            transform: Phantom rotate y ${{-1*CA}} Room
            transform: Phantom move_to 0 0 0 Room
        for>
        """)
        try:
            file_path = self._get_path(self._plan_foldername, self._plan_delivery_filename)
            with open(file_path, "w") as file:
                file.write(file_content)

            logger.info(f"File written: {file_path}")

        except PermissionError:
            logger.warning(
                f"Warning: Permission denied when trying to write to {self._plan_delivery_filename}."
            )
        except IOError as e:
            logger.warning(
                f"Warning: An error occurred while writing to {self._plan_delivery_filename}. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _write_plan_file(self, beams_fred: list[dict]) -> None:
        """
        Write the plan file for the FRED engine, defining the fields, layers, and beamlets.

        Parameters
        ----------
        beams_fred : list[dict]
            A list of dictionaries containing the beam data for the FRED engine.
        """
        try:
            file_path = self._get_path(self._plan_foldername, self._plan_filename)
            with open(file_path, "w") as file:
                file.write("nprim = {} \n".format(self._simulated_particles_per_bixel))

                # loop over fields
                layer_counter = 0
                beamlet_counter = 0
                field_counter = 0
                for f, field in enumerate(beams_fred):
                    # loop over energy layers
                    num_layer_start_field = layer_counter
                    for ly, layer in enumerate(field["layers"]):
                        file.write("#Beamlets Field {}, Layer {} \n".format(f, ly))

                        # print bixel(aka beamlet) info (ID, Position, Direction, Weight)
                        num_beamlet_start_layer = beamlet_counter
                        for b, beamlet in enumerate(layer["pb"]):
                            file.write(
                                "\t\t def: S{0} = {{'beamletID': {1}, 'P': [{2}], 'v':[{3}], 'w':{4}}} \n".format(
                                    beamlet_counter,
                                    beamlet_counter,
                                    ", ".join(beamlet["position_bev"]),
                                    ", ".join(beamlet["divergence"]),
                                    beamlet["weight"],
                                )
                            )
                            beamlet_counter = beamlet_counter + 1

                        beamlets_in_layer = [
                            ("S" + str(n))
                            for n in list(range(num_beamlet_start_layer, beamlet_counter))
                        ]

                        # TODO: print gauss corresponding parameters  -> here FWHM
                        file.write(
                            "\t def: L{0} = {{'Energy': {1}, 'Espread': {2}, 'FWHM':{3}, 'beamlets': [{4}]}} \n".format(
                                layer_counter,
                                layer["energy"],
                                layer["ESpread"],
                                layer["FWHM"],
                                ", ".join(beamlets_in_layer),
                            )
                        )
                        file.write("\n")
                        layer_counter = layer_counter + 1

                    layers_in_field = [
                        ("L" + str(n)) for n in list(range(num_layer_start_field, layer_counter))
                    ]

                    file.write(
                        "def: F{} = {{'fieldNumber':{}, 'GA': {}, 'CA': {}, 'ISO': [{}], 'dim':[{}], 'Layers': [{}]}} \n".format(
                            f,
                            f,
                            field["GA"],
                            field["CA"],
                            ", ".join(field["ISO"]),
                            ", ".join(field["field_extent"]),
                            ", ".join(layers_in_field),
                        )
                    )
                    file.write("\n")
                    field_counter = field_counter + 1

                if field_counter == 0:
                    fields = ["F" + str(0)]
                else:
                    fields = [("F" + str(n)) for n in list(range(field_counter))]
                file.write(
                    "def: plan = {{ 'SAD': {}, 'Fields': [{}] }}\n".format(
                        field["BAMS_to_iso"], ", ".join(fields)
                    )
                )
            logger.info(f"File written: {file_path}")

        except PermissionError:
            logger.warning(
                f"Warning: Permission denied when trying to write to {self._plan_filename}."
            )
        except IOError as e:
            logger.warning(
                f"Warning: An error occurred while writing to {self._plan_filename}. Error: {e}"
            )
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _transform_to_fred_isocenter(self, dij: dict, iso_center: np.ndarray) -> list[str]:
        """
        Transform the isocenter coordinates from the dose grid to FRED's coordinate system.

        This function calculates the isocenter position in FRED's coordinate system based on the
        dose grid resolution, dimensions, and the provided isocenter coordinates.

        Parameters
        ----------
        dij : dict
            The dose influence matrix containing dose grid information.
        iso_center : np.ndarray
            The isocenter coordinates in the dose grid's world coordinate system.

        Returns
        -------
        list[str]
            The transformed isocenter coordinates in FRED's coordinate system.
        """

        dose_grid: Grid = dij["dose_grid"]
        dose_grid_resolution = np.array(
            [
                dose_grid.resolution["x"],
                dose_grid.resolution["y"],
                dose_grid.resolution["z"],
            ]
        )

        first_vox_world = np.array([dose_grid.x.min(), dose_grid.y.min(), dose_grid.z.min()])

        translation = (
            dose_grid_resolution - first_vox_world
        )  # because first_vox_cube = dose_grid_resolution

        iso_dose_grid_coord = iso_center + translation

        fred_cube_surface_in_dose_cube_coords = 0.5 * dose_grid_resolution

        fred_pivot_in_cube_coordinates = 0.5 * (
            np.array(dose_grid.dimensions) * dose_grid_resolution
        ) + np.array(fred_cube_surface_in_dose_cube_coords)

        fred_iso = -(fred_pivot_in_cube_coordinates - iso_dose_grid_coord) * (-1, 1, 1)

        return fred_iso

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> Dij:
        """
        Perform the dose calculation using the FRED engine.

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
        # The CT goes to FRED with an identity direction and the isocenter is placed in the
        # same frame, so a rotated or mirrored grid would put the anatomy in the wrong place
        if not np.allclose(self.dose_grid.direction, np.eye(3)):
            raise NotImplementedError(
                "FRED only supports CTs with an identity direction matrix, got "
                f"{self.dose_grid.direction.tolist()}. Resample the CT to an identity "
                "direction first."
            )
        counter = 0
        self.dose_cube = None
        self.let_cube = None
        self.fred_order = None
        if not self._calc_dose_direct:
            counter = 0
            for b, beam in enumerate(stf.beams):
                for j, ray in enumerate(beam.rays):
                    for k in range(len(ray.beamlets)):
                        dij["beam_num"][counter] = b
                        dij["ray_num"][counter] = j
                        dij["bixel_num"][counter] = k
                        counter += 1
        # Computed from the whole plan, so a run per beam keeps the histories per beamlet
        self._total_number_of_bixels = stf.total_number_of_bixels
        self._simulated_particles_per_bixel = (
            np.floor(max([1, self.num_histories_direct / self._total_number_of_bixels]))
            if self._calc_dose_direct
            else self.num_histories_per_beamlet
        )
        read_dir = self._read_output_dir
        # Output of a single combined run (no beam_0 folder) is read as one dose
        per_beam_output = read_dir is None or (read_dir / "beam_0").is_dir()
        if self._calc_dose_direct and self.dose_per_beam and per_beam_output:
            self._calc_dose_per_beam(dij, ct, stf)
        elif read_dir is not None:
            self._process_existing_output(dij, stf)
        else:
            self._prepare_new_calculation(dij, ct, stf)

        if read_dir is not None:
            self.fred_order = self._generate_fred_order(dij, stf)
        if self.dose_cube is None:
            if self.external_calculation is not True or read_dir is not None:
                raise FileNotFoundError(
                    f"No FRED dose output found in '{self._output_dir}' (see the log above)."
                )
            self._set_zero_dose(stf)
            self._log_external_run_instructions(stf)
        dij = self._process_dose_cube(dij)

        self._check_saving_options()

        logger.info("Finalizing dose calculation...")
        t_start = time.time()
        dij = self._finalize_dose(dij)
        t_end = time.time()
        logger.info("Done in %f seconds.", t_end - t_start)

        return dij

    @property
    def _read_output_dir(self) -> Optional[Path]:
        """Folder FRED's output is read from instead of running FRED, None to run it.

        This is `use_output`, or the folder given as `external_calculation` (as in TOPAS).
        """
        if self._use_output is not None:
            return Path(self._use_output)
        if isinstance(self.external_calculation, (str, os.PathLike)):
            return Path(self.external_calculation)
        return None

    def _set_zero_dose(self, stf: SteeringInformation) -> None:
        """Zero dose of the shape FRED's output would have, for a write-only calculation."""
        num_voxels = self.dose_grid.num_voxels
        if self._calc_dose_direct:
            zero = np.zeros(num_voxels)
            self.dose_cube = [zero] * len(stf.beams) if self.dose_per_beam else zero
        else:
            self.dose_cube = sparse.csc_array((num_voxels, stf.total_number_of_bixels))
            self.fred_order = np.arange(stf.total_number_of_bixels)
        self.let_cube = None

    def _log_external_run_instructions(self, stf: SteeringInformation) -> None:
        root = self._fred_root_folder
        if self._calc_dose_direct and self.dose_per_beam:
            runs = [f"beam_{b}" for b in range(len(stf.beams))]
        else:
            runs = [None]
        commands = []
        for run in runs:
            inp, out = root / self._input_foldername, root / self._output_folder
            if run is not None:
                inp, out = inp / run, out / run
            commands.append(f'  cd "{inp}" && {self.fred_cmd} -f fred.inp -o "{out}"')
        logger.info(
            "FRED input files written to '%s', FRED was not run and the dose is zero. Run:\n"
            "%s\nthen read the dose with external_calculation='%s'.",
            root / self._input_foldername,
            "\n".join(commands),
            root / self._output_folder,
        )

    def _process_existing_output(self, dij: dict, stf: SteeringInformation) -> None:
        """
        Process existing output files from `use_output` or an `external_calculation` folder.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        stf : SteeringInformation
            The steering information for the beams.
        """
        self._output_dir = self._read_output_dir
        self._read_output_files()

    def _prepare_new_calculation(self, dij: dict, ct: CT, stf: SteeringInformation) -> None:
        """
        Prepare for a new dose calculation by setting up input files and calling FRED.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        ct : CT
            The CT object containing the patient data.
        stf : SteeringInformation
            The steering information for the beams.
        """
        beams_fred = self._generate_beams_fred(dij, stf)
        self._write_plan_file(beams_fred)
        self._set_up_files(ct)

        if self.external_calculation is True:
            self.dose_cube = None
        else:
            self._call_fred()
            if not self._calc_dose_direct:
                self.fred_order = self._generate_fred_order(dij, stf)
            self._read_output_files()

    def _calc_dose_per_beam(self, dij: dict, ct: CT, stf: SteeringInformation) -> None:
        """
        Run (or read with `use_output`) one forward FRED calculation per beam.

        The inputs and output of beam b are in inp/beam_b and out/beam_b. `dose_cube` and
        `let_cube` become lists with one cube per beam, or `dose_cube` None when the
        calculation is external.
        """
        dose_cubes, let_cubes = [], []
        for b, beam in enumerate(stf.beams):
            run_name = f"beam_{b}"
            if self._read_output_dir is not None:
                self._output_dir = self._read_output_dir / run_name
                self._read_output_files()
            else:
                self._use_run_folder(run_name)
                beam_stf = stf.model_copy(update={"beams": [beam]})
                self._prepare_new_calculation(dij, ct, beam_stf)
            dose_cubes.append(self.dose_cube)
            let_cubes.append(self.let_cube)

        if any(cube is None for cube in dose_cubes):
            self.dose_cube = None
            self.let_cube = None
        else:
            self.dose_cube = dose_cubes
            self.let_cube = None if any(cube is None for cube in let_cubes) else let_cubes

    def _use_run_folder(self, name: str) -> None:
        """Write the inputs of the next FRED run to inp/<name> and its output to out/<name>."""
        self._inp_dir = self._fred_root_folder / self._input_foldername / name
        self._plan_dir = self._inp_dir / self._plan_foldername
        self._regions_dir = self._inp_dir / self._regions_foldername
        self._output_dir = self._fred_root_folder / self._output_folder / name
        os.makedirs(self._plan_dir, exist_ok=True)
        os.makedirs(self._regions_dir, exist_ok=True)
        # FRED creates only the last level of its output folder
        os.makedirs(self._output_dir.parent, exist_ok=True)

    def _generate_fred_order(self, dij: dict, stf: SteeringInformation) -> np.ndarray:
        """
        Generate the FRED order mapping for scorer-ij mode.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        stf : SteeringInformation
            The steering information for the beams.
        """
        counter_fred = 0
        fred_order = np.full(dij["total_num_of_bixels"], np.nan)

        for i, beam in enumerate(stf.beams):
            for j, (energy_key, energy_layer) in enumerate(beam.energy_layers.items()):
                for k in range(len(energy_layer["rays_idx"])):
                    ix = np.where(
                        (i == dij["beam_num"])
                        & np.isin(dij["ray_num"], energy_layer["rays_idx"][k])
                        & np.isin(dij["bixel_num"], energy_layer["beamlet_idx"][k])
                    )
                    fred_order[ix] = counter_fred
                    counter_fred += 1
        return fred_order

    def _generate_beams_fred(self, dij: dict, stf: SteeringInformation) -> list[dict]:
        """
        Generate the FRED-compatible beam data structure.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        stf : SteeringInformation
            The steering information for the beams.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the beam data for FRED.
        """
        beams_fred = []

        for b, beam in enumerate(stf.beams):
            fred_iso = self._transform_to_fred_isocenter(dij, beam.iso_center)
            beams_dict = {
                "fieldID": b,
                "origin": beam.source_point_bev / 10,
                "GA": beam.gantry_angle,
                "CA": beam.couch_angle,
                "field_extent": [],
                "ISO": (str(n / 10) for n in fred_iso),
                "BAMS_to_iso": cast(IonAccelerator, self._machine).bams_to_iso_dist / 10,
                "layers": self._generate_layers_fred(beam, b),
            }
            fwhm_max = max(
                [beams_dict["layers"][k]["FWHM"] for k in range(len(beams_dict["layers"]))]
            )
            field_extent = [ray.ray_pos for ray in beam.rays]
            enclosing_radius_margin = 10
            field_extent = max(np.array(np.amax(field_extent, axis=1))) / 10 + 10 * fwhm_max

            # making sure no sample is smaller than the enclosing radius margin
            if field_extent <= enclosing_radius_margin:
                field_extent = enclosing_radius_margin * 2

            beams_dict["field_extent"] = [str(field_extent), str(field_extent), str(0.1)]
            beams_fred.append(beams_dict)

        return beams_fred

    def _generate_layers_fred(self, beam: Beam, beam_index: int) -> list[dict]:
        """
        Generate the FRED-compatible layers for a beam.

        Parameters
        ----------
        beam : Beam
            The beam object.
        beam_index : int
            The index of the beam.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the layer data for FRED.
        """
        layers_fred = []
        bams_to_iso = cast(IonAccelerator, self._machine).bams_to_iso_dist / 10

        # Linear projection of BEV source (x,y) points to plane at BAMStoISO distance
        def get_point_at_bams(target, source, distance, bams_to_iso):
            return (target - source) * (-bams_to_iso) / distance + source

        for key, value in beam.energy_layers.items():
            pb_fred = []
            for i in range(len(value["beamlet_idx"])):
                ray = beam.rays[value["rays_idx"][i]]
                target_point_bev = ray.target_point_bev
                if target_point_bev is None:
                    target_point_bev = 2 * ray.ray_pos_bev - beam.source_point_bev
                target_point = target_point_bev / 10
                # source_point = beam.source_point_bev / 10 #seems to not be needed in FRED
                ray_position = ray.ray_pos_bev / 10
                distance = target_point[1] - ray_position[1]

                # Project the spot to the entrance plane (BAMStoISO distance)
                ray_pos_z_proj = get_point_at_bams(
                    target_point[2], ray_position[2], distance, bams_to_iso
                )
                ray_pos_x_proj = get_point_at_bams(
                    target_point[0], ray_position[0], distance, bams_to_iso
                )

                pb_dict = {
                    "ID": i,
                    "fieldID": beam_index,
                    "particle": beam.radiation_mode,
                    "T": value["full_energy"],
                    "position_bev": [str(ray_pos_z_proj), str(ray_pos_x_proj), str(0)],
                    "divergence": [
                        str((target_point[2] - ray_position[2]) / distance),
                        str((target_point[0] - ray_position[0]) / distance),
                        str(1),
                    ],
                    "weight": beam.rays[value["rays_idx"][i]]
                    .beamlets[value["beamlet_idx"][i]]
                    .weight,
                    "Xsec": "gauss",
                }

                if self._calc_dose_direct:
                    pb_dict["weight"] *= self.scaling_factor

                pb_fred.append(pb_dict)

            machine = cast(IonAccelerator, self._machine)
            emittance = machine.foci[value["full_energy"]][0].emittance
            if emittance is None:
                raise ValueError(
                    f"FRED needs the beam emittance of the machine's foci, which the "
                    f"{machine.radiation_mode} machine '{machine.name}' does not provide "
                    f"(energy {value['full_energy']} MeV/u)."
                )
            # The spectrum's mean energy is the one Monte Carlo reproduces the measured depth
            # dose with, as in TOPAS; without a spectrum the beam is monoenergetic
            spectrum = (machine.spectra or {}).get(value["full_energy"])
            layer_dict = {
                "energy": value["full_energy"] if spectrum is None else spectrum.mean,
                "ESpread": 0.0 if spectrum is None else spectrum.fwhm,
                "FWHM": (
                    2.355
                    * (2 * emittance.sigma_x * emittance.sigma_y)
                    / (emittance.sigma_x + emittance.sigma_y)
                )
                / 10,
                "pb": pb_fred,
            }
            layers_fred.append(layer_dict)

        return layers_fred

    def _process_dose_cube(self, dij: dict) -> None:
        """
        Process the dose cube and update the dose influence matrix.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        """

        if self._calc_dose_direct:
            # Direct dose calculation: one column per beam with dose_per_beam, else one column
            per_beam = isinstance(self.dose_cube, list)
            dose_cubes = self.dose_cube if per_beam else [self.dose_cube]
            dose = np.column_stack([cube.ravel() for cube in dose_cubes])
            dij["physical_dose"].flat[0] = self._mask_outside_patient(sparse.csc_array(dose))

            if self._use_let_kernel and self.let_cube is not None:
                let_cubes = self.let_cube if per_beam else [self.let_cube]
                let = np.column_stack([cube.ravel() for cube in let_cubes])
                # LETd * dose; FRED scores LETd in MeV * cm^2 / g, hence the division by 10
                let_dose = np.zeros_like(dose)
                let_dose[self._vdose_grid] = let[self._vdose_grid] / 10 * dose[self._vdose_grid]
                self._store_let_quantities(dij, sparse.csc_array(let_dose))

            if per_beam:
                num_beams = len(dose_cubes)
                dij["num_of_beams"] = num_beams
                dij["beam_num"] = np.arange(num_beams)
                dij["ray_num"] = np.arange(num_beams)
                dij["bixel_num"] = np.arange(num_beams)
            else:
                dij_fields_to_override = [
                    "num_of_beams",
                    "total_number_of_bixels",
                    "total_number_of_rays",
                    "num_of_rays_per_beam",
                ]
                for field_name in dij_fields_to_override:
                    dij[field_name] = 1

                dij["beam_num"] = 0
                dij["bixel_num"] = 0
                dij["ray_num"] = 0
        else:
            # Scorer-ij mode
            self.dose_cube = sparse.csc_array(self.dose_cube)
            self.dose_cube = self.dose_cube[:, self.fred_order]

            dij["physical_dose"].flat[0] = self._mask_outside_patient(
                self._conversion_factor * self.dose_cube
            )

            if self._use_let_kernel and self.let_cube is not None:
                self.let_cube = sparse.csc_array(self.let_cube)
                self.let_cube = self.let_cube[:, self.fred_order]

                # LETd * dose; FRED scores LETd in MeV * cm^2 / g, hence the division by 10
                let_dose = dij["physical_dose"].flat[0] * (self.let_cube / 10)
                self._store_let_quantities(dij, let_dose)

        return dij

    def _mask_outside_patient(self, matrix: sparse.csc_array) -> sparse.csc_array:
        """Zero the rows of voxels outside all cst structures if requested."""
        if not self.ignore_outside_patient:
            return matrix
        masked = sparse.csc_array(matrix.multiply(self._vdose_grid_mask[:, np.newaxis]))
        masked.eliminate_zeros()
        return masked

    def _store_let_quantities(self, dij: dict[str, Any], let_dose: sparse.csc_array) -> None:
        """Store the LET matrix and any biological matrices derived from it."""
        if self._calc_let:
            dij["let_dose"].flat[0] = let_dose
        if self._calc_bio_dose:
            bio_influence = bio_influence_from_let(
                self._bio_evaluator,
                dij["physical_dose"].flat[0],
                let_dose,
                np.asarray(dij["alphax"])[:, 0],
                np.asarray(dij["betax"])[:, 0],
            )
            for name, matrix in bio_influence.items():
                dij[name].flat[0] = matrix

    def _check_saving_options(self) -> None:
        if self._save_input is not None:
            if not os.path.isdir(self._save_input):
                raise ValueError(
                    f"The provided path '{self._save_input}' for 'save_input' is not a valid directory."
                )
            elif Path(self._save_input).resolve() != self._fred_root_folder.resolve():
                shutil.copytree(
                    self._fred_root_folder / "inp", self._save_input / "inp", dirs_exist_ok=True
                )
        if self._save_output is not None:
            if not os.path.isdir(self._save_output):
                raise ValueError(
                    f"The provided path '{self._save_output}' for 'save_output' is not a valid directory."
                )
            elif (
                Path(self._save_output).resolve() != self._fred_root_folder.resolve()
                and (self._fred_root_folder / "out").is_dir()
            ):
                shutil.copytree(
                    self._fred_root_folder / "out", self._save_output / "out", dirs_exist_ok=True
                )

        if self._fred_root_is_temp:
            shutil.rmtree(self._fred_root_folder)
            logger.info((f"Temp tree directory deleted: {self._fred_root_folder}"))
        else:
            logger.info(f"FRED simulation files kept in: {self._fred_root_folder}")

    def _set_up_files(self, ct: CT) -> None:
        """
        Set up the necessary input files for the FRED engine.

        Parameters
        ----------
        ct : CT
            The CT object containing the patient data.
        """
        self._set_up_patient_ct(self._scenario_ct(ct))
        self._set_up_hu_table()
        self._write_run_file()
        self._write_regions_file()
        self._write_plan_delivery_file()

    def _read_output_files(self) -> None:
        """Read the output files generated by the FRED engine. Handle both direct and scorer-ij modes."""
        dose_cube = None
        let_cube = None

        if not self._calc_dose_direct:
            # Handle scorer-ij mode
            dose_dij_folder = os.path.join(self._output_dir, "scoreij")
            dose_dij_file = "Phantom.Dose.bin"
            load_file_name = os.path.join(dose_dij_folder, dose_dij_file)

            logger.info(f"Looking for scorer-ij output in sub folder: {dose_dij_folder}")

            if os.path.isfile(load_file_name):
                dose_cube = self.read_sparse_dij_bin(load_file_name)
            else:
                logger.error(f"Unable to find file: {load_file_name}")

            if self._use_let_kernel:
                letd_dij_file = "Phantom.LETd.bin"
                letd_file_name = os.path.join(dose_dij_folder, letd_dij_file)

                try:
                    let_cube = self.read_sparse_dij_bin(letd_file_name)
                except Exception as e:
                    raise ValueError(self._missing_let_message(letd_file_name)) from e

        else:
            # Handle direct dose calculation mode
            dose_cube_folder = os.path.join(self._output_dir, "score")
            dose_cube_file_name = "Phantom.Dose.mhd"
            load_file_name = os.path.join(dose_cube_folder, dose_cube_file_name)

            logger.info(f"Looking for scorer file in sub folder: {dose_cube_folder}")

            if os.path.isfile(load_file_name):
                dose_cube = sitk.GetArrayFromImage(sitk.ReadImage(load_file_name))
            else:
                logger.error(f"Unable to find file: {load_file_name}")

            if self._use_let_kernel:
                letd_file_name = os.path.join(dose_cube_folder, "Phantom.LETd.mhd")

                try:
                    let_cube = sitk.GetArrayFromImage(sitk.ReadImage(letd_file_name))
                except Exception as e:
                    raise ValueError(self._missing_let_message(letd_file_name)) from e

        # Store the results for further processing

        self.dose_cube = dose_cube
        self.let_cube = let_cube

    @staticmethod
    def _missing_let_message(file_name: str) -> str:
        return (
            f"FRED LETd output could not be read from '{file_name}': the biological model or "
            "calc_let needs the LETd scorer. Rerun FRED with the LETd scorer enabled."
        )

    def read_sparse_dij_bin(self, f_name: str) -> sparse.csc_array:
        """
        Dispatch method to read a sparse dij binary file based on the dij format version.

        Parameters
        ----------
            f_name (str): File name to read.

        Returns
        -------
            sparse.csc_array: Sparse matrix containing the dij data.
        """
        with open(f_name, "rb") as f:
            file_format_version = str(np.frombuffer(f.read(4), dtype=np.int32)[0])
        logger.info(f"File format version: {file_format_version}")
        self._dij_format_version = file_format_version
        if file_format_version == "20":
            return read_sparse_dij_bin_v20(f_name)
        elif file_format_version == "21":
            return read_sparse_dij_bin_v21(f_name)
        elif file_format_version == "31":
            return read_sparse_dij_bin_v31(f_name)
        else:
            raise ValueError(f"Unsupported dij format version: {self._dij_format_version}")

    def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]) -> Dij:
        # Loop over all requested quantities
        for q_name in names:
            # Create dij list for each quantity
            dij[q_name] = np.empty((len(self._scenarios),), dtype=object)

            # Loop over all scenarios and preallocate quantity containers
            # TODO: write test for this
            for i in self._scenarios.computed_indices():
                if self._calc_dose_direct:
                    dij[q_name].flat[i] = np.zeros(
                        (self.dose_grid.num_voxels, dij["num_of_beams"]), dtype=np.float32
                    )
                # else:
                #     # This could probably be optimized by using direct access to the
                #     # lil_matrix's data structures
                #     # TODO: we could store a single sparse pattern matrix and then only store
                #     # the values for all quantities for better memory management
                #     dij[q_name].flat[i] = lil_matrix(
                #         (self._num_of_columns_dij, self.dose_grid.num_voxels),
                #         dtype=np.float32,
                #     )

            self._computed_quantities.append(q_name)

        return dij

    def _call_fred(self) -> None:
        """
        Call the FRED executable to perform the dose calculation.

        Raises
        ------
        RuntimeError
            If the FRED execution fails.
        """
        output_path = os.path.abspath(self._output_dir)
        no_gpu = ""

        if not self.use_gpu:
            no_gpu = "-nogpu"
        try:
            logger.info("Running Simulation in FRED...")
            t_start = time.time()
            execute_cmd = (
                f'{self.fred_cmd} -f fred.inp -o "{output_path}" -i "{self._inp_dir}" {no_gpu}'
            )
            logger.info("FRED command: %s", execute_cmd)
            self._run_fred_process(execute_cmd)
        except subprocess.TimeoutExpired as e:
            logger.error(
                "FRED did not finish within execution_timeout=%s s (GPU busy or hung?).",
                self.execution_timeout,
            )
            raise RuntimeError(
                f"FRED did not finish within execution_timeout={self.execution_timeout} s"
            ) from e
        except subprocess.CalledProcessError as e:
            stdout_msg = e.stdout.decode(errors="ignore") if e.stdout else ""
            stderr_msg = e.stderr.decode(errors="ignore") if e.stderr else ""
            logger.error(
                "FRED execution failed with return code %s. stdout:\n%s\nstderr:\n%s",
                e.returncode,
                stdout_msg,
                stderr_msg or "No error details available",
            )
            raise RuntimeError(f"FRED execution failed with return code {e.returncode}") from e
        except FileNotFoundError as e:
            logger.error(
                "FRED executable not found: %s. Ensure 'fred' is in PATH.",
                e,
            )
            raise RuntimeError(f"FRED executable not found: {e}") from e
        else:
            t_end = time.time()
            logger.info("Done in %f seconds.", t_end - t_start)

    def _run_fred_process(self, execute_cmd: str) -> None:
        """Run the FRED command, enforcing ``execution_timeout`` on the process tree.

        ``fred`` is a wrapper (a ``.BAT`` on Windows) run through the shell, so a
        plain ``subprocess.run(timeout=...)`` would only kill the wrapper and leave
        the actual ``Fred.x`` simulation occupying the GPU. On timeout the whole
        process tree is terminated instead.

        Raises
        ------
        subprocess.TimeoutExpired
            If FRED does not finish within ``execution_timeout`` seconds.
        subprocess.CalledProcessError
            If FRED exits with a non-zero return code.
        """
        capture = subprocess.PIPE if not self.print_output else None
        popen_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            # New session so the shell and all its children share a process group.
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(  # noqa: S602 - command assembled from engine config
            execute_cmd,
            cwd=self._inp_dir,
            env=os.environ.copy(),
            shell=True,
            stdout=capture,
            stderr=capture,
            **popen_kwargs,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.execution_timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                import signal  # noqa: PLC0415

                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
            raise

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, execute_cmd, output=stdout, stderr=stderr
            )

    def _request_let_scoring(self) -> None:
        if "LETd" not in self.scorers:
            self.scorers = [*self.scorers, "LETd"]

    def _create_working_dirs(self) -> None:
        """Create the folder tree of one calculation.

        Without `simu_dir` it is a temporary folder deleted at the end of the calculation,
        or for a write-only `external_calculation` (without `save_input`) a new folder below
        the pyRadPlan data folder, as in TOPAS. In `simu_dir` only the inp/ and out/ folders of a previous
        calculation are replaced.
        """
        self._fred_root_is_temp = False
        if self.simu_dir is not None:
            self._fred_root_folder = self.simu_dir
        elif self.external_calculation is True and self._save_input is None:
            parent = get_data_dir() / "fred"
            parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # mkdir without exist_ok claims the name atomically, so jobs started in the
            # same second never share (and clean up) each other's tree
            for n in itertools.count():
                folder = parent / (stamp if n == 0 else f"{stamp}_{n}")
                try:
                    folder.mkdir()
                    break
                except FileExistsError:
                    continue
            self._fred_root_folder = folder
        else:
            self._fred_root_is_temp = True
            self._fred_root_folder = Path(tempfile.mkdtemp())
        self._inp_dir = self._fred_root_folder / self._input_foldername
        self._plan_dir = self._inp_dir / self._plan_foldername
        self._output_dir = self._fred_root_folder / self._output_folder
        self._regions_dir = self._inp_dir / self._regions_foldername
        # Stale files of a previous calculation would be read as this one's output
        for folder in (self._inp_dir, self._output_dir):
            if folder.exists() and not self._holds_use_output(folder):
                shutil.rmtree(folder)
        os.makedirs(self._plan_dir)
        os.makedirs(self._regions_dir)

    def _holds_use_output(self, folder: Path) -> bool:
        if self._read_output_dir is None:
            return False
        use_output = self._read_output_dir.resolve()
        return use_output == folder.resolve() or folder.resolve() in use_output.parents

    def _init_dose_calc(
        self, ct: CT, cst: StructureSet, stf: SteeringInformation
    ) -> dict[str, Any]:
        if self._fred_root_is_temp and self._fred_root_folder.exists():
            # Failed simulations can leave partial scorer files as well as inputs. Start every
            # attempt with a clean tree so a retry cannot read or save an earlier run's output.
            # Only our temporary tree is removed; `use_output` belongs to the caller.
            shutil.rmtree(self._fred_root_folder)
        self._create_working_dirs()

        dij = super()._init_dose_calc(ct, cst, stf)
        dij = self._allocate_quantity_matrices(dij, ["physical_dose"])

        radiation_mode = self._machine.radiation_mode
        if radiation_mode != "protons":
            message = (
                f"FRED's {radiation_mode} simulations are not validated: their doses do not "
                "reproduce the matRad / pencil-beam reference (the oxygen test case is about 20 "
                "times too high). Use the results with caution."
            )
            logger.warning(message)
            self.report_warning(message)

        if dij["num_of_scenarios"] > 1:
            raise NotImplementedError(
                "Multiple scenarios are not supported for FRED calculations."
            )

        return self._init_bio_model(dij)

    def _finalize_dose(self, dij: dict) -> None:
        """
        Finalize the dose influence matrix.

        Pruning the matrix and concatenating the containers to a compressed
        sparse matrix.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.

        Returns
        -------
        Dij
            The finalized dose influence matrix.
        """
        if self.external_calculation is not True:
            # Loop over all scenarios and remove dose influence for voxels outside of segmentations
            for i in self._scenarios.computed_indices():
                # Loop over all used quantities
                for q_name in self._computed_quantities:
                    if not self._calc_dose_direct:
                        tmp_matrix = dij[q_name].flat[i]

                        if not isinstance(tmp_matrix, sparse.csc_array):
                            tmp_matrix = sparse.csc_array(tmp_matrix)

                        tmp_matrix.eliminate_zeros()
                        dij[q_name].flat[i] = tmp_matrix
                        # dij[q_name].flat[i].data *= self.scaling_factor

            # if self.keep_rad_depth_cubes and self._rad_depth_cubes:
            #     dij["rad_depth_cubes"] = self._rad_depth_cubes

            # Call the finalizeDose method from the base class
        return super()._finalize_dose(dij)

    @staticmethod
    def is_available(pln, machine):
        available, msg = True, ""
        return available, msg


def read_sparse_dij_bin_v20(f_name: str) -> sparse.csc_array:
    """
    Read a sparse dij binary file in a Matlab-like format.

    Parameters
    ----------
        fName (str): File name to read.

    Returns
    -------
        sparse.csc_array: Sparse matrix containing the dij data.
    """
    with open(f_name, "rb") as f:
        # Read header
        _ = np.frombuffer(f.read(4), dtype=np.int32)[0]  # file_for_mat_versio
        dims = np.frombuffer(f.read(4 * 3), dtype=np.int32)
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # res
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # offset
        n_components = np.frombuffer(f.read(4), dtype=np.int32)[0]
        number_of_beamlets = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Preallocate lists to collect arrays
        all_values_list = []
        all_voxel_inds_list = []
        indptr = [0]
        beamlet_counter = 0

        for _ in range(number_of_beamlets):
            # Read beamlet header
            _ = np.frombuffer(f.read(4), dtype=np.int32)[0]  # bix_num (unused)
            num_vox = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            beamlet_counter += 1

            # Update indptr
            indptr.append(int(indptr[-1]) + num_vox)
            # Read voxel indices
            curr_voxel_indices = np.frombuffer(f.read(4 * num_vox), dtype=np.uint32)
            all_voxel_inds_list.append(curr_voxel_indices)

            # Read the values for the current beamlet
            tmp_values = np.frombuffer(f.read(4 * num_vox * n_components), dtype=np.float32)
            if n_components == 2:
                values = tmp_values[0::2] / tmp_values[1::2]
            else:
                values = tmp_values[0::n_components]
            all_values_list.append(values)

        # Concatenate the lists into single NumPy arrays
        all_values = np.concatenate(all_values_list)
        all_voxel_inds = np.concatenate(all_voxel_inds_list)
        indptr = np.array(indptr, dtype=np.int64)

    total_voxels = int(np.prod(dims))
    dij_matrix = sparse.csc_array(
        (all_values, all_voxel_inds, indptr), shape=(total_voxels, beamlet_counter)
    )
    return dij_matrix


def read_sparse_dij_bin_v21(f_name: str) -> sparse.csc_array:
    """
    Read a sparse dij binary file in version 2.1 format.

    Parameters
    ----------
        f_name (str): File name to read.

    Returns
    -------
        sparse.csc_array: Sparse matrix containing the dij data.
    """
    with open(f_name, "rb") as f:
        # Read header
        _ = np.frombuffer(f.read(4), dtype=np.int32)[0]  # file_format_version
        dims = np.frombuffer(f.read(4 * 3), dtype=np.int32)
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # res
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # offset
        _ = np.frombuffer(f.read(4 * 9), dtype=np.float32)  # orientation
        n_components = np.frombuffer(f.read(4), dtype=np.int32)[0]
        number_of_beamlets = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Preallocate lists to collect arrays
        all_values_list = []
        all_voxel_inds_list = []
        indptr = [0]
        beamlet_counter = 0

        for _ in range(number_of_beamlets):
            # Read beamlet header
            _ = np.frombuffer(f.read(4), dtype=np.uint32)[0]  # bix_num
            num_vox = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

            beamlet_counter += 1

            # Update indptr
            indptr.append(int(indptr[-1]) + num_vox)

            # Read voxel indices
            curr_voxel_indices = np.frombuffer(f.read(4 * num_vox), dtype=np.uint32)

            # Read the values for the current beamlet
            tmp_values = np.frombuffer(f.read(4 * num_vox * n_components), dtype=np.float32)
            values_nom = tmp_values[0::n_components]

            if n_components == 2:
                values_den = tmp_values[1::n_components]
                values = values_nom / values_den
            else:
                values = values_nom

            # FRED's raw voxel index already matches pyRadPlan's canonical (z, y, x)
            # C-order flat-index convention (X fastest, Z slowest) for `dims` = (nx, ny, nz)
            # taken directly from the header -- no axis permutation needed.
            all_voxel_inds_list.append(curr_voxel_indices)
            all_values_list.append(values)

    # Concatenate the lists into single NumPy arrays
    all_values = np.concatenate(all_values_list)
    all_voxel_inds = np.concatenate(all_voxel_inds_list)
    indptr = np.array(indptr, dtype=np.int64)

    total_voxels = int(np.prod(dims))
    dij_matrix = sparse.csc_array(
        (all_values, all_voxel_inds, indptr), shape=(total_voxels, beamlet_counter)
    )
    return dij_matrix


def read_sparse_dij_bin_v31(f_name: str) -> sparse.csc_array:
    """
    Read a sparse dij binary file in version 3.1 format.

    Parameters
    ----------
        f_name (str): File name to read.

    Returns
    -------
        sparse.csc_array: Sparse matrix containing the dij data.
    """
    with open(f_name, "rb") as f:
        # Read header
        _ = np.frombuffer(f.read(4), dtype=np.int32)[0]  # file_format_version
        dims = np.frombuffer(f.read(4 * 3), dtype=np.int32)
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # res
        _ = np.frombuffer(f.read(4 * 3), dtype=np.float32)  # offset
        _ = np.frombuffer(f.read(4 * 9), dtype=np.float32)  # orientation
        n_components = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        number_of_beamlets = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        # Read beamlet metadata
        all_bixel_meta = np.frombuffer(f.read(4 * 3 * number_of_beamlets), dtype=np.uint32)
        all_bixel_meta = all_bixel_meta.reshape((number_of_beamlets, 3))  # (PBidx, FID, PBID)

        # Read component data sizes
        component_data_sizes = [
            np.frombuffer(f.read(4), dtype=np.uint32)[0] for _ in range(n_components)
        ]

        # Preallocate lists for data
        pb_idxs = []
        voxel_indices = []
        tmp_values = []

        # Read data for each component
        for comp_idx in range(n_components):
            pb_idxs.append(
                np.frombuffer(f.read(4 * component_data_sizes[comp_idx]), dtype=np.uint32)
            )
            voxel_indices.append(
                np.frombuffer(f.read(4 * component_data_sizes[comp_idx]), dtype=np.uint32)
            )
            tmp_values.append(
                np.frombuffer(f.read(4 * component_data_sizes[comp_idx]), dtype=np.float32)
            )

    # Compute values based on components
    if n_components > 1:
        values = tmp_values[0] / tmp_values[1]
    else:
        values = tmp_values[0]

    # FRED's raw voxel index already matches pyRadPlan's canonical (z, y, x) C-order
    # flat-index convention (X fastest, Z slowest) for `dims` = (nx, ny, nz) taken
    # directly from the header -- no axis permutation needed.
    voxel_indices = voxel_indices[0]

    # Create sparse matrix
    total_voxels = int(np.prod(dims))

    # Optimization: Construct CSC directly using indptr
    # We assume the data in the file is sorted by beamlet index (pb_idxs), which is standard.
    # This avoids the overhead of coo_array -> csc_array conversion.
    counts = np.bincount(pb_idxs[0], minlength=number_of_beamlets)
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)

    dij_matrix = sparse.csc_array(
        (values, voxel_indices, indptr),
        shape=(total_voxels, number_of_beamlets),
    )

    return dij_matrix
