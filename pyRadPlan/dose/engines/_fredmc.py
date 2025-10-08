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

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Union, cast
import time
import textwrap
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.sparse import coo_matrix, lil_matrix

from pyRadPlan.ct import CT, resample_ct
from pyRadPlan.cst import StructureSet
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from pyRadPlan.stf import SteeringInformation
from ._base_montecarlo import MonteCarloEngineAbstract
from pyRadPlan.util import swap_orientation_sparse_matrix

logger = logging.getLogger(__name__)


class ParticleFredMCEngine(MonteCarloEngineAbstract):
    # constants

    short_name = "FRED"
    name = "FRED"
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen16"]  # add more if needed

    available_source_models = ["gaussian", "emittance", "sigmaSqrModel"]

    available_versions = ["3.70.0", "3.76.0"]

    external_calculation: Union[str, bool]
    calc_bio_dose: bool
    calc_let: bool

    fred_version: str
    fred_cmd: str
    fred_dir: Path
    use_gpu: bool

    save_input: Union[os.PathLike, bool]
    save_output: Union[os.PathLike, bool]
    use_output: Union[os.PathLike, bool]
    print_output: bool

    scorers: list[str]
    source_model: str
    room_material: str
    HU_table_file: str
    scaling_factor: int

    def __init__(self, pln: Union[Plan, dict] = None):
        ### Public attributes ###
        # properties with setters and getters
        self.save_input = False
        self.save_output = False
        self.use_output = False
        self.source_model = "gaussian"
        self.external_calculation = False

        self.use_gpu = True
        self.calc_let = False
        self.calc_bio_dose = False
        self.scorers = ["Dose"]
        self.source_model = "gaussian"
        self.room_material = "Air"
        self.HU_table_file = "hLut.inp"
        self.scaling_factor = 1e6
        self.fred_version = "3.70.0"

        self.fred_cmd = "fred"
        self.fred_dir = Path(os.environ.get("FREDDIR", ""))

        ### Private attributes ###
        self._computed_quantities = []
        self._total_number_of_bixels = None
        self._dij_format_version = None
        self._fred_root_folder = Path(tempfile.mkdtemp())
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
        self._inp_dir = self._fred_root_folder / self._input_foldername
        self._plan_dir = self._inp_dir / self._plan_foldername
        self._output_dir = self._fred_root_folder / self._output_folder
        self._regions_dir = self._inp_dir / self._regions_foldername
        self._conversion_factor = 1e6
        self._HU_clamping: bool = False
        self._max_hu_table_value: int = 3000  # TODO: set up max hu value using custom hu table
        self._simulated_particles_per_bixel = 1e2

        os.makedirs(self._plan_dir)
        os.makedirs(self._regions_dir)

        super().__init__(pln)

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

                if len(self.scorers) > 1:
                    for scorer in self.scorers:
                        f.write("{},".format(scorer))
                f.write("{}]\n".format(self.scorers[-1]))

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

        sitk.WriteImage(resampled_ct.cube_hu, file_path)

        max_ct_hu_value = sitk.GetArrayFromImage(resampled_ct.cube_hu).max()

        if max_ct_hu_value > self._max_hu_table_value:
            self._HU_clamping = True
            logger.warning(("HU outside of boundaries"))

    def _set_up_hu_table(self) -> None:
        """Write the HU table file for the FRED engine, defining the mapping of HU values to material properties."""
        file_content = textwrap.dedent("""
        matColumns: HU rho RSP Ipot Lrad C Ca H N O P Ti S
        mat: -1024  0.001 0.001  78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:  -999  0.001 0.0011 78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:   -90   0.95 0.95   78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:   -45   0.99 0.99   78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:     0      1 1      78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:   100  1.095 1.095  78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:   350  1.199 1.199  78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        mat:  3000  2.505 2.505  78.0 36.1 0 0 11.189400 0 88.810600 0 0 0
        """)
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

        if self._machine.radiation_mode == "protons":
            particle = """proton"""
        elif self._machine.radiation_mode == "carbon":
            particle = """C12"""

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
                        T    = $currEnergy
                        EFWHM  = $currEspread
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

                        # TODO: print gaus corresponding parameters  -> here FWHM
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
                    "def: plan = {{ 'SAD': {}, 'Fields': [{}] }}".format(
                        field["BAMS_to_iso"], ", ".join(fields)
                    )
                )
            logger.info(f"File written: {file_path}")

        except PermissionError:
            logger.warning(
                f"Warning: Permission denied when trying to write to {self.plan_filename}."
            )
        except IOError as e:
            logger.warning(
                f"Warning: An error occurred while writing to {self.plan_filename}. Error: {e}"
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

        dose_grid = dij["dose_grid"]
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
        counter = 0
        self.dose_cube = None
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
        if self._use_output is not None:
            self._process_existing_output(dij, stf)
        else:
            self._prepare_new_calculation(dij, ct, stf)

        if self._use_output is not None:
            self.fred_order = self._generate_fred_order(dij, stf)
        if self.dose_cube is None:
            dij["physical_dose"] = coo_matrix(
                (dij["dose_grid"].num_voxels, stf.total_number_of_bixels), dtype=np.int8
            )

        else:
            # Fill dij
            dij = self._process_dose_cube(dij)

        self._check_saving_options()

        logger.info("Finalizing dose calculation...")
        t_start = time.time()
        dij = self._finalize_dose(dij)
        t_end = time.time()
        logger.info("Done in %f seconds.", t_end - t_start)

        return dij

    def _process_existing_output(self, dij: dict, stf: SteeringInformation) -> None:
        """
        Process existing output files if `self._use_output` is set.

        Parameters
        ----------
        dij : dict
            The dose influence matrix.
        stf : SteeringInformation
            The steering information for the beams.
        """
        self._output_dir = self._use_output
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
        self._total_number_of_bixels = stf.total_number_of_bixels
        self._simulated_particles_per_bixel = (
            np.floor(max([1, self.num_histories_direct / self._total_number_of_bixels]))
            if self._calc_dose_direct
            else self.num_histories_per_beamlet
        )

        beams_fred = self._generate_beams_fred(dij, stf)
        self._write_plan_file(beams_fred)
        self._set_up_files(ct)

        if self.external_calculation:
            self.dose_cube = None
            logger.info((f"Files created for external calculation in dir: {self._inp_dir}"))
        else:
            self._call_fred()
            if not self._calc_dose_direct:
                self.fred_order = self._generate_fred_order(dij, stf)
            self._read_output_files()

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
                "BAMS_to_iso": self._machine.bams_to_iso_dist / 10,
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

    def _generate_layers_fred(self, beam, beam_index) -> list[dict]:
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

        for key, value in beam.energy_layers.items():
            pb_fred = []
            for i in range(len(value["beamlet_idx"])):
                target_point = beam.rays[value["rays_idx"][i]].target_point_bev / 10
                source_point = beam.source_point_bev / 10
                ray_position = beam.rays[value["rays_idx"][i]].ray_pos_bev / 10
                distance = target_point[1] - ray_position[1]

                pb_dict = {
                    "ID": i,
                    "fieldID": beam_index,
                    "particle": beam.radiation_mode,
                    "T": value["full_energy"],
                    "position_bev": [str(ray_position[2]), str(ray_position[0]), str(0)],
                    "divergence": [
                        str((target_point[2] - source_point[2]) / distance),
                        str((target_point[0] - target_point[0]) / distance),
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

            emittance = self._machine.foci[value["full_energy"]][0].emittance
            layer_dict = {
                "energy": value["full_energy"],
                "ESpread": self._machine.spectra[value["full_energy"]].fwhm / 2.355,
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
            # Direct dose calculation
            dij["physical_dose"] = np.expand_dims(self.dose_cube.flatten(), axis=1)

            if self.calc_let and self.let_cube is not None:
                dij["mLETd"].flat[0] = coo_matrix(
                    (
                        self.let_cube[self._vdose_grid] / 10,
                        (self._vdose_grid, np.ones(len(self._vdose_grid))),
                    ),
                    shape=(self.dose_grid.num_voxels, 1),
                )

                # LETd * dose
                dij["mLETDose"].flat[0] = coo_matrix(
                    (
                        (self.let_cube[self._vdose_grid] / 10) * self.dose_cube[self._vdose_grid],
                        (self._vdose_grid, np.ones(len(self._vdose_grid))),
                    ),
                    shape=(self.dose_grid.num_voxels, 1),
                )
            dij_fields_to_override = [
                "num_of_beams",
                "beam_num",
                "bixel_num",
                "ray_num",
                "total_number_of_bixels",
                "total_number_of_rays",
                "num_of_rays_per_beam",
            ]
            for field_name in dij_fields_to_override:
                dij[field_name] = 1
        else:
            # Scorer-ij mode
            self.dose_cube.col = self.fred_order[self.dose_cube.col]

            self.dose_cube = self.dose_cube.tocsr()

            self.dose_cube = self.dose_cube[:, self.fred_order]

            self.dose_cube = self.dose_cube.tocoo()

            dij["physical_dose"].flat[0] = self._conversion_factor * self.dose_cube

            if self.calc_let and self.let_cube is not None:
                self.let_cube.col = self.fred_order[self.let_cube.col]

                self.let_cube = self.let_cube.tocsr()

                self.let_cube = self.let_cube[:, self.fred_order]

                self.let_cube = self.let_cube.tocoo()

                # Divide by 10, FRED scores in MeV * cm^2 / g
                dij["mLETd"].flat[0] = self.let_cube / 10

                # LETd * dose
                dij["mLETDose"].flat[0] = coo_matrix(
                    dij["physical_dose"].flat[0].multiply(dij["mLETd"].flat[0])
                )

        if self.calc_bio_dose:
            logger.warning("Biological dose calculation is not implemented yet.")
        return dij

    def _check_saving_options(self) -> None:
        if self._save_input is None and self._save_output is None:
            return
        if self._save_input is not None:
            if not os.path.isdir(self._save_input):
                raise ValueError(
                    f"The provided path '{self._save_input}' for 'save_input' is not a valid directory."
                )
            else:
                shutil.copytree(
                    self._fred_root_folder / "inp", self._save_input / "inp", dirs_exist_ok=True
                )
        if self._save_output is not None:
            if not os.path.isdir(self._save_output):
                raise ValueError(
                    f"The provided path '{self._save_output}' for 'save_output' is not a valid directory."
                )
            else:
                shutil.copytree(
                    self._fred_root_folder / "out", self._save_output / "out", dirs_exist_ok=True
                )

        # TODO: Might implement a method to keep tmp files in the future:
        shutil.rmtree(self._fred_root_folder)
        logger.info((f"Temp tree directory deleted: {self._fred_root_folder}"))

    def _set_up_files(self, ct: CT) -> None:
        """
        Set up the necessary input files for the FRED engine.

        Parameters
        ----------
        ct : CT
            The CT object containing the patient data.
        """
        self._set_up_hu_table()
        self._set_up_patient_ct(ct)
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

            if self.calc_let:
                letd_dij_file = "Phantom.LETd.bin"
                letd_file_name = os.path.join(dose_dij_folder, letd_dij_file)

                try:
                    let_cube = self.read_sparse_dij_bin(letd_file_name)
                except Exception as e:
                    logger.error(f"Unable to load file: {letd_file_name}. Error: {e}")

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

            if self.calc_let:
                letd_dij_folder = dose_cube_folder
                letd_cube_file_name = "Phantom.LETd.mhd"

                try:
                    let_cube = sitk.GetArrayFromImage(
                        sitk.ReadImage(os.path.join(letd_dij_folder, letd_cube_file_name))
                    )
                except Exception as e:
                    logger.error(
                        f"Unable to load file: {os.path.join(letd_dij_folder, letd_cube_file_name)}. Error: {e}"
                    )

        # Store the results for further processing

        self.dose_cube = dose_cube
        self.let_cube = let_cube

    def read_sparse_dij_bin(self, f_name: str) -> coo_matrix:
        """
        Dispatch method to read a sparse dij binary file based on the dij format version.

        Parameters
        ----------
            f_name (str): File name to read.

        Returns
        -------
            coo_matrix: Sparse matrix containing the dij data.
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
            dij[q_name] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

            # Loop over all scenarios and preallocate quantity containers
            # TODO: write test for this
            for i in range(self.mult_scen.scen_mask.size):
                # Only if there is a scenario we will allocate
                if self.mult_scen.scen_mask.flat[i]:
                    if self._calc_dose_direct:
                        dij[q_name].flat[i] = np.zeros(
                            (self.dose_grid.num_voxels, dij["num_of_beams"]), dtype=np.float32
                        )
                    else:
                        # This could probably be optimized by using direct access to the
                        # lil_matrix's data structures
                        # TODO: we could store a single sparse pattern matrix and then only store
                        # the values for all quantities for better memory management
                        dij[q_name].flat[i] = lil_matrix(
                            (self._num_of_columns_dij, self.dose_grid.num_voxels),
                            dtype=np.float32,
                        )

            self._computed_quantities.append(q_name)

        return dij

    def _call_fred(self) -> None:
        """
        Call the FRED executable to perform the dose calculation.

        Raises
        ------
        SystemExit
            If the FRED execution fails.
        """
        output_path = os.path.abspath(os.path.join(self._fred_root_folder, self._output_folder))
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
            subprocess.run(
                execute_cmd,
                cwd=self._inp_dir,
                env=os.environ.copy(),
                shell=True,
                check=True,
                capture_output=not self.print_output,
            )
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode() if e.stderr else "No error details available"
            logger.error(
                "FRED execution failed with return code %s. Reason: %s",
                e.returncode,
                stderr_msg,
            )
            raise SystemExit("Aborting...")
        else:
            t_end = time.time()
            logger.info("Done in %f seconds.", t_end - t_start)

    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> None:
        dij = super()._init_dose_calc(ct, cst, stf)
        dij = self._allocate_quantity_matrices(dij, ["physical_dose"])

        if dij["num_of_scenarios"] > 1:
            raise NotImplementedError(
                "Multiple scenarios are not supported for FRED calculations."
            )

        # TODO: Add biomodel support
        # if hasattr(self, 'bioModel') and isinstance(self.bioModel, matRad_LQLETbasedModel):
        #     self._calc_bio_dose = True
        # else:
        #     self._calc_bio_dose = False

        # # Limit RBE calculation to proton models for the time being
        # if self._calc_bio_dose:
        #     if self.radiation_mode == "protons":
        #         dij = self.load_biological_data(cst, dij)
        #         dij = self._allocate_quantity_matrices(dij, ["mAlphaDose", "mSqrtBetaDose"])
        #         # Only considering LET-based models
        #         self.calc_let = True
        #     else:
        #         logger.warning(
        #             f"Biological dose calculation not supported for radiation modality: {self.radiation_mode}"
        #         )
        #         self._calc_bio_dose = False

        # TODO: Handle constant RBE models
        # if isinstance(self.bioModel, matRad_ConstantRBE):
        #     dij["RBE"] = self.bioModel.RBE

        # If LET calculation is enabled
        if self.calc_let:
            self.scorers.extend(["LETd"])
            # Allocate containers for LET*Dose and dose-weighted LET
            dij = self._allocate_quantity_matrices(dij, ["mLETDose", "mLETd"])

        return dij

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
        if not self.external_calculation:
            # Loop over all scenarios and remove dose influence for voxels outside of segmentations
            for i in range(self.mult_scen.scen_mask.size):
                # Only if there is a scenario we will allocate
                if self.mult_scen.scen_mask.flat[i]:
                    # Loop over all used quantities
                    for q_name in self._computed_quantities:
                        if not self._calc_dose_direct:
                            tmp_matrix = cast(lil_matrix, dij[q_name].flat[i])
                            tmp_matrix = tmp_matrix.tocsr()
                            tmp_matrix.eliminate_zeros()

                            if self._dij_format_version in {"21", "31"}:
                                shape = dij["dose_grid"].dimensions
                                dij[q_name].flat[i] = swap_orientation_sparse_matrix(
                                    tmp_matrix, shape, (0, 1)
                                )
                            else:
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


def read_sparse_dij_bin_v20(f_name: str) -> coo_matrix:
    """
    Read a sparse dij binary file in a Matlab-like format.

    Parameters
    ----------
        fName (str): File name to read.

    Returns
    -------
        coo_matrix: Sparse matrix containing the dij data.
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
        all_col_inds_list = []
        beamlet_counter = 0

        for _ in range(number_of_beamlets):
            # Read beamlet header
            _ = np.frombuffer(f.read(4), dtype=np.int32)[0]  # bix_num (unused)
            num_vox = np.frombuffer(f.read(4), dtype=np.int32)[0]
            beamlet_counter += 1

            # Set column indices vector for this beamlet
            all_col_inds_list.append(np.full(num_vox, beamlet_counter - 1, dtype=np.int32))

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
        all_col_inds = np.concatenate(all_col_inds_list)

    total_voxels = int(np.prod(dims))
    dij_matrix = coo_matrix(
        (all_values, (all_voxel_inds, all_col_inds)), shape=(total_voxels, beamlet_counter)
    )
    # dij_matrix = dij_matrix.tocsc()
    return dij_matrix


def read_sparse_dij_bin_v21(f_name: str) -> coo_matrix:
    """
    Read a sparse dij binary file in version 2.1 format.

    Parameters
    ----------
        f_name (str): File name to read.

    Returns
    -------
        coo_matrix: Sparse matrix containing the dij data.
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
        all_col_inds_list = []
        beamlet_counter = 0

        for _ in range(number_of_beamlets):
            # Read beamlet header
            _ = np.frombuffer(f.read(4), dtype=np.uint32)[0]  # bix_num
            num_vox = np.frombuffer(f.read(4), dtype=np.int32)[0]

            beamlet_counter += 1

            # Set column indices vector for this beamlet
            all_col_inds_list.append(np.full(num_vox, beamlet_counter - 1, dtype=np.int32))

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

            # Permute x and y components in voxel indices
            ind_y, ind_x, ind_z = np.unravel_index(curr_voxel_indices, dims)
            permuted_voxel_indices = np.ravel_multi_index(
                (ind_x, ind_y, ind_z), [dims[1], dims[0], dims[2]]
            )

            all_voxel_inds_list.append(permuted_voxel_indices)
            all_values_list.append(values)

    # Concatenate the lists into single NumPy arrays
    all_values = np.concatenate(all_values_list)
    all_voxel_inds = np.concatenate(all_voxel_inds_list)
    all_col_inds = np.concatenate(all_col_inds_list)

    total_voxels = int(np.prod(dims))
    dij_matrix = coo_matrix(
        (all_values, (all_voxel_inds, all_col_inds)), shape=(total_voxels, beamlet_counter)
    )
    return dij_matrix


def read_sparse_dij_bin_v31(f_name: str) -> coo_matrix:
    """
    Read a sparse dij binary file in version 3.1 format.

    Parameters
    ----------
        f_name (str): File name to read.

    Returns
    -------
        coo_matrix: Sparse matrix containing the dij data.
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

        # Permute x and y components in voxel indices
    ind_y, ind_x, ind_z = np.unravel_index(voxel_indices[0], dims)
    permuted_voxel_indices = np.ravel_multi_index(
        (ind_x, ind_y, ind_z), [dims[1], dims[0], dims[2]]
    )

    # Create sparse matrix
    total_voxels = int(np.prod(dims))
    dij_matrix = coo_matrix(
        (values, (permuted_voxel_indices, pb_idxs[0])),
        shape=(total_voxels, number_of_beamlets),
    )

    return dij_matrix
