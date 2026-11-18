/**
 * Static catalogs of the mock backend for package A: plan options per radiation mode,
 * configuration forms (engines, biological models, scenario models, problems, solvers),
 * the tissue schema and the objectives catalog. Modelled on the pyRadPlan registries
 * (`dose/engines`, `scenarios`, `optimization/problems|solvers|objectives`,
 * `biology/_editing.py`).
 */
import type { Choice, FieldDescriptor, FormSchema, TissueDraft } from "../protocol";
import type { BeamPreset, BioModelChoice, ObjectiveType, SolverChoice, TissuePick, TissueSchema } from "../protocol/plan";

type Extra = Omit<FieldDescriptor, "name" | "label" | "type">;

const fd = (name: string, label: string, type: FieldDescriptor["type"], extra: Extra = {}): FieldDescriptor => ({
  name,
  label,
  type,
  ...extra,
});

const choices = (values: Array<string | number | boolean | null>, labels?: string[]): Choice[] =>
  values.map((value, i) => ({ value, label: labels?.[i] ?? String(value) }));

export const RADIATION_MODES = ["photons", "protons", "helium", "carbon", "oxygen", "VHEE"];

export const MACHINES: Record<string, string[]> = {
  photons: ["Generic"],
  protons: ["Generic", "generic_MCsquare"],
  helium: ["Generic"],
  carbon: ["Generic", "HIT_APM"],
  oxygen: ["Generic"],
  VHEE: ["Generic", "Focused"],
};

export const MC_REASON =
  "Monte Carlo engines run an external executable (subprocess), which the browser runtime cannot start";

interface EngineDef {
  id: string;
  label: string;
  modes: string[];
  mc?: boolean;
  fields: FieldDescriptor[];
}

const particleEngineFields = (): FieldDescriptor[] => [
  fd("calc_let", "Calc let", "choice", {
    default: "auto",
    choices: choices(["auto", true, false], ["auto", "true", "false"]),
    description: "Also compute the LET influence matrix; auto follows the objectives and the biological model.",
  }),
  fd("calc_bio_dose", "Calc bio dose", "choice", {
    default: "auto",
    choices: choices(["auto", true, false], ["auto", "true", "false"]),
    description: "Compute the alpha/beta influence matrices for biological optimization.",
  }),
  fd("air_offset_correction", "Air offset correction", "bool", { default: true }),
  fd("lateral_model", "Lateral model", "choice", {
    default: "fastest",
    choices: choices(["auto", "single", "double", "multi", "fastest", "singleXY"]),
  }),
  fd("cut_off_method", "Cut off method", "choice", { default: "integral", choices: choices(["integral", "relative"]) }),
  fd("geometric_lateral_cutoff", "Geometric lateral cutoff", "float", {
    default: 50,
    min: 0,
    unit: "mm",
    description: "Lateral distance from the central ray beyond which no dose is computed.",
  }),
  fd("dosimetric_lateral_cutoff", "Dosimetric lateral cutoff", "float", {
    default: 0.995,
    exclusive_min: 0,
    max: 1,
    description: "Fraction of the integral dose kept laterally.",
  }),
  fd("ssd_density_threshold", "Ssd density threshold", "float", { default: 0.05, min: 0, max: 1 }),
  fd("trace_on_dose_grid", "Trace on dose grid", "bool", { default: true }),
  fd("assumed_sparsity", "Assumed sparsity", "float", { default: 5e-4, exclusive_min: 0, max: 1 }),
  fd("tissue_classes", "Tissue classes", "json", {
    default: [],
    description: "Tissue properties used in the structure set (list of models; edited as JSON).",
  }),
];

export const ENGINES: EngineDef[] = [
  {
    id: "SVDPB",
    label: "SVD Pencil Beam",
    modes: ["photons"],
    fields: [
      fd("use_custom_primary_photon_fluence", "Use custom primary photon fluence", "bool", { default: false }),
      fd("kernel_cutoff", "Kernel cutoff", "float", {
        default: 400,
        exclusive_min: 0,
        unit: "mm",
        description: "Lateral kernel cutoff [mm]",
      }),
      fd("random_seed", "Random seed", "int", { default: 0, min: 0 }),
      fd("int_conv_resolution", "Int conv resolution", "float", { default: 0.5, exclusive_min: 0, unit: "mm" }),
      fd("force_penumbra", "Force penumbra", "float", {
        default: null,
        nullable: true,
        exclusive_min: 0,
        unit: "mm",
        description: "Override the machine's penumbra (None keeps it).",
      }),
      fd("force_uniform_fluence", "Force uniform fluence", "bool", { default: false }),
      fd("enable_dij_sampling", "Enable dij sampling", "bool", { default: true }),
      fd("dij_sampling", "Dij sampling", "group", {
        description: "Random sampling of low-dose dij entries.",
        fields: [
          fd("dij_sampling.rel_dose_threshold", "Rel dose threshold", "float", { default: 0.01, min: 0, max: 1 }),
          fd("dij_sampling.lat_cut_off", "Lat cut off", "float", { default: 20, min: 0, unit: "mm" }),
          fd("dij_sampling.type", "Type", "choice", { default: "radius", choices: choices(["radius", "dose"]) }),
        ],
      }),
      fd("geometric_lateral_cutoff", "Geometric lateral cutoff", "float", { default: 50, min: 0, unit: "mm" }),
      fd("keep_rad_depth_cubes", "Keep rad depth cubes", "bool", { default: false }),
      fd("hlut", "Hlut", "json", {
        default: null,
        nullable: true,
        description: "HU → relative electron density table, [[HU, rho], ...]; null uses the default.",
      }),
    ],
  },
  {
    id: "TOPAS",
    label: "TOPAS",
    modes: ["photons", "protons", "helium", "carbon"],
    mc: true,
    fields: [
      fd("num_histories_per_beamlet", "Num histories per beamlet", "int", { default: 20000, min: 1 }),
      fd("working_dir", "Working dir", "str", { default: "", editable: false, reason: MC_REASON }),
    ],
  },
  { id: "HongPB", label: "Hong Particle Pencil-Beam", modes: ["protons", "helium", "carbon", "oxygen"], fields: particleEngineFields() },
  {
    id: "FRED",
    label: "FRED",
    modes: ["protons"],
    mc: true,
    fields: [fd("num_histories_per_beamlet", "Num histories per beamlet", "int", { default: 1e5, min: 1 })],
  },
  {
    id: "VHEEPB",
    label: "VHEE Pencil Beam",
    modes: ["VHEE"],
    fields: [fd("geometric_lateral_cutoff", "Geometric lateral cutoff", "float", { default: 50, min: 0, unit: "mm" })],
  },
];

export function enginesFor(mode: string): EngineDef[] {
  return ENGINES.filter((e) => e.modes.includes(mode));
}

export function defaultEngine(mode: string): string | null {
  return enginesFor(mode).find((e) => !e.mc)?.id ?? null;
}

interface BioOption {
  id: string;
  model: string;
  label: string;
  modes: string[];
  /** Machines offering the base data; undefined = every machine. */
  machines?: string[];
  parameters: Record<string, unknown>;
  description: string;
  fields: FieldDescriptor[];
  /** Tissue model the structures need (null: no tissue parameters needed). */
  tissue: "LEM" | "MKM" | "mMKM" | "Generic" | null;
}

export const BIO_OPTIONS: BioOption[] = [
  {
    id: "none",
    model: "none",
    label: "None (physical dose)",
    modes: RADIATION_MODES,
    parameters: {},
    description: "No biological model: the physical dose is optimized.",
    fields: [],
    tissue: null,
  },
  {
    id: "constRBE",
    model: "constRBE",
    label: "Constant RBE · 1.1",
    modes: ["protons"],
    parameters: { rbe: 1.1 },
    description: "Constant RBE of 1.1 (clinical proton standard).",
    fields: [
      fd("rbe", "RBE", "float", { default: 1.1, exclusive_min: 0, step: 0.05, description: "Constant relative biological effectiveness." }),
    ],
    tissue: null,
  },
  {
    id: "MCN",
    model: "MCN",
    label: "McNamara · variable RBE",
    modes: ["protons"],
    parameters: {},
    description: "Phenomenological LET-based variable RBE model (McNamara 2015).",
    fields: [
      fd("p0", "p0", "float", { default: 0.999064, description: "Fit parameter p0." }),
      fd("p1", "p1", "float", { default: 0.35605, unit: "Gy (keV/µm)⁻¹" }),
      fd("p2", "p2", "float", { default: 1.1012 }),
      fd("p3", "p3", "float", { default: -0.0038703, unit: "(keV/µm)⁻¹" }),
    ],
    tissue: "Generic",
  },
  {
    id: "LEM:HIT",
    model: "LEM",
    label: "LEM · LEM IV (HIT)",
    modes: ["helium", "carbon", "oxygen"],
    machines: ["Generic", "HIT_APM"],
    parameters: { data_model: "LEM IV", kernel_set: "HIT" },
    description: "Local effect model IV with the HIT base data (alpha/beta tables per tissue class).",
    fields: [
      fd("data_model", "Data model", "choice", { default: "LEM IV", choices: choices(["LEM I", "LEM IV"]) }),
      fd("interpolation", "Interpolation", "choice", { default: "linear", choices: choices(["linear", "nearest"]) }),
    ],
    tissue: "LEM",
  },
  {
    id: "mMKM:NIRS",
    model: "mMKM",
    label: "mMKM · NIRS",
    modes: ["carbon", "helium"],
    machines: ["HIT_APM"],
    parameters: { data_model: "mMKM", kernel_set: "NIRS" },
    description: "Modified microdosimetric kinetic model (NIRS clinical base data).",
    fields: [fd("data_model", "Data model", "choice", { default: "mMKM", choices: choices(["MKM", "mMKM"]) })],
    tissue: "mMKM",
  },
  {
    id: "WED",
    model: "WED",
    label: "Wedenberg · variable RBE",
    modes: ["protons"],
    machines: ["generic_MCsquare"],
    parameters: {},
    description: "Wedenberg 2013 LET-based RBE model (needs LET kernels in the base data).",
    fields: [fd("q", "q", "float", { default: 0.434, unit: "Gy µm/keV" })],
    tissue: "Generic",
  },
];

export const DEFAULT_BIO: Record<string, string> = {
  photons: "none",
  protons: "constRBE",
  helium: "LEM:HIT",
  carbon: "LEM:HIT",
  oxygen: "LEM:HIT",
  VHEE: "none",
};

export function bioOptionsFor(mode: string, machine: string, engine: string | null): BioModelChoice[] {
  const mc = ENGINES.find((e) => e.id === engine)?.mc ?? false;
  return BIO_OPTIONS.filter((o) => o.modes.includes(mode)).map((o) => {
    let reason: string | undefined;
    if (o.machines && !o.machines.includes(machine)) {
      reason = `Machine '${machine}' offers no ${o.model} base data (tissue classes) for this model.`;
    } else if (mc && o.tissue === "LEM") {
      reason = `The dose engine ${engine} does not use the machine's pencil-beam kernels and cannot compute ${o.model}.`;
    }
    return {
      value: o.id,
      label: o.label,
      identifier: o.model,
      model: o.model,
      parameters: o.parameters,
      description: o.description,
      ...(reason ? { enabled: false, reason } : {}),
    };
  });
}

export const SCENARIO_MODELS: Array<{ id: string; label: string; fields: FieldDescriptor[] }> = [
  { id: "nomScen", label: "Nominal Scenario", fields: [ctPhaseGroup()] },
  {
    id: "wcScen",
    label: "Worst-case Grid",
    fields: [
      setupGroup(),
      rangeGroup(),
      ctPhaseGroup(),
      fd("wc_sigma", "Wc sigma", "float", { default: 1, exclusive_min: 0, description: "Extent of the worst-case grid in standard deviations." }),
      fd("num_setup_grid_points", "Num setup grid points", "int", { default: 3, min: 1, max: 3, description: "1 or 3 grid points per setup axis." }),
      fd("num_range_grid_points", "Num range grid points", "int", { default: 3, min: 1, max: 3 }),
      fd("combinations", "Combinations", "choice", {
        default: "shift",
        choices: choices(["none", "shift", "all"]),
        description: "How setup and range scenarios are combined.",
      }),
    ],
  },
  {
    id: "impScen",
    label: "Importance Grid",
    fields: [
      setupGroup(),
      rangeGroup(),
      ctPhaseGroup(),
      fd("num_setup_grid_points", "Num setup grid points", "int", { default: 9, min: 1 }),
      fd("num_range_grid_points", "Num range grid points", "int", { default: 9, min: 1 }),
      fd("max_abs_setup_shift", "Max abs setup shift", "float", { default: 6, min: 0, unit: "mm" }),
    ],
  },
  {
    id: "rndScen",
    label: "Random Samples",
    fields: [
      setupGroup(),
      rangeGroup(),
      ctPhaseGroup(),
      fd("num_samples", "Num samples", "int", { default: 10, min: 1 }),
      fd("random_seed", "Random seed", "int", { default: null, nullable: true, min: 0 }),
      fd("distribution", "Distribution", "group", {
        fields: [
          fd("distribution.distribution", "Distribution", "choice", {
            default: "independent_gaussian",
            choices: choices(["independent_gaussian", "multivariate_gaussian"]),
          }),
          fd("distribution.correlation", "Correlation", "json", { default: null, nullable: true }),
        ],
      }),
    ],
  },
];

function setupGroup(): FieldDescriptor {
  return fd("setup", "Setup", "group", {
    description: "Setup uncertainty (patient positioning).",
    fields: [
      fd("setup.sd", "Sd", "number_list", { default: [2.25, 2.25, 2.25], item_type: "float", length: 3, unit: "mm", description: "Systematic setup error SD per axis (x y z)." }),
      fd("setup.shift_mean", "Shift mean", "number_list", { default: [0, 0, 0], item_type: "float", length: 3, unit: "mm" }),
    ],
  });
}

function rangeGroup(): FieldDescriptor {
  return fd("range", "Range", "group", {
    description: "Range uncertainty (absolute and relative).",
    fields: [
      fd("range.abs_sd", "Abs sd", "float", { default: 1, min: 0, unit: "mm" }),
      fd("range.rel_sd", "Rel sd", "float", { default: 3.5, min: 0, unit: "%" }),
      fd("range.combined", "Combined", "bool", { default: false, description: "Combine absolute and relative range errors into one dimension." }),
    ],
  });
}

function ctPhaseGroup(): FieldDescriptor {
  return fd("ct_phase", "Ct phase", "group", {
    description: "4D CT phases taken into account.",
    fields: [
      fd("ct_phase.phases", "Phases", "number_list", { default: [0], item_type: "int", description: "Indices of the CT phases." }),
      fd("ct_phase.probabilities", "Probabilities", "number_list", { default: null, nullable: true, item_type: "float" }),
    ],
  });
}

export const PROBLEMS: Array<{ id: string; label: string; fields: FieldDescriptor[] }> = [
  {
    id: "nonlin_fluence",
    label: "Fluence (nonlinear)",
    fields: [
      fd("apply_overlap", "Apply overlap", "bool", { default: true, description: "Assign overlapping voxels to the structure with the highest overlap priority." }),
      fd("evaluate_all_scenarios", "Evaluate all scenarios", "bool", { default: false }),
      fd("bypass_objective_jacobian", "Bypass objective jacobian", "bool", { default: true }),
      fd("max_approximation", "Max approximation", "choice", { default: "logsumexp", choices: choices(["logsumexp", "pnorm", "none"]) }),
      fd("p_norm", "P norm", "float", { default: 2, min: 1 }),
    ],
  },
  {
    id: "simple_lsq",
    label: "Simple least squares",
    fields: [fd("apply_overlap", "Apply overlap", "bool", { default: true })],
  },
];

export const IPOPT_REASON = "ipyopt (IPOPT) is not available in the browser runtime (no Pyodide build)";

export const SOLVERS: Array<SolverChoice & { fields: FieldDescriptor[] }> = [
  {
    value: "ipax",
    label: "Array-API Interior Point (ipax)",
    identifier: "ipax",
    gpu_compatible: true,
    description: "Runs on the GPU backend.",
    fields: [
      fd("max_iter", "Max iter", "int", { default: 500, min: 1 }),
      fd("abs_obj_tol", "Abs obj tol", "float", { default: 1e-6, exclusive_min: 0 }),
      fd("max_time", "Max time", "float", { default: 3600, min: 0, unit: "s" }),
      fd("display", "Display", "bool", { default: false }),
      fd("options", "Options", "group", {
        description: "ipax.Options",
        fields: [
          fd("options.tol", "Tol", "float", { default: 1e-8, exclusive_min: 0 }),
          fd("options.mu_init", "Mu init", "float", { default: 0.1, exclusive_min: 0 }),
          fd("options.mu_strategy", "Mu strategy", "choice", { default: "adaptive", choices: choices(["monotone", "adaptive"]) }),
          fd("options.linear_solver", "Linear solver", "group", {
            fields: [
              fd("options.linear_solver.kind", "Kind", "choice", { default: "cg", choices: choices(["cg", "cholesky", "lsqr"]), editable_choice: true }),
              fd("options.linear_solver.max_iter", "Max iter", "int", { default: 200, min: 1 }),
            ],
          }),
        ],
      }),
    ],
  },
  {
    value: "scipy",
    label: "SciPy minimize",
    identifier: "scipy",
    gpu_compatible: false,
    fields: [
      fd("max_iter", "Max iter", "int", { default: 500, min: 1, description: "Maximum number of iterations." }),
      fd("abs_obj_tol", "Abs obj tol", "float", { default: 1e-6, exclusive_min: 0 }),
      fd("max_time", "Max time", "float", { default: 3600, min: 0, unit: "s" }),
      fd("method", "Method", "choice", {
        default: "L-BFGS-B",
        choices: choices(["L-BFGS-B", "SLSQP", "trust-constr", "TNC"]),
        editable_choice: true,
        description: "scipy.optimize.minimize method; any other method name may be typed.",
      }),
      fd("display", "Display", "bool", { default: false }),
      fd("options", "Options", "json", { default: {}, description: "Extra options passed to scipy.optimize.minimize." }),
    ],
  },
  {
    value: "ipopt",
    label: "Interior Point Optimizer",
    identifier: "ipopt",
    gpu_compatible: false,
    enabled: false,
    reason: IPOPT_REASON,
    fields: [],
  },
];

export const DEFAULT_SOLVER = "scipy";

export const BEAM_PRESETS: Array<BeamPreset & { modes: string[] }> = [
  { name: "single", label: "", gantry_angles: [0], couch_angles: [0], description: "One beam from anterior.", typical: false, modes: [] },
  { name: "parallel opposed", label: "", gantry_angles: [90, 270], couch_angles: [0, 0], description: "Two lateral opposed beams.", typical: false, modes: ["protons", "helium", "carbon", "oxygen"] },
  { name: "3-field", label: "", gantry_angles: [0, 120, 240], couch_angles: [0, 0, 0], description: "Three equidistant coplanar beams.", typical: false, modes: ["protons"] },
  { name: "5-field IMRT", label: "", gantry_angles: [0, 72, 144, 216, 288], couch_angles: [0, 0, 0, 0, 0], description: "Five equidistant coplanar beams (TG-119 standard).", typical: false, modes: ["photons"] },
  { name: "7-field IMRT", label: "", gantry_angles: [0, 51.43, 102.86, 154.29, 205.71, 257.14, 308.57], couch_angles: [0, 0, 0, 0, 0, 0, 0], description: "Seven equidistant coplanar beams.", typical: false, modes: ["photons"] },
  { name: "9-field IMRT", label: "", gantry_angles: [0, 40, 80, 120, 160, 200, 240, 280, 320], couch_angles: [0, 0, 0, 0, 0, 0, 0, 0, 0], description: "Nine equidistant coplanar beams.", typical: false, modes: ["photons"] },
  { name: "non-coplanar 4", label: "", gantry_angles: [0, 90, 270, 0], couch_angles: [0, 0, 0, 90], description: "Three coplanar beams and a vertex beam.", typical: false, modes: [] },
];

export function beamPresetsFor(mode: string): BeamPreset[] {
  const fmt = (v: number[]) => v.map((x) => String(x)).join(" ");
  const presets = BEAM_PRESETS.map(({ modes, ...p }) => {
    let label = `${p.name} (${fmt(p.gantry_angles)}`;
    if (p.couch_angles.some(Boolean)) label += ` / couch ${fmt(p.couch_angles)}`;
    return { ...p, label: `${label})`, typical: modes.includes(mode) };
  });
  return [...presets.filter((p) => p.typical), ...presets.filter((p) => !p.typical)];
}

const NOT_IMPLEMENTED = "not yet implemented";

/** The fixed plan fields of `plan.options.fields`. */
export function planFields(lockedGrid: boolean): FieldDescriptor[] {
  return [
    fd("num_of_fractions", "Fractions", "int", { default: 30, min: 1, max: 1000, advanced: true }),
    fd("dose_convention", "Doses", "choice", {
      default: "total",
      advanced: true,
      choices: [
        { value: "per_fraction", label: "per fraction" },
        { value: "total", label: "total" },
      ],
      description:
        "Convention for dose values in objectives and results:\nper fraction - objective doses and reported doses refer to one fraction;\ntotal - objective values (including reference images) and reported results\nrefer to the whole course; doses scale with the number of fractions, LET does not.",
    }),
    fd("bixel_width", "Bixel width / spot spacing", "float", {
      default: 5,
      min: 0.1,
      max: 100,
      decimals: 2,
      unit: "mm",
      description: "Photons: width (and height) of quadratic photon bixel\nParticles: lateral spot distance",
    }),
    fd("dose_grid_resolution", "Dose grid res.", "number_list", {
      default: [5, 5, 5],
      item_type: "float",
      length: 3,
      min: 0.1,
      max: 50,
      decimals: 2,
      unit: "mm",
      ...(lockedGrid ? { editable: false, reason: "The dose grid is fixed by the web profile (locked by the owner)." } : {}),
    }),
    fd("gantry_angles", "Gantry angles", "number_list", { default: [0], item_type: "float", unit: "deg" }),
    fd("couch_angles", "Couch angles", "number_list", { default: [0], item_type: "float", unit: "deg" }),
    fd("iso_center", "Iso center", "number_list", { default: null, nullable: true, item_type: "float", length: 3, unit: "mm" }),
    fd("optimized_quantity", "Optimized quantity", "choice", {
      default: "auto",
      advanced: true,
      editable: false,
      reason: NOT_IMPLEMENTED,
      choices: [{ value: "auto", label: "auto" }],
    }),
    fd("run_sequencing", "Run sequencing", "bool", { default: false, advanced: true, editable: false, reason: NOT_IMPLEMENTED }),
    fd("sequencing_levels", "Levels", "int", { default: 7, min: 1, max: 100, advanced: true, editable: false, reason: NOT_IMPLEMENTED }),
    fd("run_dao", "Run DAO", "bool", { default: false, advanced: true, editable: false, reason: NOT_IMPLEMENTED }),
    fd("conformal_3d", "3D conformal", "bool", { default: false, advanced: true, editable: false, reason: NOT_IMPLEMENTED }),
  ];
}

// ---------------------------------------------------------------------------
// Tissues
// ---------------------------------------------------------------------------

const ref = (name: string, label: string, unit: string, description: string): FieldDescriptor =>
  fd(name, label, "float", { unit, description, nullable: true, min: 0 });

const ALPHA_X = ref("alpha_x", "α_x", "Gy⁻¹", "Linear LQ coefficient of the reference radiation.");
const BETA_X = ref("beta_x", "β_x", "Gy⁻²", "Quadratic LQ coefficient of the reference radiation.");
const MKM_FIELDS = (): FieldDescriptor[] => [
  ref("r_domain", "r_domain", "µm", "Radius of the sub-cellular domain."),
  ref("r_nucleus", "r_nucleus", "µm", "Radius of the cell nucleus."),
  ref("alpha_0", "α_0", "Gy⁻¹", "Linear LQ coefficient in the limit of zero LET."),
  ref("beta_0", "β_0", "Gy⁻²", "Quadratic LQ coefficient (LET independent)."),
  fd("reference_radiation", "Reference radiation", "choice", { default: null, nullable: true }),
  ALPHA_X,
  BETA_X,
];

export const REFERENCE_RADIATIONS: Choice[] = [
  { value: "xray_200kV", label: "200 kV X-rays (Kase 2006)", description: "200 kV X-rays: y_D = 2.6 keV/µm\nSource: Kase et al. 2006" },
  { value: "co60", label: "⁶⁰Co γ (Inaniwa 2010)", description: "⁶⁰Co γ: y_D = 0.3 keV/µm\nSource: Inaniwa et al. 2010" },
  { value: "mv6", label: "6 MV X-rays (estimated)", description: "6 MV X-rays: y_D = 0.25 keV/µm\nThe origin of this value has not been confirmed." },
];

export const TISSUE_MODELS: TissueSchema["models"] = [
  { label: "Generic", fields: [ALPHA_X, BETA_X] },
  {
    label: "LEM",
    fields: [
      fd("version", "Version", "choice", { default: "IV", choices: choices(["I", "II", "III", "IV"]), description: "LEM version." }),
      ref("d_t", "D_t", "Gy", "Threshold dose of the LEM dose-response curve."),
      ref("r_nucleus", "r_nucleus", "µm", "Radius of the cell nucleus."),
      ALPHA_X,
      BETA_X,
    ],
  },
  { label: "MKM", fields: MKM_FIELDS() },
  { label: "mMKM", fields: MKM_FIELDS() },
];

/** Base data tissue classes of the mock machines (LEM / HIT and mMKM / NIRS). */
export const TISSUE_CLASSES: Array<{ id: string; label: string; model: "LEM" | "mMKM"; values: Record<string, number | string | null>; kernel: string }> = [
  { id: "hit:default", label: "✓ HIT · default (α_x 0.1, β_x 0.05)", model: "LEM", kernel: "HIT", values: { version: "IV", d_t: 30, r_nucleus: 5, alpha_x: 0.1, beta_x: 0.05 } },
  { id: "hit:chordoma", label: "✓ HIT · chordoma (α_x 0.1, β_x 0.025)", model: "LEM", kernel: "HIT", values: { version: "IV", d_t: 30, r_nucleus: 5, alpha_x: 0.1, beta_x: 0.025 } },
  { id: "hit:prostate", label: "✓ HIT · prostate (α_x 0.0356, β_x 0.0237)", model: "LEM", kernel: "HIT", values: { version: "IV", d_t: 15, r_nucleus: 5, alpha_x: 0.0356, beta_x: 0.0237 } },
  { id: "nirs:hsg", label: "✓ NIRS · HSG", model: "mMKM", kernel: "NIRS", values: { r_domain: 0.32, r_nucleus: 3.9, alpha_0: 0.172, beta_0: 0.0615, reference_radiation: "xray_200kV" } },
];

export const CELL_LINE_PRESETS: Array<{ id: string; label: string; group: string; model: TissueDraft["model"]; values: Record<string, number | string | null> }> = [
  { id: "preset:hsg_mkm", label: "HSG (Kase 2006)", group: "MKM", model: "MKM", values: { r_domain: 0.32, r_nucleus: 3.9, alpha_0: 0.13, beta_0: 0.05, reference_radiation: "xray_200kV" } },
  { id: "preset:v79_mkm", label: "V79 (Hawkins 1996)", group: "MKM", model: "MKM", values: { r_domain: 0.42, r_nucleus: 4.1, alpha_0: 0.184, beta_0: 0.02, reference_radiation: "xray_200kV" } },
  { id: "preset:hsg_lem", label: "HSG (Elsässer 2010)", group: "LEM", model: "LEM", values: { version: "IV", d_t: 30, r_nucleus: 5, alpha_x: 0.313, beta_x: 0.0615 } },
  { id: "preset:late", label: "Late-responding tissue α/β = 2", group: "Generic", model: "Generic", values: { alpha_x: 0.1, beta_x: 0.05 } },
  { id: "preset:early", label: "Early-responding tissue α/β = 10", group: "Generic", model: "Generic", values: { alpha_x: 0.3, beta_x: 0.03 } },
];

export function tissuePicks(preferredModel: "LEM" | "mMKM" | null, machine: string): TissuePick[] {
  const classes = TISSUE_CLASSES.filter((c) => c.model === preferredModel && (machine === "HIT_APM" || c.kernel === "HIT"));
  const base: TissuePick[] = classes.map((c) => ({
    id: c.id,
    kind: "base_data",
    label: c.label,
    group: `Tissue classes of ${machine}`,
    exact: true,
    preferred: true,
    model: c.model,
    values: c.values,
    description: `Tissue class of the ${c.kernel} base data; matches the base data exactly.`,
  }));
  const presets: TissuePick[] = CELL_LINE_PRESETS.map((p) => ({
    id: p.id,
    kind: "preset",
    label: p.label,
    group: base.length ? `Other presets · ${p.group}` : `Presets · ${p.group}`,
    exact: false,
    preferred: !base.length || p.model === preferredModel,
    model: p.model,
    values: p.values,
    description:
      preferredModel && p.model !== preferredModel && p.model !== "Generic"
        ? `A ${p.model} preset; converted to ${preferredModel}, keeping the parameters both share.`
        : `${p.model} cell line preset.`,
  }));
  return [...base, ...presets];
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

const ROBUSTNESS: Choice[] = [
  { value: "none", label: "None", identifier: "none" },
  { value: "expected_value", label: "Expected value", identifier: "expected_value" },
  { value: "expected_quantity", label: "Expected quantity", identifier: "expected_quantity" },
  { value: "objective_worst_case", label: "Objective worst case", identifier: "objective_worst_case" },
  { value: "composite_worst_case", label: "Composite worst case", identifier: "composite_worst_case" },
];

const dose = (name: string, def: number, description: string): FieldDescriptor =>
  fd(name, name, "float", { default: def, min: 0, unit_from: "quantity", description });

const volume = (name: string, def: number, description: string): FieldDescriptor =>
  fd(name, name, "float", { default: def, min: 0, max: 100, unit: "%", description });

const fDiff = (): FieldDescriptor =>
  fd("f_diff", "f_diff", "choice", {
    default: "squared",
    choices: choices(["squared", "linear"]),
    description: "Penalty of the deviation: squared or linear.",
  });

export const OBJECTIVES: ObjectiveType[] = [
  { name: "Squared Deviation", scenario_aware: false, robustness: ROBUSTNESS, parameters: [dose("d_ref", 60, "Reference dose the structure should achieve.")] },
  { name: "Squared Overdosing", scenario_aware: false, robustness: ROBUSTNESS, parameters: [dose("d_max", 30, "Maximum dose above which dose is penalized.")] },
  { name: "Squared Underdosing", scenario_aware: false, robustness: ROBUSTNESS, parameters: [dose("d_min", 60, "Minimum dose below which dose is penalized.")] },
  {
    name: "Squared Mimicking",
    scenario_aware: false,
    robustness: ROBUSTNESS,
    parameters: [fd("d_ref", "d_ref", "image_ref", { description: "Reference dose image the dose should mimic." })],
  },
  {
    name: "Min DVH",
    scenario_aware: false,
    robustness: ROBUSTNESS,
    parameters: [dose("d", 30, "Dose point of the DVH constraint."), volume("v_min", 95, "Minimum relative volume receiving at least d.")],
  },
  {
    name: "Max DVH",
    scenario_aware: false,
    robustness: ROBUSTNESS,
    parameters: [dose("d", 30, "Dose point of the DVH constraint."), volume("v_max", 95, "Maximum relative volume receiving more than d.")],
  },
  {
    name: "Mean Dose",
    scenario_aware: false,
    robustness: ROBUSTNESS,
    parameters: [dose("d_ref", 0, "Reference mean dose to achieve."), fDiff()],
  },
  {
    name: "EUD",
    scenario_aware: false,
    robustness: ROBUSTNESS,
    parameters: [dose("eud_ref", 0, "Reference equivalent uniform dose (EUD)."), fd("k", "k", "float", { default: 1, description: "EUD exponent (volume-effect parameter)." }), fDiff()],
  },
  { name: "Dose Uniformity", scenario_aware: false, robustness: ROBUSTNESS, parameters: [] },
  {
    name: "Variance",
    scenario_aware: true,
    robustness: [{ value: "scenario_stack", label: "Over all scenarios", identifier: "scenario_stack" }],
    parameters: [dose("d_ref", 0, "Reference dose of the variance.")],
  },
];

export const QUANTITIES = [
  { identifier: "physical_dose", name: "dose", meta_label: "Dose", unit: "Gy" },
  { identifier: "rbe_x_dose", name: "RBExDose", meta_label: "RBE-weighted Dose", unit: "Gy(RBE)" },
  { identifier: "effect", name: "Effect", meta_label: "Effect", unit: "" },
  { identifier: "let", name: "LETd", meta_label: "LET", unit: "keV/µm" },
  { identifier: "let_dose", name: "LETxDose", meta_label: "LET x Dose", unit: "Gy keV/µm" },
  { identifier: "alpha_dose", name: "AlphaDose", meta_label: "Alpha x Dose", unit: "Gy²" },
  { identifier: "sqrt_beta_dose", name: "SqrtBetaDose", meta_label: "sqrt(Beta) x Dose", unit: "Gy²" },
];

export function formSchema(id: string, title: string, fields: FieldDescriptor[], values: Record<string, unknown>, notes?: string[]): FormSchema {
  return { id, title, fields, values, explicit: Object.keys(values), ...(notes?.length ? { notes } : {}) };
}
