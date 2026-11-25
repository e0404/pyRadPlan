/**
 * Mock handlers of package C: file export, base data, settings, licenses, log level and
 * history, snapshots, and the `capabilities` of `app.info`. Registered in `mockBackend.ts`.
 *
 * Note: `./mockBackend` imports this module and this module imports error helpers back from
 * it (only used inside handlers, never while the module is evaluated).
 */
import { ndarray } from "../codec";
import type {
  AppInfo,
  Capabilities,
  ExportFormat,
  FieldDescriptor,
  FormSchema,
  ImportFormat,
  LogLevel,
  SessionState,
} from "../protocol";
import type {
  BaseDataAddResult,
  BaseDataFolder,
  BaseDataKind,
  BaseDataList,
  BaseDataRow,
  LicenseDetail,
  LicenseRow,
  SettingsSection,
  SettingsSectionId,
} from "../protocol/io";
import { MethodError, precondition, type MockContext, type MockHandlers, type MockSession } from "./mockBackend";
import { addMockDoseResult } from "./viewer";
import { zip } from "./zip";

export type MockRuntime = "pyodide" | "native";

/** Runtime of the mock: the browser's (default) or the native one (`?runtime=native`). */
export function mockRuntime(): MockRuntime {
  try {
    return new URLSearchParams(globalThis.location?.search ?? "").get("runtime") === "native" ? "native" : "pyodide";
  } catch {
    return "pyodide";
  }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

export const BROWSER_REASONS = {
  image_file_io: "Image files need SimpleITK file IO, which the browser runtime does not provide.",
  dicom_import: "DICOM CT and dose import need SimpleITK file IO, which the browser runtime does not provide.",
  ai: "The AI features need pydantic_ai, which is not installed in the browser runtime.",
  ipopt: "IPOPT (ipyopt) is not available in the browser runtime.",
  gpu: "The browser runtime has no GPU array backend.",
  numba: "numba is not available in the browser runtime.",
  monte_carlo: "Monte Carlo engines run external programs, which a browser cannot start.",
  pause: "Pause and resume need a second thread; the browser runtime serves one request at a time.",
  open_folder: "A browser cannot open folders on your computer.",
  host_paths: "Folders of the host computer are not visible in the browser.",
  base_data_persistence: "Base data added in the browser lasts until the page is closed.",
  pickle_load:
    "Loading a pickle runs code from the file, which could read the data in this page; it is disabled in the browser. Use .mat or .npz instead",
} as const;

export function mockCapabilities(runtime: MockRuntime): Capabilities {
  const web = runtime === "pyodide";
  const features: Capabilities["features"] = {
    image_file_io: { available: !web, reason: web ? BROWSER_REASONS.image_file_io : null },
    dicom_import: { available: !web, reason: web ? BROWSER_REASONS.dicom_import : null },
    dicom_export: { available: true, reason: null },
    pickle_load: { available: !web, reason: web ? BROWSER_REASONS.pickle_load : null },
    ai: { available: !web, reason: web ? BROWSER_REASONS.ai : null },
    ipopt: { available: !web, reason: web ? BROWSER_REASONS.ipopt : null },
    gpu: { available: !web, reason: web ? BROWSER_REASONS.gpu : null },
    numba: { available: !web, reason: web ? BROWSER_REASONS.numba : null },
    monte_carlo: { available: !web, reason: web ? BROWSER_REASONS.monte_carlo : null },
    pause: { available: !web, reason: web ? BROWSER_REASONS.pause : null },
    open_folder: { available: !web, reason: web ? BROWSER_REASONS.open_folder : null },
    host_paths: { available: !web, reason: web ? BROWSER_REASONS.host_paths : null },
    base_data_persistence: { available: !web, reason: web ? BROWSER_REASONS.base_data_persistence : null },
  };
  const imageReason = web ? BROWSER_REASONS.image_file_io : null;
  const mat = "application/x-matlab-data";
  const bin = "application/octet-stream";
  const exp = (
    e: Partial<ExportFormat> & Pick<ExportFormat, "object" | "format" | "label" | "extension"> & { structure_format?: string },
  ): ExportFormat => ({
    mime: bin,
    available: true,
    reason: null,
    archive: false,
    ...e,
  });
  const labelMap = (format: string, label: string, extension: string): ExportFormat =>
    exp({
      object: "cst",
      format,
      label: `${label} label map`,
      extension,
      available: !web,
      reason: imageReason,
      archive: true,
      keeps_objectives: false,
    });
  const image = (format: string, label: string, extension: string): ExportFormat =>
    exp({ object: "result", format, label, extension, available: !web, reason: imageReason });
  const imp = (e: Omit<ImportFormat, "available" | "reason"> & { image?: boolean; pickle?: boolean }): ImportFormat => {
    const { image: isImage, pickle: isPickle, ...rest } = e;
    const blocked = web && (isImage || isPickle);
    return { ...rest, available: !blocked, reason: blocked ? (isImage ? imageReason : BROWSER_REASONS.pickle_load) : null };
  };
  const exports: ExportFormat[] = [
    exp({ object: "pln", format: "mat", label: "MATLAB (*.mat)", extension: ".mat", mime: mat }),
    exp({ object: "pln", format: "pickle", label: "Pickle (*.pkl)", extension: ".pkl" }),
    exp({ object: "dij", format: "mat", label: "MATLAB (*.mat)", extension: ".mat", mime: mat }),
    exp({ object: "dij", format: "pickle", label: "Pickle (*.pkl)", extension: ".pkl" }),
    exp({
      object: "cst",
      format: "mat",
      label: "MATLAB (*.mat)",
      extension: ".mat",
      mime: mat,
      keeps_objectives: true,
      note: "A structure set alone needs a CT when it is loaded again.",
    }),
    exp({ object: "cst", format: "pickle", label: "Pickle (*.pkl)", extension: ".pkl", keeps_objectives: true }),
    exp({ object: "cst", format: "hdf5", label: "HDF5 (*.h5)", extension: ".h5", keeps_objectives: true, note: "Includes the CT" }),
    exp({ object: "cst", format: "npz", label: "NumPy (*.npz)", extension: ".npz", keeps_objectives: false }),
    labelMap("nifti", "NIfTI", ".nii.gz"),
    labelMap("nrrd", "NRRD", ".nrrd"),
    {
      ...labelMap("nrrd", "NRRD", ".zip"),
      label: "NRRD Slicer segmentation, keeps overlaps (zip)",
      structure_format: "layers",
      note: "One layer per overlapping structure (cst.seg.nrrd)",
    } as ExportFormat,
    labelMap("meta", "MetaImage", ".mha"),
    exp({
      object: "cst",
      format: "dcm",
      label: "DICOM RTSTRUCT",
      extension: ".zip",
      mime: "application/zip",
      archive: true,
      keeps_objectives: false,
      structure_format: "rtstruct",
    }),
    exp({
      object: "cst",
      format: "dcm",
      label: "DICOM SEG",
      extension: ".zip",
      mime: "application/zip",
      archive: true,
      keeps_objectives: false,
      structure_format: "seg",
    }),
    exp({ object: "result", format: "npz", label: "NumPy (*.npz)", extension: ".npz" }),
    exp({ object: "result", format: "mat", label: "MATLAB resultGUI (*.mat)", extension: ".mat", mime: mat }),
    exp({
      object: "result",
      format: "dcm",
      label: "DICOM RTDOSE",
      extension: ".zip",
      mime: "application/zip",
      archive: true,
      note: "With the CT, one RTDOSE per physical or RBE-weighted dose and the RT (Ion) Plan of the active result",
    }),
    image("nifti", "NIfTI (*.nii.gz)", ".nii.gz"),
    image("nrrd", "NRRD (*.nrrd)", ".nrrd"),
    image("meta", "MetaImage (*.mha)", ".mha"),
    exp({
      object: "workspace",
      format: "hdf5",
      label: "pyRadPlan workspace, HDF5 (*.h5)",
      extension: ".h5",
      note: "Loads without running code from the file",
    }),
    exp({
      object: "workspace",
      format: "pickle",
      label: "pyRadPlan workspace, pickle (*.pkl)",
      extension: ".pkl",
      note: web ? "Not interchangeable with pickles of the native application." : undefined,
    }),
  ];
  const imports: ImportFormat[] = [
    imp({ format: "mat", label: "matRad MATLAB", extensions: [".mat"], objects: ["ct", "cst", "pln", "stf", "dij", "result"] }),
    imp({ format: "npz", label: "NumPy", extensions: [".npz"], objects: ["ct", "cst", "dose"] }),
    imp({
      format: "hdf5",
      label: "pyRadPlan workspace, HDF5 (*.h5, *.hdf5)",
      extensions: [".h5", ".hdf5"],
      objects: ["ct", "cst", "pln", "stf", "dij", "result", "dose"],
    }),
    imp({
      format: "pickle",
      label: "pyRadPlan workspace, pickle (*.pkl)",
      extensions: [".pkl", ".pickle"],
      objects: ["ct", "cst", "pln", "stf", "dij", "result", "dose"],
      pickle: true,
      note: "Unpickling runs code from the file: load only files you trust",
    }),
    imp({ format: "nifti", label: "NIfTI", extensions: [".nii", ".nii.gz"], objects: ["ct", "cst", "dose"], image: true }),
    imp({ format: "nrrd", label: "NRRD", extensions: [".nrrd"], objects: ["ct", "cst", "dose"], image: true }),
    imp({ format: "meta", label: "MetaImage", extensions: [".mha", ".mhd"], objects: ["ct", "cst", "dose"], image: true }),
  ];
  return { features, formats: { import: imports, export: exports } };
}

/**
 * The `actions` entries of package C for the objects present in a session. `has.result` is the
 * active result, which a snapshot copies; `anyResult` also counts snapshots and imported doses.
 */
export function ioActionFlags(has: SessionState["has"], hasQuantities: boolean, anyResult: boolean = has.result): Record<string, boolean> {
  return {
    load_dij: true,
    snapshot: has.result && hasQuantities,
    save_pln: has.pln,
    save_dij: has.dij,
    save_cst: has.cst,
    save_result: anyResult,
    save_workspace: has.ct,
  };
}

// ---------------------------------------------------------------------------
// Log level and history
// ---------------------------------------------------------------------------

const LEVEL_ORDER: Record<string, number> = { debug: 10, info: 20, warning: 30, error: 40, critical: 50 };

interface LogRecord {
  seq: number;
  level: LogLevel;
  message: string;
  logger?: string;
}

/** Process-wide `revisions.base_data` of the mock (+1 per add / remove). */
let baseDataRevision = 0;

const logLevels = new Map<string, LogLevel>();
const logRecords = new Map<string, LogRecord[]>();
const logCounters = new Map<string, number>();

/**
 * Number a `log` event, keep it for `session.history` and drop it below the session's level.
 * Other events pass through.
 */
export function recordLogEvent<E extends { event: string; session?: string }>(event: E): E | null {
  if (event.event !== "log" || !event.session) return event;
  const e = event as unknown as { session: string; level: LogLevel; message: string; logger?: string; seq?: number };
  const floor = LEVEL_ORDER[logLevels.get(e.session) ?? "info"];
  if ((LEVEL_ORDER[e.level] ?? 20) < floor) return null;
  const seq = (logCounters.get(e.session) ?? 0) + 1;
  logCounters.set(e.session, seq);
  e.seq = seq;
  const list = logRecords.get(e.session) ?? [];
  list.push({ seq, level: e.level, message: e.message, ...(e.logger ? { logger: e.logger } : {}) });
  logRecords.set(e.session, list.slice(-2000));
  return event;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

function settingsSections(runtime: MockRuntime): Array<{ id: SettingsSectionId; title: string; fields: FieldDescriptor[]; values: Record<string, unknown> }> {
  const web = runtime === "pyodide";
  const locked = (reason: string) => (web ? { editable: false, reason } : {});
  return [
    {
      id: "general",
      title: "General",
      fields: [
        {
          name: "base_data_path",
          label: "Base data path",
          type: "str",
          default: "",
          description: "Shared base data folders searched after your own, separated like PATH (';' on Windows, ':' elsewhere); each holds e.g. phantoms/ and machines/ folders.",
          ...locked(BROWSER_REASONS.host_paths),
        },
      ],
      values: { base_data_path: "" },
    },
    {
      id: "xp",
      title: "XP (Backend)",
      fields: [
        {
          name: "prefer_gpu",
          label: "Prefer gpu",
          type: "bool",
          default: true,
          description: "Prefer a GPU array backend whenever one is available.",
          ...locked(BROWSER_REASONS.gpu),
        },
        {
          name: "preferred_cpu_array_backend",
          label: "Preferred cpu array backend",
          type: "str",
          default: "numpy",
          description: "Preferred Array-API namespace for CPU computations. Algorithms may locally enforce a different backend.",
          ...locked("The browser runtime has numpy only."),
        },
        {
          name: "preferred_gpu_array_backend",
          label: "Preferred gpu array backend",
          type: "str",
          nullable: true,
          default: null,
          description: "Preferred Array-API namespace for GPU computations. None selects the best available backend automatically.",
          ...locked(BROWSER_REASONS.gpu),
        },
        {
          name: "jit_backends",
          label: "Jit backends",
          type: "str",
          default: "numpy,jax",
          description: "Comma-separated backends that run jit-compiled kernel paths. An empty string disables jitting entirely.",
          ...locked(BROWSER_REASONS.numba),
        },
      ],
      values: { prefer_gpu: true, preferred_cpu_array_backend: "numpy", preferred_gpu_array_backend: null, jit_backends: "numpy,jax" },
    },
    {
      id: "ai",
      title: "AI",
      fields: [
        { name: "agents_model", label: "Agents model", type: "str", default: "claude-sonnet-4-5", ...locked(BROWSER_REASONS.ai) },
        { name: "agents_display_usage", label: "Agents display usage", type: "bool", default: true, ...locked(BROWSER_REASONS.ai) },
        { name: "modelhub_hf_org", label: "Modelhub hf org", type: "str", default: "DKFZ-RadOpt", ...locked(BROWSER_REASONS.ai) },
        { name: "modelhub_offline", label: "Modelhub offline", type: "bool", default: false, ...locked(BROWSER_REASONS.ai) },
        { name: "modelhub_device", label: "Modelhub device", type: "str", default: "cpu", ...locked(BROWSER_REASONS.ai) },
      ],
      values: { agents_model: "claude-sonnet-4-5", agents_display_usage: true, modelhub_hf_org: "DKFZ-RadOpt", modelhub_offline: false, modelhub_device: "cpu" },
    },
    {
      id: "gui",
      title: "GUI",
      fields: [
        {
          name: "ui_mode",
          label: "Ui mode",
          type: "choice",
          default: "default",
          description: "Presentation mode of the GUI widgets: 'default' shows every control, 'educational' reduces the widgets to the basic operations for teaching.",
          choices: [
            { value: "default", label: "default" },
            { value: "educational", label: "educational" },
          ],
        },
      ],
      values: { ui_mode: "default" },
    },
  ];
}

// ---------------------------------------------------------------------------
// Base data
// ---------------------------------------------------------------------------

const BUNDLED_ROOT = "/lib/python3.12/site-packages/pyRadPlan/data";
const USER_ROOT = "/home/pyodide/.local/share/pyRadPlan";

const RBE_BUNDLED: BaseDataRow[] = [
  {
    name: "alpha_beta_proton.mat",
    file: "alpha_beta_proton.mat",
    source: "bundled",
    removable: false,
    model: "α/β tables + fluence spectra",
    tissue_classes: "α_x=0.1, β_x=0.05, α_x=0.15, β_x=0.05",
    reference: "Wilkens & Oelfke 2004",
  },
  {
    name: "carbon_lem.mat",
    file: "carbon_lem.mat",
    source: "bundled",
    removable: false,
    model: "LEM I",
    tissue_classes: "α_x=0.1, β_x=0.05",
    reference: "Scholz et al. 1997",
  },
  {
    name: "legacy_table.mat",
    file: "legacy_table.mat",
    source: "bundled",
    removable: false,
    problem: "KeyError: 'tissue_classes' (the table does not list tissue classes)",
  },
];

function initialBaseData(): Record<BaseDataKind, BaseDataRow[]> {
  const machine = (radiation_mode: string, name: string): BaseDataRow => ({
    name,
    file: `${radiation_mode}_${name}.mat`,
    source: "bundled",
    removable: false,
    radiation_mode,
  });
  return {
    machines: [
      machine("photons", "Generic"),
      machine("protons", "Generic"),
      machine("protons", "generic_MCsquare"),
      machine("helium", "Generic"),
      machine("carbon", "Generic"),
    ],
    rbe_tables: RBE_BUNDLED.map((r) => ({ ...r })),
    stopping_power_tables: [
      { name: "ICRU49.mat", file: "ICRU49.mat", source: "bundled", removable: false },
      { name: "PSTAR_water.mat", file: "PSTAR_water.mat", source: "bundled", removable: false },
    ],
    phantoms: ["TG119", "Liver", "Prostate", "Head and Neck"].map((n) => ({
      name: n,
      file: `${n}.mat`,
      source: "bundled" as const,
      removable: false,
    })),
  };
}

function folders(kind: BaseDataKind): BaseDataFolder[] {
  return [
    { source: "bundled", label: "Shipped with pyRadPlan", path: `${BUNDLED_ROOT}/${kind}` },
    { source: "user", label: "Added by you", path: `${USER_ROOT}/${kind}` },
  ];
}

// ---------------------------------------------------------------------------
// Licenses
// ---------------------------------------------------------------------------

const MIT = `MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.`;

const BSD3 = `BSD 3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.`;

const APACHE = `Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at the address above. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.`;

const LICENSES: LicenseDetail[] = [
  { name: "annotated-types", version: "0.7.0", license: "MIT", text: MIT },
  { name: "array-api-compat", version: "1.12.0", license: "MIT", text: MIT },
  { name: "h5py", version: "3.14.0", license: "BSD-3-Clause", text: BSD3 },
  { name: "numpy", version: "2.2.5", license: "BSD-3-Clause", text: BSD3 },
  { name: "packaging", version: "25.0", license: "Apache-2.0 OR BSD-2-Clause", text: APACHE },
  { name: "pydantic", version: "2.11.7", license: "MIT", text: MIT },
  { name: "pydantic-core", version: "2.33.2", license: "MIT", text: MIT },
  { name: "pydicom", version: "3.0.1", license: "MIT", text: MIT },
  { name: "pyRadPlan", version: "0.9.0", license: "BSD-3-Clause", text: BSD3 },
  { name: "scipy", version: "1.15.2", license: "BSD-3-Clause", text: BSD3 },
  { name: "SimpleITK", version: "2.5.0 (pyodide shim)", license: "Apache-2.0", text: APACHE },
  { name: "six", version: "1.17.0", license: "MIT", text: MIT },
  { name: "tqdm", version: "4.67.1", license: "MPL-2.0 AND MIT", text: null },
  { name: "typing-extensions", version: "4.14.0", license: "PSF-2.0", text: null },
];

// ---------------------------------------------------------------------------
// Export payloads
// ---------------------------------------------------------------------------

const ENC = new TextEncoder();

function npy(values: Float32Array): Uint8Array {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${values.length},), }`;
  const padded = header.padEnd(Math.ceil((header.length + 11) / 64) * 64 - 1, " ") + "\n";
  const out = new Uint8Array(10 + padded.length + values.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, padded.length & 0xff, padded.length >> 8]);
  out.set(ENC.encode(padded), 10);
  out.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), 10 + padded.length);
  return out;
}

function matFile(note: string): Uint8Array {
  const header = ENC.encode(`MATLAB 5.0 MAT-file, Platform: MOCK, Created by the pyRadPlan web mock: ${note}`.padEnd(116, " "));
  const out = new Uint8Array(128 + 8);
  out.set(header.subarray(0, 116));
  out.set([0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x01, 0x49, 0x4d], 116);
  return out;
}

function pickleFile(note: string): Uint8Array {
  const text = ENC.encode(note);
  return new Uint8Array([0x80, 0x04, 0x95, ...text, 0x2e]);
}

function hdf5File(note: string): Uint8Array {
  const text = ENC.encode(note);
  return new Uint8Array([0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a, ...text]);
}

function dicomFile(note: string): Uint8Array {
  const out = new Uint8Array(132 + note.length);
  out.set(ENC.encode("DICM"), 128);
  out.set(ENC.encode(note), 132);
  return out;
}

const EXPORT_SIZE = 64;

/** File name stem of a component key: the `/` after a result name is no file name character. */
const fileStem = (key: string) => key.replace(/\//g, "_");

function exportPayload(
  object: string,
  format: string,
  structureFormat: string | undefined,
  quantities: string[],
  basename: string | undefined,
  extension: string,
  mime: string,
): { filename: string; mime: string; data: Uint8Array; files?: string[] } {
  const stem = basename ?? (object === "workspace" ? "workspace" : object);
  const values = new Float32Array(EXPORT_SIZE).map((_, i) => i / EXPORT_SIZE);
  const note = `mock ${object} (${format})`;
  if (object === "result") {
    if (format === "dcm") {
      const doses = quantities.filter((q) => /(^|\/)(physical_dose|rbe_x_dose)$/.test(q));
      const results = [...new Set(doses.map((q) => (q.includes("/") ? q.split("/")[0] : "")))];
      const folder = (result: string) => (results.length > 1 ? `${result || "current"}/` : "");
      const files = results.flatMap((result) => [
        `${folder(result)}CT/CT.0001.dcm`,
        ...doses
          .filter((q) => (q.includes("/") ? q.split("/")[0] : "") === result)
          .map((q) => `${folder(result)}RTDOSE_${q.split("/").pop()}.dcm`),
        ...(result === "" ? [`${folder(result)}RTPLAN.dcm`] : []),
      ]);
      const data = zip(files.map((name) => ({ name, data: dicomFile(`mock ${name}`) })));
      return { filename: `${basename ?? "result"}_dicom.zip`, mime: "application/zip", data, files };
    }
    if (format === "mat") {
      const name = basename ?? (quantities.length === 1 ? fileStem(quantities[0]) : "result");
      return { filename: `${name}${extension}`, mime, data: matFile(`resultGUI ${quantities.join(", ")}`) };
    }
    const one = (q: string): { name: string; data: Uint8Array } => ({
      name: `${fileStem(q)}${extension}`,
      data: zip([{ name: `${fileStem(q)}.npy`, data: npy(values) }]),
    });
    if (quantities.length === 1) {
      const entry = one(quantities[0]);
      return { filename: `${basename ?? fileStem(quantities[0])}${extension}`, mime, data: entry.data };
    }
    const entries = quantities.map(one);
    return { filename: `${basename ?? "result"}.zip`, mime: "application/zip", data: zip(entries), files: entries.map((e) => e.name) };
  }
  if (format === "nrrd" && structureFormat === "layers") {
    const files = ["cst.seg.nrrd", "cst.json"];
    return {
      filename: `${stem}.zip`,
      mime: "application/zip",
      data: zip(files.map((name) => ({ name, data: ENC.encode(`mock ${name}`) }))),
      files,
    };
  }
  if (format === "dcm") {
    const kind = structureFormat === "seg" ? "SEG" : "RTSTRUCT";
    const files = ["CT/CT.0001.dcm", `${kind}.dcm`];
    return {
      filename: `${stem}_${kind.toLowerCase()}.zip`,
      mime: "application/zip",
      data: zip(files.map((name) => ({ name, data: dicomFile(`mock ${name}`) }))),
      files,
    };
  }
  if (format === "npz") {
    return { filename: `${stem}${extension}`, mime, data: zip([{ name: "cst.npy", data: npy(values) }]) };
  }
  if (format === "pickle") return { filename: `${stem}${extension}`, mime, data: pickleFile(note) };
  if (format === "hdf5") return { filename: `${stem}${extension}`, mime, data: hdf5File(note) };
  return { filename: `${stem}${extension}`, mime, data: matFile(note) };
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

function bytesLength(ctx: MockContext, ref: unknown): number {
  const index = (ref as { $array?: number } | undefined)?.$array;
  const buffer = index === undefined ? undefined : ctx.request.buffers?.[index];
  return buffer?.byteLength ?? 0;
}

function stem(filename: string): string {
  const base = filename.split(/[\\/]/).pop() ?? filename;
  return base.replace(/\.[^.]*$/, "");
}

function formSchema(section: { id: SettingsSectionId; title: string; fields: FieldDescriptor[] }, values: Record<string, unknown>): FormSchema {
  return { id: `settings:${section.id}`, title: `${section.title} settings`, fields: section.fields, values: { ...values }, explicit: [] };
}

const IMAGE_EXTENSIONS = /\.(nii|nii\.gz|nrrd|mha|mhd|dcm)$/i;
const PICKLE_EXTENSIONS = /\.(pkl|pickle)$/i;
const CONTAINER_EXTENSIONS = /\.(mat|npz|h5|hdf5|pkl|pickle)$/i;

/** Handler table plus the process-wide mock state (settings, base data) it works on. */
export function createIoHandlers(runtime: MockRuntime = mockRuntime()): MockHandlers {
  const capabilities = mockCapabilities(runtime);
  const web = runtime === "pyodide";
  const settings = new Map(settingsSections(runtime).map((s) => [s.id, { ...s.values }]));
  const baseData = initialBaseData();

  const settingsResult = (): { sections: SettingsSection[] } => ({
    sections: settingsSections(runtime).map((s) => ({
      id: s.id,
      title: s.title,
      form: formSchema(s, settings.get(s.id) ?? s.values),
    })),
  });

  const baseDataRow = (kind: BaseDataKind, name: string, mode: string, file: string): BaseDataRow => ({
    name,
    file,
    source: "user",
    removable: true,
    path: `${USER_ROOT}/${kind}/${file}`,
    ...(kind === "machines" ? { radiation_mode: mode } : {}),
    ...(kind === "rbe_tables" ? { model: "α/β tables + fluence spectra", tissue_classes: "α_x=0.1, β_x=0.05", reference: "" } : {}),
  });

  const handlers: MockHandlers = {
    "app.info": async (_params, ctx) => {
      const info = (await ctx.fallback()) as AppInfo;
      return { ...info, runtime, capabilities, default_profile: web ? "web" : "native" };
    },

    "data.load_file": (params, ctx) => {
      const filename = String(params.filename ?? "");
      const kind = String(params.kind ?? "auto");
      const s = ctx.session();
      // A dose image (with image IO) or a dose-only `.npz` (recognized by its name; the mock reads
      // no files) joins the patient as a result of its own; everything else is a new patient.
      const doseImage = kind === "dose" && !web;
      const doseNpz = kind === "auto" && /\.npz$/i.test(filename) && /dose/i.test(filename) && s.phantom !== null;
      if (doseImage || doseNpz) {
        if (!s.phantom) throw precondition("Load a CT before importing a dose.", "no_data");
        return ctx.run(s, `Loading ${filename}`, async () => {
          ctx.progress(s, `Reading ${filename}`, null);
          await ctx.sleep(150);
          const name = addMockDoseResult(s, params.name === undefined || params.name === null ? stem(filename) : String(params.name));
          ctx.log(s, "info", `Imported the dose of ${filename} as the result '${name}'`);
          return ctx.publish(s);
        });
      }
      if (kind !== "auto") {
        throw precondition(
          web && IMAGE_EXTENSIONS.test(filename)
            ? BROWSER_REASONS.image_file_io
            : `Importing an image as '${kind}' (Qt Import Image dialog) is not available through the protocol yet`,
          "not_available",
        );
      }
      if (PICKLE_EXTENSIONS.test(filename) && web) throw precondition(BROWSER_REASONS.pickle_load, "not_available");
      if (IMAGE_EXTENSIONS.test(filename)) {
        if (web) throw precondition(`Cannot load ${filename}: ${BROWSER_REASONS.image_file_io}`, "not_available");
      } else if (!CONTAINER_EXTENSIONS.test(filename)) {
        throw new MethodError("ValueError", `Unsupported file type: ${filename}`);
      }
      return ctx.fallback();
    },

    "data.load_dij": (params, ctx) => {
      const s = ctx.session();
      const filename = String(params.filename ?? "");
      if (!/\.mat$/i.test(filename)) throw new MethodError("ValueError", `Unsupported file type: ${filename}`);
      return ctx.run(s, "Loading dose influence", async () => {
        ctx.progress(s, "Loading dose influence", null);
        await ctx.sleep(200);
        if (/nodij/i.test(filename)) throw precondition("No dose-influence matrix found in the file.", "no_dij");
        ctx.touch(s, "stf");
        ctx.touch(s, "dij");
        ctx.log(s, "info", `Loaded steering information and dose influence from ${filename}`);
        return ctx.publish(s);
      });
    },

    "export.save": (params, ctx) => {
      const s = ctx.session();
      const state = ctx.state(s);
      const object = String(params.object ?? "");
      const format = String(params.format ?? "");
      const structureFormat = ["seg", "rtstruct", "layers"].includes(String(params.structure_format))
        ? String(params.structure_format)
        : undefined;
      const variant = (e: ExportFormat) => (e as ExportFormat & { structure_format?: string }).structure_format;
      const entry = capabilities.formats.export.find(
        (e) =>
          e.object === object &&
          e.format === format &&
          (object !== "cst" ||
            (format === "dcm" ? variant(e) === (structureFormat ?? "rtstruct") : variant(e) === structureFormat)),
      );
      if (!entry) throw new MethodError("ValueError", `Cannot export ${object} as ${format}`);
      if (!entry.available) throw precondition(entry.reason ?? "This format is not available", "not_available");
      const hasKey = ({ pln: "pln", dij: "dij", cst: "cst", result: "result", workspace: "ct" } as const)[entry.object];
      // Any result can be exported, also without an active one.
      const present = object === "result" ? Boolean(state.actions.save_result) : state.has[hasKey];
      if (!present) throw precondition(`No ${object} available to save.`, "no_data");
      // Default: the full quantities of all results, by their component keys.
      const quantities = (params.quantities as string[] | undefined) ?? state.quantities;
      if (object === "result" && quantities.length === 0) throw precondition("The result has no exportable image quantities.", "no_data");
      if (object === "result" && format === "dcm" && !quantities.some((q) => /(^|\/)(physical_dose|rbe_x_dose)$/.test(q))) {
        throw precondition("DICOM RTDOSE stores physical and RBE-weighted doses only; select one of them", "no_data");
      }
      return ctx.run(s, `Saving ${object}`, async () => {
        ctx.progress(s, `Saving ${object}`, null);
        await ctx.sleep(150);
        const payload = exportPayload(object, format, structureFormat, quantities, params.basename as string | undefined, entry.extension, entry.mime);
        ctx.log(s, "info", `Saved ${object} as ${payload.filename} (${payload.data.length} bytes)`);
        return {
          filename: payload.filename,
          mime: payload.mime,
          data: ndarray(payload.data, [payload.data.length]),
          ...(payload.files ? { files: payload.files } : {}),
        };
      });
    },

    "workflow.snapshot": async (params, ctx) => {
      // The results module owns the data model; this stands in when it has no handler.
      try {
        return await ctx.fallback();
      } catch (err) {
        if (!(err instanceof MethodError && err.type === "NotImplementedError")) throw err;
      }
      const s = ctx.session();
      const state = ctx.state(s);
      if (!state.has.result) throw precondition("No active result available", "no_data");
      const raw = params.tag === undefined ? "result" : String(params.tag);
      const tag = raw.trim().replace(/\W+/g, "_").replace(/^_+|_+$/g, "");
      if (!tag) throw new MethodError("ValueError", "The snapshot needs a name");
      ctx.log(s, "info", `Saved result snapshot '${tag}'`);
      return { state: ctx.publish(s), tag };
    },

    "base_data.list": (params) => {
      const kind = String(params.kind) as BaseDataKind;
      const rows = baseData[kind];
      if (!rows) throw new MethodError("ValueError", `Unknown base data kind ${kind}`);
      const root = (source: string) => (source === "user" ? USER_ROOT : BUNDLED_ROOT);
      const result: BaseDataList = {
        kind,
        folders: folders(kind),
        rows: rows.map((r) => ({ ...r, path: r.path ?? `${root(r.source)}/${kind}/${r.file}` })),
      };
      return result;
    },

    "base_data.add": (params, ctx) => {
      const kind = String(params.kind) as BaseDataKind;
      const rows = baseData[kind];
      if (!rows || kind === "phantoms") throw new MethodError("ValueError", `Cannot add base data of kind ${kind}`);
      const filename = String(params.filename ?? "");
      const label = kind === "machines" ? "machine" : kind === "rbe_tables" ? "RBE table" : "stopping power table";
      if (!/\.mat$/i.test(filename) || bytesLength(ctx, params.data) === 0 || /invalid|broken/i.test(filename)) {
        throw new MethodError("ValueError", `Could not read a ${label} from ${filename}: not a valid MATLAB file`);
      }
      let mode = String(params.radiation_mode ?? "");
      let suggested = stem(filename);
      if (kind === "machines") {
        const [m, ...rest] = suggested.split("_");
        if (["photons", "protons", "helium", "carbon", "oxygen", "VHEE"].includes(m) && rest.length) {
          mode = mode || m;
          suggested = rest.join("_");
        }
        mode = mode || "photons";
      } else {
        suggested = `${suggested}.mat`;
      }
      const name = params.name === undefined ? suggested : String(params.name);
      const file = kind === "machines" ? `${mode}_${name}.mat` : name.toLowerCase().endsWith(".mat") ? name : `${name}.mat`;
      const existing = rows.find((r) => r.file === file);
      const clash = existing ? { file: existing.file, source: existing.source } : null;
      const result: BaseDataAddResult = {
        row: baseDataRow(kind, kind === "machines" ? name : file, mode, file),
        suggested_name: suggested,
        radiation_mode: mode,
        clash,
      };
      if (params.dry_run) return result;
      if (existing && (existing.source !== "user" || !params.overwrite)) {
        throw precondition(`A ${label} named '${name}' already exists`, "name_clash");
      }
      if (existing) rows.splice(rows.indexOf(existing), 1);
      rows.push(result.row as BaseDataRow);
      baseDataRevision += 1;
      const s = ctx.request.session ? ctx.session() : null;
      if (s) {
        ctx.log(s, "info", `Added the ${label} '${name}'`);
        ctx.publish(s);
      }
      return result;
    },

    "base_data.remove": (params, ctx) => {
      const kind = String(params.kind) as BaseDataKind;
      const rows = baseData[kind];
      const row = rows?.find((r) => r.file === params.file);
      if (!row) throw new MethodError("FileNotFoundError", `No ${kind} file ${String(params.file)}`);
      if (!row.removable) throw precondition("Only files you added can be removed", "not_removable");
      rows.splice(rows.indexOf(row), 1);
      baseDataRevision += 1;
      if (ctx.request.session) ctx.publish(ctx.session());
      return {};
    },

    "session.create": async (_params, ctx) => {
      const created = (await ctx.fallback()) as { session: string };
      logLevels.delete(created.session);
      logRecords.delete(created.session);
      logCounters.delete(created.session);
      return created;
    },

    "app.settings": () => settingsResult(),

    "app.settings_set": (params) => {
      const id = String(params.section) as SettingsSectionId;
      const section = settingsSections(runtime).find((s) => s.id === id);
      if (!section) throw new MethodError("ValueError", `Unknown settings section ${id}`);
      const values = (params.values ?? {}) as Record<string, unknown>;
      const details = Object.keys(values).flatMap((name) => {
        const field = section.fields.find((f) => f.name === name);
        if (!field) return [{ field: name, message: "unknown setting" }];
        if (field.editable === false) return [{ field: name, message: field.reason ?? "not editable" }];
        if (field.type === "choice" && !field.choices?.some((c) => c.value === values[name])) {
          return [{ field: name, message: `must be one of ${field.choices?.map((c) => String(c.value)).join(", ")}` }];
        }
        return [];
      });
      if (details.length) throw new MethodError("ValidationError", "Invalid settings", "invalid", details);
      settings.set(id, { ...(settings.get(id) ?? {}), ...values });
      return settingsResult();
    },

    "app.licenses": (params) => {
      if (params.name === undefined) {
        const rows: LicenseRow[] = LICENSES.map(({ name, version, license }) => ({ name, version, license }));
        return rows;
      }
      const found = LICENSES.find((l) => l.name.toLowerCase() === String(params.name).toLowerCase());
      if (!found) throw new MethodError("KeyError", `Unknown distribution ${String(params.name)}`);
      return found;
    },

    "session.log_level": (params, ctx) => {
      const s = ctx.session();
      const level = String(params.level);
      if (!(level in LEVEL_ORDER) || level === "critical") throw new MethodError("ValueError", `Unknown log level ${level}`);
      logLevels.set(s.id, level as LogLevel);
      return { level };
    },

    "session.history": async (params, ctx) => {
      const s: MockSession = ctx.session();
      const since = Number(params.since ?? 0);
      let optimization: unknown = null;
      try {
        const rest = (await ctx.fallback()) as { optimization?: unknown } | undefined;
        optimization = rest?.optimization ?? null;
      } catch {
        optimization = null;
      }
      return { log: (logRecords.get(s.id) ?? []).filter((r) => r.seq > since), optimization };
    },
  };
  return handlers;
}

/** Additions of package C to every SessionState (`actions`). */
export function ioStateExtension(_s: MockSession, state: SessionState): Partial<SessionState> {
  return {
    actions: { ...state.actions, ...ioActionFlags(state.has, state.quantities.length > 0, state.has.result || (state.results?.names.length ?? 0) > 0) },
    ...(state.revisions ? { revisions: { ...state.revisions, base_data: baseDataRevision } } : {}),
  };
}
