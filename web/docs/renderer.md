# Slice renderer: options and decision

The web viewer has to reproduce the Qt slice viewer (`pyRadPlan/gui/widgets/result/quantity_widget.py`,
a pyqtgraph 2-D viewer with custom overlays): grayscale CT with window/level, a colour-mapped
quantity overlay (zero = transparent, opacity, range, colormap, colourbar), VOI contours, isolines,
three planes, slice scrolling, zoom/pan, physical aspect ratio and a hover readout. A 3-D view is
not needed now (the Qt GUI has none).

## Decision

**Hand-written WebGL2** for the image layers (`src/renderer/sliceRenderer.ts`, `glVolume.ts`,
`shaders.ts`) plus a **Canvas2D overlay** for contours, isolines, isocenter, orientation letters and
the crosshair. CT and quantity are uploaded once as 3-D textures; slicing, window/level, LUT and
blending happen in one fragment shader, so scrolling and windowing never go back to Python or even
re-upload data. Contours and isolines are traced on the CPU with marching squares per slice
(`marchingSquares.ts`, `contours.ts`), cropped to each structure's bounding box and LRU-cached like
the Qt viewer's contour cache. No rendering dependency: the renderer is ~900 lines of framework-free
TypeScript with a thin React wrapper (`components/SliceViewer.tsx`).

## Options compared

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **(a) Hand-written WebGL2** | Exact control over the Qt look (LUT indexing, zero-transparency, nearest-voxel display); 3-D textures keep volumes on the GPU; integer textures (`R16I`) hold HU without conversion; no dependency, tiny bundle; our own data model (`(z, y, x)` arrays + grid). | We own the code (shader, texture formats, fallbacks); no ready-made 3-D/MPR tools. | **Chosen.** |
| (b) vtk.js | Full VTK pipeline, volume rendering, MPR (`vtkImageResliceMapper`), good for a later 3-D view. | Large (≈1–2 MB min for the needed parts), VTK data model (`vtkImageData`) and its widget/interactor conventions; reproducing pyqtgraph's exact overlays (zero-transparent LUT, isocurves, Qt colourbar) means fighting the abstractions. | Use later for a 3-D view, next to (a). |
| (c) Cornerstone3D | Clinical-grade viewport tools (WL presets, segmentation display, synchronizers). | DICOM-centric (image ids, metadata providers, volume loaders); RT dose/structure display needs its segmentation model; heavy (≈ vtk.js + its own core); a lot of indirection for arrays that already sit in memory. | Rejected: wrong data model for numpy arrays from Python. |
| (d) NiiVue | Small, fast WebGL2 viewer, NIfTI-oriented, multiplanar + 3-D, overlays with colormaps. | Own UI/interaction model and canvas layout; overlays are volumes with its own thresholds; contour lines from masks and Qt-like isolines are not its idiom; controlling every pixel like pyqtgraph is harder than writing the shader. | Good fallback if a 3-D render is wanted quickly; not chosen. |
| (e) Canvas2D `putImageData` | Trivial; CPU code matches numpy exactly. | Every slice change / WL drag recomputes and re-uploads all pixels on the main thread (512² × LUT per frame), zoom needs `drawImage` scaling (smoothing control, no exact voxel grid at high zoom). Fine for 2-D but sluggish for interactive windowing on 512² slices. | Rejected as main path; the slice fallback below keeps a similar per-slice path on the GPU. |
| (f) three.js / regl / twgl | Helpers reduce WebGL boilerplate. | three.js is a scene graph we do not need (≈600 kB); regl/twgl save ~100 lines but add a dependency for a single full-screen pass. | Not needed: one program, four textures. |
| WebGPU | Compute shaders (e.g. GPU marching squares, DVH), storage textures, `r16sint` 3-D textures. | Not available everywhere yet (Safari/Firefox rollout, Linux/Android variance), no SwiftShader-equivalent in all CI images. | Future path: the shader is small and ports to WGSL one-to-one. |

## Texture formats and limits

| Item | Status |
|---|---|
| `texelFetch` on `R16I` (`isampler3D`) for int16 CT, `R32F` for float32 CT/dose, `R8UI`/`R16UI`/`R32I` | **verified** (mock CT is int16, dose float32; real app layer sends float32 CT per amendment 12) |
| Nearest-voxel display (pyqtgraph `ImageItem` draws voxels as squares) | **verified**: `texelFetch` needs no filtering, so float textures work without `OES_texture_float_linear` |
| Float linear filtering | `OES_texture_float_linear` reported present in SwiftShader/Edge (verified); not used. Integer textures are never filterable; a smooth display would need `R16F`/`R32F` + `LINEAR` |
| `MAX_3D_TEXTURE_SIZE` | **verified 2048** (Edge 157, ANGLE/SwiftShader); WebGL2 spec minimum is 256 (spec fact); desktop GPUs typically report 2048 (assumed from ANGLE D3D11/GL limits) |
| Fallback for larger volumes or failed allocation (`GL_OUT_OF_MEMORY`) | **implemented and verified**: the volume stays on the CPU and only the displayed slice is uploaded (1-voxel-deep 3-D texture, same shader). Forced with `?sliceMode` |
| 512 × 512 × 300 volume | **assumed fine** on desktop: 157 MB as int16, 315 MB as float32 (GPU) plus the same in the JS heap; each dim ≤ 2048. Not measured. On memory-tight devices the float32 CT from the backend is the cost driver: converting HU to int16 client-side (or sending int16) halves it |
| TG119 (167 × 167 × 129) | **verified**: 7.2 MB CT texture, renders and scrolls in SwiftShader |
| Software rendering (SwiftShader, headless CI) | **verified** with Playwright + `--use-angle=swiftshader` |

Not measured: frame times on real GPUs, behaviour on Safari/iOS (WebGL2 3-D integer textures are
supported there since Safari 15; assumed), multi-GB volumes.

## Geometry

`geometry.ts` maps planes to array axes. The default convention mirrors the Qt viewer (and matRad):
axial shows x → right, y ↓; sagittal z → right, y ↓; coronal z → right, x ↓. A `radiological`
convention (head up in sagittal/coronal) is available in the renderer (`convention` setting) but not
exposed in the UI. The slice is fitted into the canvas at the physical aspect ratio
(`spacing[h] / spacing[v]`, as `setAspectLocked(ratio=…)` in Qt), then zoomed/panned. The shader gets
one matrix from `gl_FragCoord` to slice coordinates and per-volume integer texel mappings, so the same
code serves all planes and the slice fallback. Direction cosines are used for the world readout only;
display assumes the array axes are the patient axes (as the Qt viewer does).

## Contours and isolines

Decision: **CPU marching squares per slice, drawn on a Canvas2D overlay.**

- It reproduces the Qt algorithm (`contourpy` on the edge-padded slice, samples at voxel centres,
  coordinates clipped to the slice) so contours coincide with the Qt GUI; unit-tested.
- Lines are vector paths: crisp and antialiased at any zoom, constant 1.5 px width (Qt pen width),
  colours from the structure set.
- Cost is small: each structure is cropped to its bounding box (+1 voxel), results are cached per
  (structure, plane, slice) in a 4096-entry LRU, so scrolling back is free. Isolines (50/80/95 % of the
  maximum by default, or custom levels; coloured by the quantity colormap at `level / max`) use the
  same code on the quantity slice.

The shader alternative (edge detection on a label / bitmask 3-D texture) would draw outlines without
CPU work and scale to many structures, but gives voxel-stair, zoom-dependent outlines (thickness in
voxels, not pixels), needs a packed label texture (max 32 structures per `R32UI` channel, overlaps need
bits not labels) and cannot do sub-voxel interpolated isolines. It becomes attractive with a 3-D view or
hundreds of structures; the protocol would then better send one bit-packed mask volume instead of one
uint8 volume per structure (which costs `n_structures × voxels` bytes today).

## v1 additions (round 2)

- **Beams** (`renderer/beams.ts`): `view.beams` geometry (LPS mm) is projected per plane in
  "slice millimetres"; wedge widths, ⊙/⊗ marks, the 0.035 / 0.087 / 0.3 thresholds, the source
  radius (largest circle inside the slice) and the orbit drag (`orbitAngle` solves the 2x2 orbit
  matrix, Esc cancels) follow `quantity_widget.py`. The CT direction matrix maps LPS to voxels (Qt
  ignores it, parity ambiguity 8). Pending (draft) beams are dashed with "*".
- **Orientation marker** (`renderer/orientation.ts`): the Qt mannequin (capsules, perspective,
  cut at the slice, depth shading, edge letters) drawn into the overlay's top-left corner.
- **Compact masks**: `MaskStorage` holds a full uint8 volume, a v1 `"crops"` crop with its bbox, or
  one bit of a v1 `"bits"` plane; contours, bounds and the target centre of mass read all three
  without expanding them. The frontend requests `"crops"` from a v1 backend.
- **int16 CT**: `view.volume {encoding: "auto"}`; the `R16I` texture path was already in place.
