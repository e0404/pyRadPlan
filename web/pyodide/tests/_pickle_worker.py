"""Subprocess helper for ``test_pickle_compat.py``: pickle with real SimpleITK or the shim.

    python _pickle_worker.py {real|shim} {make-images|load-images|save-tg119|load-tg119} IN/OUT [OUT]

``shim`` puts ``web/pyodide/shims`` first on ``sys.path``, so ``import SimpleITK`` (and the
``SimpleITK.SimpleITK`` module pickles refer to) resolve to the shim.
"""

import json
import pickle
import sys
from pathlib import Path

SHIMS = Path(__file__).resolve().parents[1] / "shims"
PROTOCOLS = (2, 4, 5)


def make_images(sitk, np):
    rng = np.random.default_rng(42)
    ct = sitk.GetImageFromArray((rng.normal(size=(5, 6, 7)) * 300).astype(np.int16))
    ct.SetSpacing((3.0, 3.0, 2.5))
    ct.SetOrigin((-250.0, -250.0, -160.0))
    mask = sitk.GetImageFromArray((rng.random((5, 6, 7)) > 0.5).astype(np.uint8))
    mask.CopyInformation(ct)
    dose = sitk.GetImageFromArray(rng.random((4, 5, 6)))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q[:, 0] *= np.sign(np.linalg.det(q))
    dose.SetDirection(tuple(q.ravel()))
    dose.SetMetaData("unit", "Gy")
    vec = sitk.GetImageFromArray(rng.random((3, 4, 5, 2)).astype(np.float32), isVector=True)
    four_d = sitk.JoinSeries([mask, mask])
    flat = sitk.GetImageFromArray(rng.integers(0, 9, size=(6, 8)).astype(np.int64))
    return {"ct": ct, "mask": mask, "dose": dose, "vector": vec, "4d": four_d, "2d": flat}


def describe(sitk, np, images):
    arrays, meta = {}, {}
    for key, img in images.items():
        assert isinstance(img, sitk.Image), type(img)
        arrays[key] = sitk.GetArrayFromImage(img)
        meta[key] = {
            "size": list(img.GetSize()),
            "pixel_id": img.GetPixelIDValue(),
            "components": img.GetNumberOfComponentsPerPixel(),
            "spacing": list(img.GetSpacing()),
            "origin": list(img.GetOrigin()),
            "direction": list(img.GetDirection()),
            "metadata": {k: img.GetMetaData(k) for k in img.GetMetaDataKeys()},
        }
    return arrays, meta


def save_description(np, out, arrays, meta):
    np.savez(out, meta=np.array(json.dumps(meta)), **arrays)


def main(runtime, action, *paths):
    if runtime == "shim":
        sys.path.insert(0, str(SHIMS))
    import numpy as np
    import SimpleITK as sitk

    if (runtime == "shim") != bool(getattr(sitk, "IS_PYRADPLAN_SHIM", False)):
        raise RuntimeError(f"wrong SimpleITK for {runtime}: {sitk.__file__}")

    if action == "make-images":
        images = make_images(sitk, np)
        for protocol in PROTOCOLS:
            Path(f"{paths[0]}.p{protocol}").write_bytes(pickle.dumps(images, protocol=protocol))
    elif action == "load-images":
        images = pickle.loads(Path(paths[0]).read_bytes())
        save_description(np, paths[1], *describe(sitk, np, images))
    elif action == "save-tg119":
        from pyRadPlan import load_tg119, save_data

        ct, cst = load_tg119()
        save_data(ct=ct, cst=cst, file_name=paths[0], format="pickle")
    elif action == "load-tg119":
        from pyRadPlan import load_data

        data = load_data(paths[0])
        images = {"ct": data["ct"].cube_hu}
        images.update({f"voi:{voi.name}": voi.mask for voi in data["cst"].vois})
        arrays, meta = describe(sitk, np, images)
        meta["voi_types"] = [voi.voi_type for voi in data["cst"].vois]
        meta["ct_type"] = type(data["ct"]).__name__
        save_description(np, paths[1], arrays, meta)
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main(*sys.argv[1:])
