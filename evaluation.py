import argparse
import csv
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from skimage.metrics import structural_similarity
from sklearn.metrics import auc, roc_auc_score, roc_curve


CLIENTS = {"T1": (1, 2, 3, 4), "T6": (5, 6), "T1_denoising": (25, 26, 27, 28),
           "T2": (7, 8, 9, 10), "T3": (11, 12, 13, 14, 3),
           "T4": (15, 16, 17, 18, 19, 20), "T5": (21, 22, 23, 24), "T7": (15, 16, 17, 18, 19, 20)}
CELLS = ("neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte",
         "immature_granulocyte", "erythroblast", "platelet")
ORGANS = ("kidney_left", "kidney_right", "liver", "spleen", "pancreas", "stomach")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def resolved(value, root):
    return str((Path(root) / value).resolve())


class References:
    def __init__(self, path, root):
        self.root, self.index = root, defaultdict(list)
        for row in read_csv(path) if path else []:
            if row.get("split", "test") != "test":
                continue
            client_value = row.get("client", "") or row.get("data_client", "")
            client = str(int(client_value)) if client_value else "*"
            for field in ("image_id", "case_id"):
                if row.get(field):
                    self.index[client, field, row[field]].append(row)
            source = next((row[k] for k in ("source_path", "source", "image", "path") if row.get(k)), "")
            if source:
                self.index[client, "source", resolved(source, root)].append(row)

    def find(self, item):
        source = next((item[k] for k in ("source_path", "source", "path") if item.get(k)), "")
        queries = [("image_id", item.get("image_id", "")),
                   ("source", resolved(source, self.root) if source else ""),
                   ("case_id", item.get("case_id", ""))]
        for field, value in queries:
            if not value:
                continue
            rows = self.index.get((str(item["client"]), field, value), []) or self.index.get(("*", field, value), [])
            if rows:
                if len(rows) != 1:
                    raise ValueError(f"Ambiguous reference for client {item['client']}, {field}={value}; identify each source image.")
                return rows[0]
        raise ValueError(f"Missing reference/ROI entry: {item}")


@lru_cache(maxsize=2)
def load_array(path, key=""):
    file = Path(path)
    spacing, affine = None, None
    if file.name.endswith((".nii", ".nii.gz")):
        image = nib.load(file)
        factor = {"meter": 1000., "micron": .001}.get(image.header.get_xyzt_units()[0], 1.)
        value = np.asanyarray(image.dataobj)
        spacing = tuple(v * factor for v in image.header.get_zooms())
        affine = image.affine.copy()
        affine[:3] *= factor
    elif file.suffix == ".npy":
        value = np.load(file, allow_pickle=False)
    elif file.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}:
        with Image.open(file) as image:
            value = np.array(image)
        if value.ndim == 3:
            if not np.array_equal(value[..., 0], value[..., 1]) or not np.array_equal(value[..., 0], value[..., 2]):
                raise ValueError(f"Reference masks must be grayscale: {file}")
            value = value[..., 0]
    else:
        if file.suffix == ".npz":
            with np.load(file, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        elif file.suffix.lower() in {".h5", ".hdf5"}:
            with h5py.File(file, "r") as data:
                chosen = key or next((k for k in ("reconstruction_rss", "target", "mask", "segmentation") if k in data), "")
                if not chosen:
                    raise ValueError(f"Specify a reference key for {file}")
                arrays = {chosen: data[chosen][()]}
                key = chosen
        else:
            arrays = loadmat(file)
        chosen = key or next((k for k in ("target", "ndct", "mask", "segmentation", "label", "data", "arr_0", "image") if k in arrays), "")
        if chosen not in arrays:
            raise ValueError(f"Missing reference array {chosen!r}: {file}")
        value = arrays[chosen]
        if "spacing" in arrays:
            spacing = tuple(np.asarray(arrays["spacing"]).ravel())
    return np.asarray(value), spacing, affine


def numbers(text):
    return tuple(float(x) for x in re.split(r"[,x\s]+", str(text).strip("[]() ")) if x)


def reference(row, item, prediction, args):
    name = next((row[k] for k in ("reference", "mask", "path") if row.get(k)), "")
    if not name:
        raise ValueError("Reference CSV needs reference or mask; path also accepts a prepared reference file.")
    path = resolved(name, args.data_root)
    value, spacing, affine = load_array(path, row.get("reference_key", "") or args.reference_key)
    if args.task == "T3" and args.space == "preprocessed" and value.ndim == 4:
        if value.shape[0] != 2:
            raise ValueError("nnUNet reference data must contain image and label channels.")
        value = value[1]
    if value.ndim == prediction.ndim + 1:
        index = row.get("slice", item.get("slice"))
        if index is None or index == "":
            raise ValueError(f"Specify a reference slice: {path}")
        axis = int(row.get("axis", 2 if path.endswith((".nii", ".nii.gz")) else 0))
        value = np.take(value, int(index), axis=axis)
        if spacing is not None:
            spacing = tuple(v for i, v in enumerate(spacing) if i != axis)
        affine = None
    spacing = numbers(row["spacing"]) if row.get("spacing") else spacing
    return value, spacing, affine, path


def validate_pair(prediction, truth):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError(f"Prediction/reference shape or finite-value mismatch: {prediction.shape}, {truth.shape}")
    if np.iscomplexobj(prediction) or np.iscomplexobj(truth):
        raise ValueError("Evaluate magnitude MRI images or real CT images, not complex arrays.")


def reconstruction_metrics(prediction, truth, ct, args):
    validate_pair(prediction, truth)
    left, right = prediction.astype(np.float64), truth.astype(np.float64)
    if left.ndim != 2 or min(left.shape) < 11:
        raise ValueError("Reconstruction metrics require full 2D images at least 11 by 11 pixels.")
    if ct:
        scale, offset, units = 3000., -1000., "HU"
    else:
        scale, offset = args.data_range or float(right.max()), 0.
        units = "reference-normalized MRI" if args.mri_scale == "reference_max" else "source MRI units"
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A positive reference data range is required; specify --data-range for zero-valued MRI references.")
    mse = float(np.mean((left - right) ** 2))
    rmse = math.sqrt(mse) / (scale if not ct and args.mri_scale == "reference_max" else 1)
    ssim = structural_similarity((right - offset) / scale, (left - offset) / scale, data_range=1.,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)
    return dict(psnr=math.inf if mse == 0 else 10 * math.log10(scale ** 2 / mse), ssim=float(ssim),
                rmse=rmse, data_range=scale, rmse_units=units)


def segmentation_metrics(prediction, truth, spacing):
    validate_pair(prediction, truth)
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    left, right = int(prediction.sum()), int(truth.sum())
    if not left and not right:
        return dict(dsc=math.nan, assd=math.nan, jc=math.nan)
    intersection = int((prediction & truth).sum())
    distance = math.inf
    if left and right:
        structure = generate_binary_structure(prediction.ndim, 1)
        a = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
        b = truth ^ binary_erosion(truth, structure=structure, border_value=0)
        distance = .5 * (distance_transform_edt(~b, sampling=spacing)[a].mean()
                          + distance_transform_edt(~a, sampling=spacing)[b].mean())
    return dict(dsc=200. * intersection / (left + right), assd=float(distance),
                jc=intersection / (left + right - intersection))


def average(values):
    data = np.asarray(values, dtype=float)
    data = data[~np.isnan(data)]
    return float(data.mean()) if len(data) else math.nan


def summarize(rows, keys, metrics, ddof):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    result = []
    for values, samples in sorted(groups.items(), key=lambda x: str(x[0])):
        row = dict(zip(keys, values), n=len(samples))
        for metric in metrics:
            data = np.asarray([s.get(metric, math.nan) for s in samples], dtype=float)
            data = data[~np.isnan(data)]
            row.update({metric + "_mean": average(data), metric + "_std": float(data.std(ddof=ddof)) if len(data) > ddof and np.isfinite(data).all() else math.nan,
                        metric + "_n": len(data)})
        result.append(row)
    return result


def roi_mask(value, shape, args):
    if value.startswith("["):
        bounds = json.loads(value)
        if len(bounds) != 4 or any(not isinstance(v, int) for v in bounds):
            raise ValueError("ROI bounds must be four integers.")
        y0, y1, x0, x1 = bounds
        if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
            raise ValueError("ROI rectangles use [y0,y1,x0,x1] with exclusive upper bounds.")
        result = np.zeros(shape, dtype=bool)
        result[y0:y1, x0:x1] = True
    else:
        result = load_array(resolved(value, args.data_root))[0] > 0
    if result.shape != shape or not result.any():
        raise ValueError("ROI masks must be nonempty and match the exported image grid.")
    return result


def cnr(image, tissue, background):
    contrast = abs(float(image[tissue].mean() - image[background].mean()))
    noise = float(image[background].std())
    return contrast / noise if noise else math.inf if contrast else math.nan


def mat_samples(args, refs):
    records, seen = [], set()
    root = Path(args.results)
    for path in [root] if root.is_file() else sorted(root.rglob("*.mat")):
        data = loadmat(path, simplify_cells=True)
        if "output" not in data:
            continue
        item = json.loads(data["metadata"]) if "metadata" in data else {k: data[k] for k in ("client", "data_client", "model_client", "case_id", "source_path") if k in data}
        client = int(item.get("client", item.get("data_client", -1)))
        if client not in args.clients:
            continue
        privacy = item.get("task") == "T7" or {"run", "mode"} <= item.keys()
        if (args.task == "T4" and privacy) or (args.task == "T7" and not privacy):
            continue
        if args.task in {"T1", "T6"} and str(data.get("units", "HU")).upper() != "HU":
            raise ValueError(f"CT reconstruction exports must use HU: {path}")
        item["client"] = client
        identity = (client, str(item.get("case_id", "")), str(item.get("source_path", item.get("source", item.get("path", "")))), item.get("slice", ""), item.get("model_client", ""), item.get("run", ""), item.get("mode", ""))
        if identity in seen:
            raise ValueError(f"Duplicate exported sample: {path}")
        seen.add(identity)
        row = dict(method=args.method, client=client, case_id=item.get("case_id", ""), result=str(path.resolve()),
                   source=identity[2], slice=item.get("slice", ""))
        prediction_key = args.prediction_key if args.task in {"T1", "T2", "T6"} else "output"
        if prediction_key not in data:
            raise ValueError(f"Missing prediction field {prediction_key}: {path}")
        prediction = np.asarray(data[prediction_key])
        if prediction.ndim != 2:
            raise ValueError(f"Expected an exported 2D result, got {prediction.shape}: {path}")
        if args.task == "T1_denoising":
            entry = refs.find(item)
            tissue, background = [roi_mask(entry[k], prediction.shape, args) for k in ("tissue_roi", "background_roi")]
            if np.any(tissue & background) or not np.isfinite(prediction).all():
                raise ValueError("Tissue/background ROIs must be disjoint and predictions finite.")
            row.update(model_client=int(item["model_client"]), data_client=client, cnr=cnr(prediction, tissue, background),
                       input_cnr=cnr(np.asarray(data["low_dose"]), tissue, background))
        elif args.task in {"T1", "T2", "T6"}:
            key = "target" if args.task == "T2" else "ndct"
            if key in data:
                truth, source = np.asarray(data[key]), "embedded " + key
            else:
                truth, _, _, source = reference(refs.find(item), item, prediction, args)
            row.update(reference=source, **reconstruction_metrics(prediction, truth, args.task != "T2", args))
        else:
            entry = refs.find(item)
            truth, spacing, _, source = reference(entry, item, prediction, args)
            validate_pair(prediction, truth)
            if not np.isin(prediction, (0, 1)).all() or not np.isin(truth, (0, 1, 2, 255)).all():
                raise ValueError("Prostate predictions must be binary; references use 0/1, 0/255, or 0/1/2 zones.")
            spacing = args.spacing or spacing
            units = "mm" if spacing is not None else "pixel"
            spacing = checked_spacing(spacing, prediction.ndim)
            row.update(reference=source, distance_unit=units, **segmentation_metrics(prediction, truth > 0, spacing))
            if args.task == "T7":
                row.update(run=int(item["run"]), mode=item["mode"])
        records.append(row)
    return records


def checked_spacing(spacing, dimensions):
    value = tuple(spacing) if spacing is not None else (1.,) * dimensions
    if len(value) != dimensions or not np.isfinite(value).all() or min(value) <= 0:
        raise ValueError("Spacing must provide one positive value per array axis.")
    return value


def ct_segmentation(args, refs):
    rows, seen = [], set()
    suffix = "*.nii.gz" if args.space == "native" else "*.npz"
    for path in sorted(Path(args.results).rglob(suffix)):
        sidecar = Path(str(path)[:-7] + ".json") if args.space == "native" else path.with_suffix(".json")
        if not sidecar.exists():
            continue
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        if "label_map" not in info or int(info["client"]) not in args.clients:
            continue
        client = int(info["client"])
        identity = (client, info["case_id"], info["source_path"])
        if identity in seen:
            raise ValueError(f"Duplicate segmentation case: {identity}")
        seen.add(identity)
        prediction, spacing, affine = load_array(str(path), "segmentation" if args.space == "preprocessed" else "")
        if prediction.ndim != 3:
            raise ValueError(f"T3 requires exported 3D segmentation volumes: {path}")
        truth, reference_spacing, reference_affine, source = reference(refs.find(info), info, prediction, args)
        validate_pair(prediction, truth)
        if affine is not None and reference_affine is not None and not np.allclose(affine, reference_affine, atol=1e-4):
            raise ValueError("Native prediction and reference affines differ; supply registered labels on the same grid.")
        if affine is not None:
            directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
            if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
                raise ValueError("ASSD requires an orthogonal voxel grid; resample sheared native grids before evaluation.")
        spacing = checked_spacing(args.spacing or spacing or reference_spacing, prediction.ndim)
        if reference_spacing is not None and not np.allclose(spacing, reference_spacing, atol=1e-4):
            raise ValueError("Prediction/reference voxel spacing differs.")
        labels = [int(info["label_map"][organ]) for organ in ORGANS]
        entry = refs.find(info)
        truth_labels = list(numbers(entry["label_ids"])) if entry.get("label_ids") else args.label_ids or labels
        if len(truth_labels) != 6 or len(set(truth_labels)) != 6 or min(truth_labels) < 1 or any(v != int(v) for v in truth_labels):
            raise ValueError("CT reference label IDs must identify six distinct foreground classes.")
        if not np.isin(prediction, [0, *labels]).all() or not np.equal(truth, np.rint(truth)).all() or (truth < 0).any():
            raise ValueError("Unexpected prediction labels or noninteger CT reference labels.")
        scores = {organ: segmentation_metrics(prediction == left, truth == right, spacing) for organ, left, right in zip(ORGANS, labels, truth_labels)}
        scores["kidneys"] = {metric: average([scores["kidney_left"][metric], scores["kidney_right"][metric]]) for metric in ("dsc", "assd", "jc")}
        scores["mean"] = {metric: average([scores[organ][metric] for organ in ("kidneys", *ORGANS[2:])]) for metric in ("dsc", "assd", "jc")}
        for organ, metrics in scores.items():
            rows.append(dict(method=args.method, client=client, case_id=info["case_id"], result=str(path.resolve()),
                             reference=source, organ=organ, distance_unit="mm", **metrics))
    return rows


def cell_id(value):
    value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {name + "s": i for i, name in enumerate(CELLS)} | {name: i for i, name in enumerate(CELLS)}
    aliases.update(ig=5, immature_granulocytes=5)
    return aliases[value] if value in aliases else int(value)


def classification(args, refs):
    root = Path(args.results)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("client*.csv") if p.parent.name == "results")
    rows, groups, seen = [], defaultdict(list), set()
    for path in files:
        for item in read_csv(path):
            client, fold = int(item["client"]), int(item["fold"])
            if client not in args.clients:
                continue
            identifier = item["image_id"]
            if identifier in seen or fold not in range(5):
                raise ValueError(f"Repeated original image or invalid fold: {identifier}")
            seen.add(identifier)
            pair = (2 * (client - 21), 2 * (client - 21) + 1)
            target = item.get("global_target", "")
            if target == "":
                entry = refs.find(item)
                target = entry.get("global_target", "") or entry.get("label", "")
            target = cell_id(target)
            probabilities = np.asarray([float(item["probability_0"]), float(item["probability_1"])])
            if target not in pair or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-5):
                raise ValueError(f"Invalid cell labels or probabilities: {identifier}")
            if item.get("local_target", "") != "" and int(item["local_target"]) != pair.index(target):
                raise ValueError(f"Local/global reference class mapping differs: {identifier}")
            predicted = int(item["predicted_local"])
            if predicted != int(probabilities.argmax()) or int(item["predicted_global"]) != pair[predicted]:
                raise ValueError(f"Prediction/probability/class mapping differs: {identifier}")
            if item.get("predicted_name", "") and item["predicted_name"] != CELLS[pair[predicted]]:
                raise ValueError(f"Predicted cell name differs from its global class: {identifier}")
            row = dict(method=args.method, client=client, fold=fold, image_id=identifier, target=pair.index(target),
                       predicted=predicted, probability_1=probabilities[1], positive_class=CELLS[pair[1]])
            rows.append(row)
            groups[client, fold].append(row)
    metrics, curves, interpolated = [], [], defaultdict(list)
    grid = np.linspace(0, 1, 1001)
    for (client, fold), items in sorted(groups.items()):
        target, prediction, probability = [np.asarray([item[k] for item in items]) for k in ("target", "predicted", "probability_1")]
        score = math.nan
        if len(np.unique(target)) == 2:
            fpr, tpr, thresholds = roc_curve(target, probability, pos_label=1, drop_intermediate=False)
            score = float(roc_auc_score(target, probability))
            curves.extend(dict(client=client, fold=fold, fpr=x, tpr=y, threshold=t) for x, y, t in zip(fpr, tpr, thresholds))
            values = np.interp(grid, fpr, tpr)
            values[0], values[-1] = 0., 1.
            interpolated[client].append(values)
        metrics.append(dict(method=args.method, client=client, fold=fold, images=len(items), acc=100. * float((target == prediction).mean()), auc=score))
    if not args.allow_partial and any({fold for c, fold in groups if c == client} != set(range(5)) for client in {c for c, _ in groups}):
        raise ValueError("T5 evaluation requires all five test folds; use --allow-partial for an explicit subset.")
    mean_curves = []
    for client, values in interpolated.items():
        mean = np.mean(values, axis=0)
        deviation = np.std(values, axis=0, ddof=args.ddof) if len(values) > args.ddof else np.full_like(grid, np.nan)
        mean_curves.extend(dict(client=client, fpr=x, tpr_mean=y, tpr_std=s) for x, y, s in zip(grid, mean, deviation))
    write_csv(Path(args.output) / "fold_metrics.csv", metrics)
    write_csv(Path(args.output) / "roc.csv", curves)
    write_csv(Path(args.output) / "roc_mean.csv", mean_curves)
    summary = summarize(metrics, ("method", "client"), ("acc", "auc"), args.ddof)
    for row in summary:
        values = interpolated.get(row["client"], [])
        row["mean_roc_auc"] = float(auc(grid, np.mean(values, axis=0))) if values else math.nan
    return rows, summary


def ratings(args):
    rows, seen = [], set()
    for item in read_csv(args.ratings):
        identity = (item["method"], item.get("client", ""), item["image_id"], item["reader_id"])
        if identity in seen:
            raise ValueError(f"Repeated reader/image rating: {identity}")
        seen.add(identity)
        row = dict(method=item["method"], client=item.get("client", ""), image_id=item["image_id"], reader_id=item["reader_id"])
        for metric in ("noise_reduction", "structure_preservation", "overall_quality"):
            value = float(item[metric])
            if not 1 <= value <= 5:
                raise ValueError("Subjective ratings must lie between 1 and 5.")
            row[metric] = value
        rows.append(row)
    return rows, summarize(rows, ("method",), ("noise_reduction", "structure_preservation", "overall_quality"), args.ddof)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate exported CUDIRCL predictions independently of training/testing")
    parser.add_argument("task", choices=(*CLIENTS, "ratings"))
    parser.add_argument("--results", default="", help="Export directory or a single MAT/CSV result")
    parser.add_argument("--references", default="", help="CSV: client,case_id,reference; source/image identifies slices; CT references use HU; original mask manifests accepted")
    parser.add_argument("--rois", default="", help="CNR CSV: client,source_path,tissue_roi,background_roi; rectangles are JSON [y0,y1,x0,x1] or mask paths")
    parser.add_argument("--ratings", default="", help="CSV: method,reader_id,image_id,noise_reduction,structure_preservation,overall_quality; optional client")
    parser.add_argument("--data-root", default=".", help="Base directory for reference, source and ROI paths in CSV files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clients", nargs="+", type=int)
    parser.add_argument("--method", default="CUDIRCL")
    parser.add_argument("--reference-key", default="", help="Optional array key for external MAT/NPZ/HDF5 references")
    parser.add_argument("--prediction-key", default="output", help="Reconstruction field to evaluate; use fbp or initial for input baselines")
    parser.add_argument("--space", choices=("native", "preprocessed"), default="native", help="T3 exported volume space")
    parser.add_argument("--label-ids", nargs=6, type=int, help="T3 reference IDs: left kidney,right kidney,liver,spleen,pancreas,stomach; default exported IDs")
    parser.add_argument("--spacing", nargs="+", type=float, help="Physical mm spacing in array-axis order; 2D PNG masks otherwise use pixel units")
    parser.add_argument("--mri-scale", choices=("reference_max", "source"), default="reference_max", help="T2 RMSE: shared reference-maximum normalization or exported source units")
    parser.add_argument("--data-range", type=float, help="T2 shared reference intensity range; default reference maximum")
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow-partial", action="store_true", help="Allow a subset of T5 folds or T7 runs/modes")
    args = parser.parse_args(argv)
    if (args.task == "ratings" and not args.ratings) or (args.task != "ratings" and not args.results):
        parser.error("Provide --ratings for ratings, or --results for a task.")
    if args.task == "T1_denoising" and not args.rois:
        parser.error("Real CT denoising needs --rois; low-dose images are not normal-dose references.")
    if args.task in {"T3", "T4", "T7"} and not args.references:
        parser.error("Segmentation evaluation needs --references with true labels.")
    if args.data_range is not None and (not math.isfinite(args.data_range) or args.data_range <= 0):
        parser.error("data-range must be positive and finite.")
    args.clients = args.clients or list(CLIENTS.get(args.task, ()))
    if args.task in CLIENTS and not set(args.clients) <= set(CLIENTS[args.task]):
        parser.error("Selected clients do not belong to this task.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    refs = References(args.rois if args.task == "T1_denoising" else args.references, args.data_root)
    if args.task == "T5":
        rows, summary = classification(args, refs)
    elif args.task == "ratings":
        rows, summary = ratings(args)
    else:
        rows = ct_segmentation(args, refs) if args.task == "T3" else mat_samples(args, refs)
        keys = ("method", "client")
        if args.task == "T1_denoising":
            keys, metrics = ("method", "data_client", "model_client"), ("cnr", "input_cnr")
        elif args.task in {"T1", "T2", "T6"}:
            keys, metrics = (*keys, "rmse_units"), ("psnr", "ssim", "rmse")
        else:
            keys, metrics = (*keys, "distance_unit"), ("dsc", "assd", "jc")
        if args.task == "T3":
            keys = (*keys, "organ")
        if args.task == "T7":
            cohorts = defaultdict(dict)
            for row in rows:
                key = (row["mode"], row["run"])
                if key[0] not in {"dp", "no_dp"} or key[1] not in range(5):
                    raise ValueError("T7 modes are dp/no_dp and run IDs are 0 through 4.")
                cohorts[row["client"]].setdefault(key, set()).add((row["case_id"], row["source"]))
            for client, samples in cohorts.items():
                if not args.allow_partial and set(samples) != {(mode, run) for mode in ("dp", "no_dp") for run in range(5)}:
                    raise ValueError("T7 requires both modes and five repetitions; use --allow-partial for a subset.")
                if any(items != next(iter(samples.values())) for items in samples.values()):
                    raise ValueError(f"T7 test cohorts differ across runs/modes for client {client}.")
            per_run = summarize(rows, (*keys, "mode", "run"), metrics, args.ddof)
            write_csv(output / "run_metrics.csv", per_run)
            means = [{**row, **{metric: row[metric + "_mean"] for metric in metrics}} for row in per_run]
            summary = summarize(means, (*keys, "mode"), metrics, args.ddof)
        else:
            summary = summarize(rows, keys, metrics, args.ddof)
        if args.task in {"T1", "T2", "T6"}:
            client_means = [{**row, **{metric: row[metric + "_mean"] for metric in metrics}} for row in summary]
            write_csv(output / "client_average.csv", summarize(client_means, ("method", "rmse_units"), metrics, args.ddof))
        if args.task == "T1_denoising":
            matrix = [{"data_client": client, **{f"model{row['model_client']}": row["cnr_mean"] for row in summary if row["data_client"] == client}} for client in sorted({r["data_client"] for r in summary})]
            write_csv(output / "cnr_matrix.csv", matrix)
    if not rows:
        raise ValueError("No compatible exported test predictions or ratings found.")
    write_csv(output / "samples.csv", rows)
    write_csv(output / "summary.csv", summary)
    protocol = dict(arguments=vars(args), ct_scale="(HU + 1000)/3000 without clipping or per-image min-max; RMSE in HU; full exported image",
                    ssim="11x11 Gaussian, sigma=1.5, population covariance", dsc_units="percent", jc_units="fraction", assd="mean of two directed surface-distance means, connectivity=1",
                    nifti_units="meter/micron converted to mm; unspecified NIfTI units assumed mm", roc_averaging="equal fold weights; linear interpolation on 1001 FPR points",
                    empty_masks="both empty: undefined; one empty: DSC/JC=0, ASSD=infinity", mri_scaling="prediction and reference share the same reference-derived scale",
                    classification="local class 1 is positive; ACC/AUC summarized across folds", t7="summarized across independent per-run means; no privacy budget calculation")
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} records; saved metrics to {output.resolve()}")


if __name__ == "__main__":
    main()
