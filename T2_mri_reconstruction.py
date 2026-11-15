import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import fastmri
import h5py
import nibabel as nib
import numpy as np
import torch
from fastmri.models.unet import Unet
from scipy.io import loadmat, savemat
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLIENTS = {7: "fastMRI", 8: "M4Raw", 9: "Calgary-Campinas", 10: "BraTS"}
KINDS = {"kspace", "calgary", "coil_images", "image", "measured"}


def fft2c(x):
    return torch.view_as_complex(fastmri.fft2c(torch.view_as_real(x.contiguous())).contiguous())


def ifft2c(x):
    return torch.view_as_complex(fastmri.ifft2c(torch.view_as_real(x.contiguous())).contiguous())


def rss(x):
    return torch.linalg.vector_norm(x, dim=-3)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value, root):
    return str((Path(root) / value).resolve()) if value else ""


@lru_cache(maxsize=4)
def cached_arrays(path, modified, size):
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if path.lower().endswith(".npy"):
        return {"image": np.load(path, allow_pickle=False)}
    return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}


def packed_arrays(path):
    status = Path(path).stat()
    return cached_arrays(path, status.st_mtime_ns, status.st_size)


def is_hdf5(path):
    return path.lower().endswith((".h5", ".hdf5")) or h5py.is_hdf5(path)


def is_nifti(path):
    return path.lower().endswith((".nii", ".nii.gz"))


def array_keys(path):
    if is_nifti(path):
        return ["image"]
    if is_hdf5(path):
        with h5py.File(path, "r") as data:
            return list(data.keys())
    return list(packed_arrays(path))


def array_shape(path, key):
    if is_nifti(path):
        return nib.load(path).shape
    if is_hdf5(path):
        with h5py.File(path, "r") as data:
            return data[key].shape
    return packed_arrays(path)[key].shape


def array_read(path, key, selection=None):
    if is_nifti(path):
        data = nib.load(path).dataobj
        return np.asarray(data if selection is None else data[selection])
    if is_hdf5(path):
        with h5py.File(path, "r") as data:
            return np.asarray(data[key][()] if selection is None else data[key][selection])
    data = packed_arrays(path)[key]
    return np.asarray(data if selection is None else data[selection])


def describe(row):
    path, kind = row["path"], row.get("kind", "") or "kspace"
    if kind not in KINDS:
        raise ValueError(f"Unknown data kind: {kind}")
    if is_nifti(path) and kind != "image":
        raise ValueError("NIfTI magnitude volumes require kind=image (one virtual coil).")
    keys = array_keys(path)
    key = row.get("key", "") or next((k for k in (kind, "kspace", "coil_images", "image") if k in keys), "")
    if not key:
        raise ValueError(f"Specify the image/k-space key: {path}")
    shape = array_shape(path, key)
    layout = row.get("layout", "")
    if not layout:
        if is_nifti(path):
            layout = "HWS"
        elif kind == "calgary":
            layout = "SHWI"
        elif kind == "image":
            layout = {2: "HW", 3: "SHW"}.get(len(shape), "")
        else:
            layout = {2: "HW", 3: "CHW", 4: "SCHW"}.get(len(shape), "")
    if len(layout) != len(shape) or len(set(layout)) != len(layout) or not {"H", "W"} <= set(layout):
        raise ValueError(f"Invalid layout {layout} for {shape}: {path}")
    if set(layout) - set("SCHWRI") or ("R" in layout and (layout[-1] != "R" or shape[-1] != 2)):
        raise ValueError("Layouts use S,C,H,W and optionally trailing R=real/imag or I=interleaved coils.")
    if kind == "calgary" and (layout != "SHWI" or shape[-1] % 2):
        raise ValueError("Calgary requires SHWI with interleaved real/imaginary coil channels.")
    if kind != "calgary" and "I" in layout:
        raise ValueError("Interleaved I channels are supported only for kind=calgary.")
    slices = shape[layout.index("S")] if "S" in layout else 1
    start, stop = int(row.get("start", "") or 0), int(row.get("stop", "") or slices)
    if not 0 <= start < stop <= slices:
        raise ValueError(f"Invalid slice interval {start}:{stop} for {path}")
    crop = [int(row.get(f"crop_{axis}", "") or 0) for axis in ("h", "w")]
    if any(crop) and min(crop) < 1:
        raise ValueError("Set both crop_h and crop_w, or neither.")
    if not any(crop) and kind != "measured" and "reconstruction_rss" in keys:
        crop = list(array_shape(path, "reconstruction_rss")[-2:])
    if kind == "measured" and any(crop):
        raise ValueError("Measured k-space must retain its acquisition grid; do not crop/re-FFT it.")
    if is_hdf5(path):
        with h5py.File(path, "r") as data:
            contrast = str(data.attrs.get("acquisition", data.attrs.get("contrast", ""))).upper()
            if any(word in contrast for word in ("T2", "FLAIR", "T1GD", "T1CE")):
                raise ValueError(f"T2 uses T1 data only: {path}, contrast={contrast}")
    return dict(row, kind=kind, key=key, layout=layout, slices=slices, start=start, stop=stop, crop=crop)


def read_manifest(path, root, index=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"client", "case_id", "path"} | (set() if index else {"split"})
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows, groups, occupied = [], {}, defaultdict(list)
        for item in reader:
            item = {k: (v or "").strip() for k, v in item.items() if k}
            client, case, split = int(item["client"]), item["case_id"], item.get("split", "")
            if client not in CLIENTS or not case or (not index and split not in {"train", "val", "test"}):
                raise ValueError(f"Invalid client/case_id/split: {item}")
            if item.get("contrast", "T1").upper() not in {"", "T1"}:
                raise ValueError("T2 requires T1 contrast.")
            item.update(client=client, path=resolve_path(item["path"], root),
                        mask_path=resolve_path(item.get("mask_path", ""), root))
            row = describe(item)
            if row["kind"] == "measured" and (index or split != "test"):
                raise ValueError("Measured-only data without a full reference are supported for test only.")
            group = (client, case)
            if not index and group in groups and groups[group] != split:
                raise ValueError(f"Subject leakage across splits: {group}")
            groups[group] = split
            acquisition = (row["path"], row["key"])
            for previous in occupied[acquisition]:
                if max(row["start"], previous["start"]) < min(row["stop"], previous["stop"]):
                    raise ValueError(f"Overlapping/duplicate acquisition slices: {acquisition}")
                if (client, case) != (previous["client"], previous["case_id"]):
                    raise ValueError("All disjoint slice intervals of one acquisition need the same subject ID.")
            occupied[acquisition].append(row)
            rows.append(row)
    if not rows:
        raise ValueError("The data manifest is empty.")
    return rows


def make_split(args):
    rows = read_manifest(args.index, args.data_root, index=True)
    rng = random.Random(args.seed)
    for client in sorted({row["client"] for row in rows}):
        groups = defaultdict(list)
        for row in rows:
            if row["client"] == client:
                groups[row["case_id"]].append(row)
        if len(groups) < 3:
            raise ValueError("At least three subjects per client are needed for train/val/test splitting.")
        subjects = sorted(groups)
        rng.shuffle(subjects)
        subjects.sort(key=lambda case: -sum(r["stop"] - r["start"] for r in groups[case]))
        target = np.asarray([.8, .05, .15]) * sum(r["stop"] - r["start"] for r in rows if r["client"] == client)
        used, counts = np.zeros(3), np.zeros(3, dtype=int)
        for number, case in enumerate(subjects):
            empty = np.where(counts == 0)[0]
            fold = int(empty[0] if len(subjects) - number == len(empty) else np.argmax(target - used))
            for row in groups[case]:
                row["split"] = ("train", "val", "test")[fold]
            used[fold] += sum(r["stop"] - r["start"] for r in groups[case])
            counts[fold] += 1
    fields = ["client", "case_id", "split", "path", "kind", "key", "layout", "start", "stop", "crop_h", "crop_w", "mask_path"]
    destination = Path(args.manifest)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def slice_array(row, index):
    layout = row["layout"]
    selection = tuple(index if axis == "S" else slice(None) for axis in layout)
    array = array_read(row["path"], row["key"], selection)
    layout = layout.replace("S", "")
    if "R" in layout:
        array = array[..., 0] + 1j * array[..., 1]
        layout = layout[:-1]
    if row["kind"] == "calgary":
        array = (array[..., ::2] + 1j * array[..., 1::2]).transpose(2, 0, 1)
        image = np.fft.ifftshift(np.fft.ifft2(array, norm="ortho"), axes=(-2, -1))
        array = np.fft.fftshift(np.fft.fft2(image, norm="ortho"), axes=(-2, -1))
    else:
        if "C" not in layout:
            array = array[None]
            layout = "C" + layout
        array = array.transpose(*(layout.index(axis) for axis in "CHW"))
    if row["kind"] in {"kspace", "calgary", "measured"} and not np.iscomplexobj(array):
        raise ValueError("K-space must be complex; specify an explicit trailing R layout for real/imaginary data.")
    if row["kind"] == "image" and array.shape[0] != 1:
        raise ValueError("Magnitude-image simulation uses one virtual coil.")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite input: {row['path']}, slice {index}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.complex64))


def mask_array(path, key, index, slices, height, width):
    data = array_read(path, key)
    shape = data.shape
    data = np.squeeze(data)
    slice_lines = len(shape) > 2 and shape[0] == slices and math.prod(shape[1:]) == width
    if slice_lines and data.shape == (slices, width):
        data = data[index]
    elif data.ndim == 3 and data.shape == (slices, height, width):
        data = data[index]
    elif data.ndim == 2 and data.shape == (slices, width) and slices != height:
        data = data[index]
    if data.shape == (width,):
        data = np.broadcast_to(data[None], (height, width))
    if data.shape != (height, width):
        raise ValueError(f"Mask {path}:{key} has {data.shape}; expected ({height},{width}) or a column mask.")
    if not np.isin(data, (0, 1)).all():
        raise ValueError("A mask must contain binary 0/1 values on a Cartesian k-space grid.")
    return torch.from_numpy(np.array(data, dtype=bool, copy=True))


def infer_acs(mask):
    height, width = mask.shape
    y, x = height // 2, width // 2
    acs = torch.zeros_like(mask)
    if torch.equal(mask, mask[:1].expand_as(mask)):
        left, right = x, x + 1
        while left > 0 and mask[0, left - 1]:
            left -= 1
        while right < width and mask[0, right]:
            right += 1
        acs[:, left:right] = True
    else:
        radius = 0
        while y - radius >= 0 and y + radius < height and x - radius >= 0 and x + radius < width:
            if not mask[y - radius:y + radius + 1, x - radius:x + radius + 1].all():
                break
            radius += 1
        if radius:
            acs[y - radius + 1:y + radius, x - radius + 1:x + radius] = True
    if not acs.any() or (acs & ~mask).any():
        raise ValueError("No fully sampled centered ACS region; supply an explicit acs_mask with the official mask.")
    return acs


def load_masks(row, index, shape, global_path, nominal):
    height, width = shape
    inline = "mask" in array_keys(row["path"])
    source = row.get("mask_path") or global_path or (row["path"] if inline else "")
    if not source:
        raise ValueError("Supply the official R=10 mask in the data, mask_path column, or --mask-file.")
    keys = array_keys(source)
    key = "mask" if "mask" in keys else "image" if len(keys) == 1 and keys[0] == "image" else ""
    if not key:
        raise ValueError(f"Mask file requires a mask array: {source}")
    mask = mask_array(source, key, index, row["slices"], height, width)
    if inline and row["kind"] == "measured" and source != row["path"]:
        if not torch.equal(mask, mask_array(row["path"], "mask", index, row["slices"], height, width)):
            raise ValueError("External and measured-data acquisition masks differ.")
    units = width if torch.equal(mask, mask[:1].expand_as(mask)) else height * width
    samples = mask.float().mean().item() * units
    if abs(samples - units / nominal) > max(1.01, .05 * units / nominal):
        raise ValueError(f"Loaded mask does not match nominal R={nominal}: actual R={mask.numel()/max(mask.sum().item(),1):.4f}")
    acs = mask_array(source, "acs_mask", index, row["slices"], height, width) if "acs_mask" in keys else infer_acs(mask)
    if not acs.any() or (acs & ~mask).any():
        raise ValueError("ACS must be nonempty and a subset of the acquisition mask.")
    return mask, acs, source


class MRIDataset(Dataset):
    def __init__(self, rows, client, split, mask_file="", acceleration=10):
        self.items = [(row, index) for row in rows if row["client"] == client and row["split"] == split
                      for index in range(row["start"], row["stop"])]
        self.mask_file, self.acceleration = mask_file, acceleration
        if not self.items:
            raise ValueError(f"No {split} slices for client {client}.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        row, index = self.items[item]
        array = slice_array(row, index)
        original = list(array.shape[-2:])
        crop = row["crop"] if any(row["crop"]) else original
        if min(crop) < 8 or any(a > b for a, b in zip(crop, original)):
            raise ValueError(f"Invalid reconstruction crop {crop} for {original}.")
        offset = [(size - length) // 2 for size, length in zip(original, crop)]
        if row["kind"] != "measured":
            image = array if row["kind"] in {"image", "coil_images"} else ifft2c(array)
            image = image[:, offset[0]:offset[0] + crop[0], offset[1]:offset[1] + crop[1]]
            full = fft2c(image)
            target = rss(image)
        else:
            full, target = array, None
        mask, acs, source = load_masks(row, index, crop, self.mask_file, self.acceleration)
        measured = torch.where(mask, full, 0)
        initial = rss(ifft2c(measured))
        maximum = initial.amax()
        scale = maximum if maximum > 1e-12 else maximum.new_ones(())
        actual_r = mask.numel() / mask.sum().item()
        condition = torch.zeros(8)
        condition[row["client"] - 7] = 1
        condition[4:] = torch.tensor([actual_r / 10, measured.shape[0] / 32, crop[0] / 512, crop[1] / 512])
        metadata = dict(client=row["client"], dataset=CLIENTS[row["client"]], case_id=row["case_id"],
                        path=row["path"], key=row["key"], layout=row["layout"], kind=row["kind"], slice=index,
                        coil_count=measured.shape[0], virtual_single_coil=row["kind"] == "image",
                        nominal_acceleration=self.acceleration, actual_acceleration=actual_r,
                        mask_source=source, original_shape=original, crop_shape=crop, crop_offset=offset,
                        units="source arbitrary MRI units", fft="centered orthonormal",
                        mask_protocol="author-supplied; trajectory provenance is not inferred")
        if is_nifti(row["path"]):
            metadata["affine"] = nib.load(row["path"]).affine.tolist()
        result = dict(measured=measured / scale, mask=mask[None], acs=acs[None],
                      condition=condition, scale=scale, metadata=json.dumps(metadata))
        if target is not None:
            result["target"] = (target / scale)[None]
        return result


def group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            setattr(module, name, nn.GroupNorm(1, child.num_features, affine=False))
        else:
            group_norm(child)


class ComplexUnet(nn.Module):
    def __init__(self, width=18, pools=4, conditioned=False):
        super().__init__()
        self.unet = Unet(2, 2, chans=width, num_pool_layers=pools, drop_prob=0)
        group_norm(self.unet)
        self.multiple = 2 ** pools
        channels = [block.out_chans for block in self.unet.down_sample_layers]
        channels += [self.unet.conv.out_chans]
        channels += [block.out_chans if hasattr(block, "out_chans") else block[0].out_chans for block in self.unet.up_conv]
        self.channels = channels
        self.modulation = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 2 * sum(channels))) if conditioned else None

    def forward(self, x, condition=None):
        value = torch.cat((x.real, x.imag), dim=1)
        mean = value.mean((-2, -1), keepdim=True)
        std = value.std((-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        value = (value - mean) / std
        height, width = value.shape[-2:]
        ph, pw = (-height) % self.multiple, (-width) % self.multiple
        value = F.pad(value, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        films = iter(self.modulation(condition).split([2 * c for c in self.channels], 1)) if self.modulation else None

        def modulate(tensor):
            if films is None:
                return tensor
            gamma, beta = next(films).chunk(2, 1)
            return tensor * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        skips = []
        for block in self.unet.down_sample_layers:
            value = modulate(block(value))
            skips.append(value)
            value = F.avg_pool2d(value, 2)
        value = modulate(self.unet.conv(value))
        for up, block in zip(self.unet.up_transpose_conv, self.unet.up_conv):
            value = torch.cat((up(value), skips.pop()), dim=1)
            value = block[1](modulate(block[0](value))) if isinstance(block, nn.Sequential) else modulate(block(value))
        value = value[:, :, ph // 2:ph // 2 + height, pw // 2:pw // 2 + width] * std + mean
        return torch.complex(value[:, :1].float(), value[:, 1:].float())


class SensitivityModel(nn.Module):
    def __init__(self, width=8, pools=4):
        super().__init__()
        self.regularizer = ComplexUnet(width, pools)

    def forward(self, measured, acs):
        batch, coils, height, width = measured.shape
        if coils == 1:
            return torch.ones_like(measured)
        image = ifft2c(torch.where(acs, measured, 0)).reshape(batch * coils, 1, height, width)
        sensitivity = self.regularizer(image).reshape(batch, coils, height, width)
        return sensitivity / sensitivity.abs().square().sum(1, keepdim=True).clamp_min(1e-12).sqrt()


class Cascade(nn.Module):
    def __init__(self, width, pools, conditioned=False):
        super().__init__()
        self.regularizer = ComplexUnet(width, pools, conditioned)
        self.dc_weight = nn.Parameter(torch.ones(()))

    def forward(self, current, measured, mask, sensitivity, condition=None):
        image = (ifft2c(current) * sensitivity.conj()).sum(1, keepdim=True)
        update = fft2c(self.regularizer(image, condition) * sensitivity)
        return current - self.dc_weight * torch.where(mask, current - measured, 0) - update


class ReconstructionModel(nn.Module):
    def __init__(self, width=18, pools=4, sens_width=8, sens_pools=4):
        super().__init__()
        self.sensitivity = SensitivityModel(sens_width, sens_pools)
        self.private = nn.ModuleList(Cascade(width, pools) for _ in range(2))
        self.sharing = nn.ModuleList(Cascade(width, pools) for _ in range(2))

    def forward(self, measured, mask, acs):
        sensitivity = self.sensitivity(measured, acs)
        current = measured
        for cascade in self.private:
            current = cascade(current, measured, mask, sensitivity)
        feature = current
        for cascade in self.sharing:
            current = cascade(current, measured, mask, sensitivity)
        return rss(ifft2c(current))[:, None], feature, sensitivity


class Proxy(nn.Module):
    def __init__(self, width=18, pools=4):
        super().__init__()
        self.cascades = nn.ModuleList(Cascade(width, pools, conditioned=True) for _ in range(2))

    def forward(self, feature, measured, mask, sensitivity, condition):
        current, sensitivity = feature.detach(), sensitivity.detach()
        for cascade in self.cascades:
            current = cascade(current, measured, mask, sensitivity, condition)
        return rss(ifft2c(current))[:, None]


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = fastmri.SSIMLoss(win_size=7)

    def forward(self, output, target):
        data_range = target.flatten(1).amax(1).clamp_min(1e-6)
        return F.mse_loss(output, target) + .1 * self.ssim(output, target, data_range)


def train_batch(model, proxy, optimizer, proxy_optimizer, batch, device, criterion):
    measured, mask, acs, condition = [batch[key].to(device) for key in ("measured", "mask", "acs", "condition")]
    target = batch["target"].to(device)
    optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    local, feature, sensitivity = model(measured, mask, acs)
    shared = proxy(feature, measured, mask, sensitivity, condition)
    local_loss = criterion(local, target) + F.huber_loss(local, shared.detach(), delta=.1)
    proxy_loss = criterion(shared, target) + F.huber_loss(shared, local.detach(), delta=.1)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite training loss.")
    local_loss.backward()
    proxy_loss.backward()
    optimizer.step()
    proxy_optimizer.step()
    return local_loss.item(), proxy_loss.item()


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total, count = defaultdict(float), 0
    for batch in loader:
        output, _, _ = model(*(batch[key].to(device) for key in ("measured", "mask", "acs")))
        target = batch["target"].to(device)
        for prediction, truth in zip(output.split(1), target.split(1)):
            span = truth.amax().clamp_min(1e-6)
            mse = ((prediction - truth) / span).square().mean()
            psnr = -10 * mse.clamp_min(1e-12).log10()
            ssim = 1 - criterion.ssim(prediction, truth, span[None])
            score = 100 * (1 - ((45 - psnr).clamp_min(0) / 45 + (1 - ssim).abs() + mse) / 3)
            score = score.clamp(0, 100) if psnr > 0 and ssim > 0 and mse < 1 else score.new_zeros(())
            total["loss"] += criterion(prediction, truth).item()
            total["score"] += score.item()
            count += 1
    result = {key: value / count for key, value in total.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite validation result.")
    return result


def singleton_batch(samples):
    if len(samples) != 1:
        raise ValueError("MRI uses batch size 1 to preserve variable coil counts and acquisition grids.")
    return {key: value[None] if isinstance(value, torch.Tensor) else [value] for key, value in samples[0].items()}


def loader_for(rows, client, split, args):
    generator = torch.Generator().manual_seed(args.seed + client)
    return DataLoader(MRIDataset(rows, client, split, args.mask_file, args.acceleration), batch_size=1,
                      shuffle=split == "train", num_workers=args.workers, collate_fn=singleton_batch,
                      pin_memory=str(args.device).startswith("cuda"), generator=generator)


def data_signature(rows, mask_file, acceleration):
    records = [{key: value for key, value in row.items() if key not in {"slices"}} for row in rows]
    masks = sorted({row["mask_path"] for row in rows if row["mask_path"]} | ({mask_file} if mask_file else set()))
    inline = {}
    for path in sorted({row["path"] for row in rows}):
        keys = array_keys(path)
        values = [np.ascontiguousarray(array_read(path, key)).tobytes() for key in ("mask", "acs_mask") if key in keys]
        if values:
            inline[path] = hashlib.sha256(b"".join(values)).hexdigest()
    content = dict(rows=sorted(records, key=lambda r: (r["client"], r["case_id"], r["path"], r["start"])),
                   masks={path: file_hash(path) for path in masks}, inline_masks=inline, acceleration=acceleration)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def model_config(args):
    return {key: getattr(args, key) for key in ("width", "pools", "sens_width", "sens_pools")}


def checkpoint_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def train(args, rows):
    clients = sorted(CLIENTS)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output directory.")
    device = torch.device(args.device)
    models = {client: ReconstructionModel(**model_config(args)).to(device) for client in clients}
    proxy = Proxy(args.width, args.pools).to(device)
    criterion = ReconstructionLoss().to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in clients}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                  for client, opt in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                        for client, opt in proxy_optimizers.items()}
    training = {client: loader_for(rows, client, "train", args) for client in clients}
    validation = {client: loader_for(rows, client, "val", args) for client in clients}
    signature = data_signature(rows, args.mask_file, args.acceleration)
    best, retired, order, first_round = {client: -math.inf for client in clients}, set(), clients.copy(), 1
    random.Random(args.seed).shuffle(order)
    if args.resume:
        state = checkpoint_load(args.resume)
        controlled = ("seed", "local_epochs", "extra_epochs", "max_batches", "lr", "lr_patience",
                      "fixed_order", "no_withdraw", "score_target", "acceleration")
        if state["model_config"] != model_config(args) or state["data_signature"] != signature:
            raise ValueError("Resume requires the same model, manifest and external mask contents.")
        if any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume settings differ from the saved run.")
        proxy.load_state_dict(state["proxy"], strict=True)
        for client in clients:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
        best, retired, order, first_round = state["best"], set(state["retired"]), state["order"], state["round"] + 1
        if any(math.isfinite(best[c]) and not (output / "checkpoints" / f"client{c}_best.pt").exists() for c in clients):
            raise FileNotFoundError("Resume in the run directory containing its best checkpoints.")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), clients=CLIENTS, data_signature=signature), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        extra_client = order[-1] if order and round_number > 1 and not args.fixed_order else None
        losses, scores = {}, {}
        print(f"Round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()
            training[client].generator.manual_seed(args.seed + 10000 * round_number + client)
            totals, count = np.zeros(2), 0
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    totals += train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client], batch, device, criterion)
                    count += 1
            losses[client] = dict(local=float(totals[0] / count), proxy=float(totals[1] / count))
        for client in clients:
            result = validate(models[client], validation[client], device, criterion)
            scores[client] = result
            if client in order:
                schedulers[client].step(result["loss"])
                proxy_schedulers[client].step(result["loss"])
            if result["score"] > best[client]:
                best[client] = result["score"]
                torch.save(dict(client=client, task="T2", round=round_number, model_config=model_config(args),
                                acceleration=args.acceleration, data_signature=signature,
                                subject_splits={row["case_id"]: row["split"] for row in rows if row["client"] == client},
                                source_splits={row["path"]: row["split"] for row in rows if row["client"] == client},
                                model=models[client].state_dict()),
                           output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation loss={result['loss']:.6f}, score={result['score']:.4f}")
        if not args.fixed_order:
            if not args.no_withdraw:
                retired.update(client for client in clients if scores[client]["score"] >= args.score_target)
            order = sorted((client for client in clients if client not in retired), key=lambda client: (-scores[client]["score"], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, training=losses, validation=scores,
                                         next_order=order, retired=sorted(retired))) + "\n")
        torch.save(dict(args=vars(args), model_config=model_config(args), data_signature=signature, round=round_number,
                        models={c: m.state_dict() for c, m in models.items()}, proxy=proxy.state_dict(),
                        optimizers={c: opt.state_dict() for c, opt in optimizers.items()},
                        proxy_optimizers={c: opt.state_dict() for c, opt in proxy_optimizers.items()},
                        schedulers={c: sch.state_dict() for c, sch in schedulers.items()},
                        proxy_schedulers={c: sch.state_dict() for c, sch in proxy_schedulers.items()},
                        best=best, retired=sorted(retired), order=order, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), output / "last.pt")
        if len(retired) == len(clients):
            break


@torch.no_grad()
def test(args, rows):
    device = torch.device(args.device)
    for client in args.clients:
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output) / "checkpoints" / f"client{client}_best.pt"
        state = checkpoint_load(checkpoint)
        if state.get("task") != "T2" or state["client"] != client:
            raise ValueError("Testing requires a T2 best checkpoint for the requested client.")
        if state["acceleration"] != args.acceleration:
            raise ValueError("Test acceleration differs from training.")
        for row in rows:
            if row["client"] == client and row["split"] == "test":
                if (state.get("subject_splits", {}).get(row["case_id"], "test") != "test"
                        or state.get("source_splits", {}).get(row["path"], "test") != "test"):
                    raise ValueError("Known training/validation subject or acquisition cannot be reassigned to test.")
        model = ReconstructionModel(**state["model_config"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        destination = Path(args.output) / "results" / f"client{client}"
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in loader_for(rows, client, "test", args):
            measured, mask, acs = [batch[key].to(device) for key in ("measured", "mask", "acs")]
            prediction, _, _ = model(measured, mask, acs)
            scale = float(batch["scale"][0])
            metadata = json.loads(batch["metadata"][0])
            metadata.update(checkpoint=str(checkpoint.resolve()), round=state["round"])
            identity = json.dumps([metadata["path"], metadata["key"], metadata["slice"]])
            name = hashlib.sha256(identity.encode()).hexdigest()[:16] + f"_s{metadata['slice']:04d}.mat"
            result = dict(output=prediction[0, 0].cpu().numpy() * scale,
                          initial=rss(ifft2c(measured))[0].cpu().numpy() * scale,
                          mask=batch["mask"][0, 0].numpy(), acs_mask=batch["acs"][0, 0].numpy(),
                          scale=scale, metadata=json.dumps(metadata, ensure_ascii=False))
            if "target" in batch:
                result["target"] = batch["target"][0, 0].numpy() * scale
            savemat(destination / name, result, do_compression=True)
            count += 1
        print(f"Client {client}: exported {count} reconstruction slices to {destination}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDIRCL T2 MRI reconstruction")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", help="Subject-level source CSV without split; repeated acquisitions share case_id")
    parser.add_argument("--mask-file", default="", help="Official binary mask, optionally with acs_mask; no generated default")
    parser.add_argument("--output", default="mri_run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clients", type=int, nargs="+", choices=tuple(CLIENTS), default=list(CLIENTS), help="Clients to export in test")
    parser.add_argument("--checkpoint", default="", help="Single-client best checkpoint for test")
    parser.add_argument("--resume", default="")
    parser.add_argument("--acceleration", type=float, default=10)
    parser.add_argument("--width", type=int, default=18)
    parser.add_argument("--pools", type=int, default=4)
    parser.add_argument("--sens-width", type=int, default=8)
    parser.add_argument("--sens-pools", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=50)
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100, help="Training epoch limit; 0 uses all training slices")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--score-target", type=float, default=98)
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--no-withdraw", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    if args.command == "split" and not args.index:
        parser.error("split requires --index with explicit participant case_id values")
    if args.checkpoint and (args.command != "test" or len(args.clients) != 1):
        parser.error("--checkpoint requires test with one --clients value")
    if args.command == "train" and sorted(args.clients) != list(CLIENTS):
        parser.error("T2 training requires all four clients; --clients selects test exports")
    if len(set(args.clients)) != len(args.clients):
        parser.error("Test clients must be unique")
    if min(args.width, args.pools, args.sens_width, args.sens_pools, args.rounds, args.local_epochs) < 1:
        parser.error("Model sizes, rounds and local epochs must be positive")
    if min(args.workers, args.extra_epochs, args.max_batches, args.lr_patience) < 0 or args.lr <= 0 or args.acceleration <= 1:
        parser.error("Invalid worker count, training budget, learning rate or acceleration")
    if not 0 <= args.score_target <= 100:
        parser.error("score-target must be in [0,100]")
    args.mask_file = resolve_path(args.mask_file, args.data_root)
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.command == "split":
        make_split(args)
    else:
        rows = read_manifest(args.manifest, args.data_root)
        (train if args.command == "train" else test)(args, rows)


if __name__ == "__main__":
    main()
