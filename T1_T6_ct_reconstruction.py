import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


HU_MIN, HU_MAX = -1024.0, 1500.0
SINO_MIN, SINO_MAX = -0.1, 6.9
EVAL_HU_MIN, EVAL_HU_MAX = -1000.0, 2000.0


@dataclass(frozen=True)
class Geometry:
    client: int
    views: int
    bins: int
    detector_mm: float
    voxel_mm: float
    dsd: float
    dso: float
    kvp: int
    dose: float
    mode: str = "parallel"
    size: int = 512

    def conditioning(self):

        return np.asarray([self.views / 1000, self.bins / 1000,
                           self.detector_mm, self.voxel_mm, self.dsd / 1000,
                           self.dso / 1000, math.log10(self.dose) / 6,
                           self.kvp / 140], dtype=np.float32)


CLIENT_GEOMETRY = {
    1: (512, 1024, .650, .750, 750.2, 476.8, 120, 1e5),
    2: (512, 1024, .650, .750, 750.2, 476.8, 140, 2e5),
    3: (768, 904, .500, .664, 1085.6, 595.0, 120, 1e5),
    4: (896, 1008, .650, .750, 750.2, 476.8, 120, 1e5),
    5: (896, 1024, .500, .664, 946.7, 538.5, 120, 1e5),
    6: (512, 904, .548, .742, 865.2, 613.0, 120, 1e5),
}


def geometry(client, mode="parallel"):
    return Geometry(client, *CLIENT_GEOMETRY[client], mode=mode)


def to_hu(image):
    return image * (HU_MAX - HU_MIN) + HU_MIN


def source_key(path, root):
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path, root):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"client", "case_id", "split", "path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs client,case_id,split,path columns.")
        rows, groups, files = [], {}, set()
        for item in reader:
            client, case, split = int(item["client"]), item["case_id"].strip(), item["split"]
            source = (Path(root) / item["path"]).resolve()
            if client not in CLIENT_GEOMETRY or not case or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid manifest row: {item}")
            key = (client, case)
            if key in groups and groups[key] != split:
                raise ValueError(f"Case leakage across splits: client {client}, {case}")
            if source in files:
                raise ValueError(f"Duplicate source file in manifest: {source}")
            groups[key] = split
            files.add(source)
            rows.append(dict(client=client, case_id=case, split=split, path=source,
                             source_key=source_key(source, root)))
    return rows


def case_id(path, folder):
    value = loadmat(path, variable_names=["case_id"]).get("case_id")
    if value is not None:
        flat = np.asarray(value).reshape(-1)
        return "".join(str(v) for v in flat) if flat.dtype.kind in "US" else str(flat[0])
    parts = path.relative_to(folder).parts
    if parts[0] in {"train", "val", "test"}:
        parts = parts[1:]
    if len(parts) == 2:
        return parts[0]
    raise ValueError(f"Missing subject ID: {path}. Supply case_id or a case folder.")


def partition_cases(groups, ratios, rng):

    if len(groups) < len(ratios):
        raise ValueError("Not enough cases for nonempty independent splits.")
    cases = list(groups)
    rng.shuffle(cases)
    cases.sort(key=lambda case: len(groups[case]), reverse=True)
    target = np.asarray(ratios) * sum(map(len, groups.values()))
    counts, allocation = np.zeros(len(ratios)), [[] for _ in ratios]
    for index, case in enumerate(cases):
        empty = [i for i, items in enumerate(allocation) if not items]
        if len(cases) - index == len(empty):
            chosen = empty[0]
        else:
            chosen = int(np.argmax(target - counts))
        allocation[chosen].extend(groups[case])
        counts[chosen] += len(groups[case])
    return allocation


def legacy_partition(groups, preset, rng):
    training = {case: items for case, items in groups.items() if preset[case] == "train"}
    held_out = {case: items for case, items in groups.items() if preset[case] == "test"}
    train = [item for items in training.values() for item in items]
    test = [item for items in held_out.values() for item in items]
    candidates = []
    if len(training) >= 2:
        candidates.append([*partition_cases(training, (80 / 85, 5 / 85), rng), test])
    if len(held_out) >= 2:
        candidates.append([train, *partition_cases(held_out, (.25, .75), rng)])
    candidates = [parts for parts in candidates if all(parts)]
    if not candidates:
        raise ValueError("Legacy train/test data need at least three independent cases; supply a final three-way manifest.")
    total = len(train) + len(test)
    return min(candidates, key=lambda parts: np.abs(np.asarray(list(map(len, parts))) / total - [.8, .05, .15]).sum())


def make_manifest(args):
    root, records = Path(args.data_root).resolve(), []
    rng = random.Random(args.seed)
    for client, name in enumerate(args.folders, 1):
        folder = root / name
        if not folder.exists():
            continue
        paths = sorted(folder.rglob("*.mat"))
        if not paths:
            raise ValueError(f"No MAT files in {folder}")
        grouped = defaultdict(list)
        preset = {}
        for path in paths:
            case = case_id(path, folder)
            first = path.relative_to(folder).parts[0]
            split = first if first in {"train", "val", "test"} else "unsplit"
            if case in preset and preset[case] != split:
                raise ValueError(f"Case {case} occurs in multiple source splits in {folder}")
            preset[case] = split
            grouped[case].append((path, case))
        splits = set(preset.values())
        if splits == {"unsplit"}:
            allocation = partition_cases(grouped, (.80, .05, .15), rng)
        elif splits == {"train", "test"}:
            allocation = legacy_partition(grouped, preset, rng)
        elif splits == {"train", "val", "test"}:
            allocation = [[item for case, items in grouped.items() if preset[case] == split for item in items]
                          for split in ("train", "val", "test")]
        else:
            raise ValueError(f"Use unsplit cases, train/test, or train/val/test in {folder}.")
        for split, items in zip(("train", "val", "test"), allocation):
            for path, case in items:
                records.append(dict(client=client, case_id=case, split=split,
                                    path=path.relative_to(root).as_posix()))
        print(f"Client {client}: train/val/test = {[len(items) for items in allocation]}")
    if not records:
        raise ValueError("No client folders found; check --data-root and --folders.")
    target = Path(args.manifest)
    if target.exists():
        raise FileExistsError(f"Preserve existing split manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client", "case_id", "split", "path"])
        writer.writeheader()
        writer.writerows(records)
    read_manifest(target, root)


class CTDataset(Dataset):
    def __init__(self, rows, geo, split):
        self.rows = [row for row in rows if row["client"] == geo.client and row["split"] == split]
        self.geo, self.split = geo, split
        if not self.rows:
            raise ValueError(f"Missing {split} data for client {geo.client}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, geo = self.rows[index], self.geo

        use_reference = self.split == "test" or geo.client != 6
        keys = ["sinogram", "fbp"] + (["ndct"] if use_reference else [])
        mat = loadmat(row["path"], variable_names=keys)

        def array(key, shape):
            if key not in mat:
                raise ValueError(f"Missing {key} in {row['path']}")
            value = np.asarray(mat[key], dtype=np.float32).squeeze()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} in {row['path']}: expected {shape}, got {value.shape}")
            return torch.from_numpy(value.copy()).unsqueeze(0)

        sino = array("sinogram", (geo.views, geo.bins))
        fbp = array("fbp", (geo.size, geo.size))
        present = use_reference and "ndct" in mat
        if self.split != "test" and geo.client != 6 and not present:
            raise ValueError(f"Paired training/validation requires ndct: {row['path']}")
        ndct = array("ndct", (geo.size, geo.size)) if present else torch.zeros_like(fbp)
        return dict(sinogram=(sino - SINO_MIN) / (SINO_MAX - SINO_MIN),
                    fbp=(fbp - HU_MIN) / (HU_MAX - HU_MIN),
                    ndct=(ndct - HU_MIN) / (HU_MAX - HU_MIN), has_reference=present,
                    parameters=torch.from_numpy(geo.conditioning()),
                    path=str(row["path"]), source_key=row["source_key"], case_id=row["case_id"])


class ResidualBlock(nn.Module):

    def __init__(self, channels, sinogram=False):
        super().__init__()
        kernel, padding = ((1, 3), (0, 1)) if sinogram else (3, 1)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, padding=padding),
            nn.GroupNorm(1, channels, affine=False), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(channels, channels, kernel, padding=padding),
            nn.GroupNorm(1, channels, affine=False), nn.LeakyReLU(.2, inplace=True))
        self.sinogram = sinogram
        self.attention = (nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(channels, max(1, channels // 16), 1), nn.ReLU(),
                          nn.Conv2d(max(1, channels // 16), channels, 1), nn.Sigmoid())
                          if sinogram else nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()))

    def forward(self, image):
        features = self.body(image)
        attention_input = features if self.sinogram else torch.cat(
            (features.mean(1, keepdim=True), features.amax(1, keepdim=True)), 1)
        return image + features * self.attention(attention_input)


class SinoNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        channels = 4 * width
        self.network = nn.Sequential(
            nn.Conv2d(1, width, (1, 3), padding=(0, 1)),
            nn.GroupNorm(1, width, affine=False), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(width, channels, (1, 3), padding=(0, 1)),
            nn.GroupNorm(1, channels, affine=False), nn.LeakyReLU(.2, inplace=True),
            *[ResidualBlock(channels, True) for _ in range(9)],
            nn.Conv2d(channels, width, (1, 3), padding=(0, 1)),
            nn.Conv2d(width, 1, (1, 3), padding=(0, 1)))

    def forward(self, sinogram):
        return self.network(sinogram)


class BackProjection(nn.Module):

    def __init__(self, geo, chunk_views=16):
        super().__init__()
        self.geo, self.chunk_views = geo, chunk_views
        axis = (torch.arange(geo.size) - (geo.size - 1) / 2) * geo.voxel_mm
        yy, xx = torch.meshgrid(-axis, axis, indexing="ij")
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)
        self.gain = nn.Parameter(torch.ones(1, 1, geo.size, geo.size))

    def forward(self, sinogram):
        geo, batch = self.geo, sinogram.shape[0]
        if sinogram.shape[1:] != (1, geo.views, geo.bins):
            raise ValueError("Sinogram shape disagrees with configured acquisition geometry.")
        sino = sinogram * (SINO_MAX - SINO_MIN) + SINO_MIN
        detector = (torch.arange(geo.bins, device=sino.device, dtype=sino.dtype)
                    - (geo.bins - 1) / 2) * geo.detector_mm
        if geo.mode == "fanflat":
            sino = sino * geo.dsd / torch.sqrt(geo.dsd ** 2 + detector ** 2)
        nfft = 1 << (2 * geo.bins - 1).bit_length()
        response = sino.new_zeros(nfft)
        response[0] = .25
        odd = torch.arange(1, nfft, 2, device=sino.device)
        offsets = torch.where(odd < nfft // 2, odd, odd - nfft).to(sino.dtype)
        response[odd] = -1 / (math.pi * offsets).square()
        ramp = 2 * torch.fft.rfft(response).real / geo.detector_mm
        filtered = torch.fft.irfft(torch.fft.rfft(sino, n=nfft) * ramp, n=nfft)[..., :geo.bins]
        angle_range = math.pi if geo.mode == "parallel" else 2 * math.pi
        angles = torch.arange(geo.views, device=sino.device, dtype=sino.dtype) * angle_range / geo.views
        result = sino.new_zeros(batch, 1, geo.size, geo.size)
        for start in range(0, geo.views, self.chunk_views):
            theta = angles[start:start + self.chunk_views, None, None]
            u = self.xx * theta.cos() + self.yy * theta.sin()
            weight = 1
            if geo.mode == "fanflat":
                distance = geo.dso - self.xx * theta.sin() + self.yy * theta.cos()
                if torch.any(distance <= 0):
                    raise ValueError("Image FOV reaches beyond the source circle.")
                u = geo.dsd * u / distance
                weight = geo.dso * geo.dsd / distance.square()
            grid_x = 2 * u / ((geo.bins - 1) * geo.detector_mm)
            grid = torch.stack((grid_x, torch.zeros_like(grid_x)), -1)
            count = theta.shape[0]
            grid = grid.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(batch * count, geo.size, geo.size, 2)
            lines = filtered[:, 0, start:start + count].reshape(batch * count, 1, 1, geo.bins)
            samples = F.grid_sample(lines, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            samples = samples.reshape(batch, count, geo.size, geo.size)
            result = result + (samples * weight).sum(1, keepdim=True)
        redundancy = 2 if geo.mode == "parallel" else 4
        return result * (angle_range / (redundancy * geo.views)) * self.gain


class ImageNet(nn.Module):
    def __init__(self, width=16, proxy=False):
        super().__init__()
        channels = 4 * width
        self.start = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1),
                                   nn.GroupNorm(1, channels, affine=False), nn.LeakyReLU(.2, inplace=True))
        self.blocks = nn.ModuleList([nn.Sequential(*[ResidualBlock(channels)
                                   for _ in range(3 if proxy else 2)]) for _ in range(4)])
        self.end = nn.Sequential(nn.Conv2d(channels, width, 1), nn.Conv2d(width, 1, 1))
        self.film = (nn.Sequential(nn.Linear(8, 2 * channels), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(2 * channels, 4 * channels), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(4 * channels, 8 * channels)) if proxy else None)

    def forward(self, image, parameters=None):
        features = self.start(image)
        if self.film is not None:
            gamma, beta = self.film(parameters).chunk(2, 1)
            gamma = gamma.reshape(image.shape[0], 4, -1, 1, 1)
            beta = beta.reshape_as(gamma)
        for index, block in enumerate(self.blocks):
            features = block(features)
            if self.film is not None:
                features = gamma[:, index] * features + beta[:, index]
        return self.end(features)


class CTModel(nn.Module):

    def __init__(self, geo, width=16, chunk_views=16):
        super().__init__()
        self.sinogram = SinoNet(width)
        self.backprojection = BackProjection(geo, chunk_views)
        self.image = ImageNet(width)

    def forward(self, sinogram):
        private_features = self.backprojection(self.sinogram(sinogram))
        return private_features, self.image(private_features)


def ssim_per_image(prediction, target):

    size = min(11, prediction.shape[-2], prediction.shape[-1])
    size -= 1 - size % 2
    axis = torch.arange(size, device=prediction.device, dtype=prediction.dtype) - (size - 1) / 2
    gaussian = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    mu_x, mu_y = F.conv2d(prediction, window), F.conv2d(target, window)
    var_x = F.conv2d(prediction.square(), window) - mu_x.square()
    var_y = F.conv2d(target.square(), window) - mu_y.square()
    covariance = F.conv2d(prediction * target, window) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + .01 ** 2) * (2 * covariance + .03 ** 2)
             / ((mu_x.square() + mu_y.square() + .01 ** 2) * (var_x + var_y + .03 ** 2)))
    return score.flatten(1).mean(1)


def task_loss(prediction, target, paired):
    fidelity = F.mse_loss(prediction, target)
    if paired:
        return fidelity + .1 * (1 - ssim_per_image(prediction, target).mean())

    tv = ((prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean()
          + (prediction[..., :, 1:] - prediction[..., :, :-1]).abs().mean())
    return fidelity + .3 * tv


def train_batch(model, proxy, local_optimizer, proxy_optimizer, batch, device, paired):
    sinogram = batch["sinogram"].to(device)
    target = batch["ndct" if paired else "fbp"].to(device)
    parameters = batch["parameters"].to(device)
    local_optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    private, local = model(sinogram)
    shared = proxy(private.detach(), parameters)
    local_loss = task_loss(local, target, paired) + F.huber_loss(local, shared.detach(), delta=.1)
    proxy_loss = task_loss(shared, target, paired) + F.huber_loss(shared, local.detach(), delta=.1)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite training loss; check data units and geometry.")
    local_loss.backward()
    proxy_loss.backward()
    local_optimizer.step()
    proxy_optimizer.step()
    return local_loss.item(), proxy_loss.item()


@torch.no_grad()
def validate(model, loader, device, paired):
    model.eval()
    totals, count = defaultdict(float), 0
    for batch in loader:
        _, prediction = model(batch["sinogram"].to(device))
        target = batch["ndct" if paired else "fbp"].to(device)
        number = prediction.shape[0]
        totals["loss"] += task_loss(prediction, target, paired).item() * number
        if paired:

            span = EVAL_HU_MAX - EVAL_HU_MIN
            pred = (to_hu(prediction) - EVAL_HU_MIN) / span
            truth = (to_hu(target) - EVAL_HU_MIN) / span
            mse = (pred - truth).square().flatten(1).mean(1)
            psnr = -10 * mse.clamp_min(1e-12).log10()
            ssim = ssim_per_image(pred, truth)
            score = 100 * (1 - ((45 - psnr).clamp_min(0) / 45 + (1 - ssim).abs() + mse) / 3)
            score = torch.where((psnr > 0) & (ssim > 0) & (mse < 1), score.clamp(0, 100), 0)
            for key, values in (("psnr", psnr), ("ssim", ssim), ("rmse_hu", mse.sqrt() * span), ("score", score)):
                totals[key] += values.sum().item()
        count += number
    result = {key: value / count for key, value in totals.items()}
    if not paired:
        result["score"] = -result["loss"]
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite validation result.")
    return result


def checkpoint_load(path):

    return torch.load(path, map_location="cpu", weights_only=True)


def model_config(args):
    return dict(geometry=args.geometry, width=args.width, chunk_views=args.chunk_views,
                backprojection_filter="astra_ram_lak_v1")


def loader_for(rows, geo, split, args, round_number=0):
    generator = torch.Generator().manual_seed(args.seed + 10000 * round_number + geo.client)
    return DataLoader(CTDataset(rows, geo, split), batch_size=args.batch_size,
                      shuffle=split == "train", num_workers=args.workers,
                      pin_memory=str(args.device).startswith("cuda"), generator=generator)


def manifest_signature(rows):

    records = sorted((r["client"], r["case_id"], r["split"], r["source_key"]) for r in rows)
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def client_signature(rows, client):
    return manifest_signature([row for row in rows if row["client"] == client])


def train(args, rows):
    clients = list(range(1, 5 if args.task == "T1" else 7))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output already contains a run. Use --resume or a new --output.")
    device = torch.device(args.device)
    models = {client: CTModel(geometry(client, args.geometry), args.width, args.chunk_views).to(device) for client in clients}
    proxy = ImageNet(args.width, proxy=True).to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}

    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in clients}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                  for client, opt in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                        for client, opt in proxy_optimizers.items()}
    training = {client: loader_for(rows, geometry(client, args.geometry), "train", args) for client in clients}
    validation = {client: loader_for(rows, geometry(client, args.geometry), "val", args) for client in clients}
    best, retired, order, first_round = {client: -math.inf for client in clients}, set(), [1, 2, 3, 4], 1
    random.Random(args.seed).shuffle(order)
    source = args.resume or args.initialize_from
    if source:
        state = checkpoint_load(source)
        if state["model_config"] != model_config(args):
            raise ValueError("Checkpoint model configuration differs; loading is strict.")
        for client, weights in state["models"].items():
            if client in models:
                models[client].load_state_dict(weights, strict=True)
        proxy.load_state_dict(state["proxy"], strict=True)
        if args.resume:
            if state["args"]["task"] != args.task or state["manifest_signature"] != manifest_signature(rows):
                raise ValueError("Resume requires the same task and data split.")
            if args.join_round is None:
                args.join_round = state["args"]["join_round"]
            controlled = ("seed", "batch_size", "join_round", "local_epochs", "extra_epochs",
                          "fixed_order", "no_withdraw", "score_target", "lr", "lr_patience", "max_batches")
            if any(state["args"][key] != getattr(args, key) for key in controlled):
                raise ValueError("Resume training settings differ from the saved run.")
            for client in clients:
                optimizers[client].load_state_dict(state["optimizers"][client])
                schedulers[client].load_state_dict(state["schedulers"][client])
                proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
                proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
            best, retired, order = state["best"], set(state["retired"]), state["order"]
            for client in clients:
                if math.isfinite(best[client]) and not (output / "checkpoints" / f"client{client}_best.pt").exists():
                    raise FileNotFoundError("Resume in the run directory containing its best checkpoints.")
            first_round = state["round"] + 1
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            if args.task != "T6" or state["args"]["task"] != "T1":
                raise ValueError("--initialize-from is for initializing T6 from a T1 last.pt checkpoint.")
            for client in range(1, 5):
                previous = checkpoint_load(Path(source).parent / "checkpoints" / f"client{client}_best.pt")
                if previous["split_signature"] != client_signature(rows, client):
                    raise ValueError("T6 must preserve the original T1 subject/split assignments.")
                best[client] = state["best"][client]
                torch.save(previous, output / "checkpoints" / f"client{client}_best.pt")
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), geometry={c: asdict(geometry(c, args.geometry)) for c in clients}), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        joined = clients if args.task == "T6" and round_number >= args.join_round else [1, 2, 3, 4]
        arrivals = [client for client in joined if client not in retired and client not in order]
        order = [client for client in order if client in joined and client not in retired]
        order = arrivals + order
        extra_client = order[-1] if order and round_number > 1 and not args.fixed_order else None
        losses = {}
        print(f"Round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()

            training[client].generator.manual_seed(args.seed + 10000 * round_number + client)
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            total = np.zeros(2)
            samples = 0
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    values = train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client],
                                         batch, device, paired=client != 6)
                    number = batch["sinogram"].shape[0]
                    total += np.asarray(values) * number
                    samples += number
            losses[client] = dict(local=float(total[0] / samples), proxy=float(total[1] / samples))
        scores = {}
        for client in joined:
            result = validate(models[client], validation[client], device, paired=client != 6)
            scores[client] = result
            if client in order:
                schedulers[client].step(result["loss"])
                proxy_schedulers[client].step(result["loss"])
            if result["score"] > best[client]:
                best[client] = result["score"]
                torch.save(dict(client=client, task=args.task, round=round_number,
                                split_signature=client_signature(rows, client), model_config=model_config(args),
                                model=models[client].state_dict()), output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation loss={result['loss']:.6f}, score={result['score']:.4f}")
        if not args.fixed_order:
            if not args.no_withdraw:
                retired.update(client for client in joined if client != 6 and scores[client]["score"] >= args.score_target)

            order = sorted((client for client in joined if client not in retired),
                           key=lambda client: (-scores[client]["score"], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, training=losses, validation=scores,
                                         next_order=order, retired=sorted(retired))) + "\n")
        state = dict(args=vars(args), model_config=model_config(args), round=round_number,
                     manifest_signature=manifest_signature(rows), models={c: m.state_dict() for c, m in models.items()},
                     proxy=proxy.state_dict(), optimizers={c: opt.state_dict() for c, opt in optimizers.items()},
                     schedulers={c: sch.state_dict() for c, sch in schedulers.items()},
                     proxy_optimizers={c: opt.state_dict() for c, opt in proxy_optimizers.items()},
                     proxy_schedulers={c: sch.state_dict() for c, sch in proxy_schedulers.items()},
                     best=best, retired=sorted(retired), order=order, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        torch.save(state, output / "last.pt")
        if len(joined) == len(clients) and all(client in retired for client in clients):
            break


def safe_name(value):
    clean = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)[:100]
    return clean or "case"


@torch.inference_mode()
def test(args, rows):

    clients = args.clients or list(range(1, 5 if args.task == "T1" else 7))
    if args.task == "T1" and any(client > 4 for client in clients):
        raise ValueError("Clients 5/6 are external clients in T6.")
    output, device = Path(args.output), torch.device(args.device)
    for client in clients:
        state = checkpoint_load(output / "checkpoints" / f"client{client}_best.pt")
        config = state["model_config"]
        if config.get("backprojection_filter") != "astra_ram_lak_v1":
            raise ValueError("This checkpoint predates the ASTRA-aligned Ram-Lak filter; retrain with the corrected operator.")
        geo = geometry(client, config["geometry"])
        model = CTModel(geo, config["width"], config["chunk_views"]).to(device)
        if state["client"] != client:
            raise ValueError("Checkpoint client ID mismatch.")
        if state["split_signature"] != client_signature(rows, client):
            raise ValueError("Test manifest differs from the training run's client split.")
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        loader = loader_for(rows, geo, "test", args)
        count = 0
        for batch in loader:
            _, prediction = model(batch["sinogram"].to(device))
            prediction = to_hu(prediction).cpu().numpy()
            for index in range(len(prediction)):
                source = Path(batch["path"][index])
                case = batch["case_id"][index]
                token = hashlib.sha256(batch["source_key"][index].encode()).hexdigest()[:10]
                folder = output / "results" / f"client{client}" / safe_name(case)
                folder.mkdir(parents=True, exist_ok=True)
                data = dict(output=prediction[index, 0], fbp=to_hu(batch["fbp"][index, 0]).numpy(),
                            sinogram=(batch["sinogram"][index, 0] * (SINO_MAX - SINO_MIN) + SINO_MIN).numpy(),
                            case_id=case, source_name=source.name, source_path=batch["source_key"][index], client=client,
                            checkpoint_round=state["round"], checkpoint_task=state["task"],
                            units="HU", geometry=asdict(geo))
                if bool(batch["has_reference"][index]):
                    data["ndct"] = to_hu(batch["ndct"][index, 0]).numpy()
                savemat(folder / f"{safe_name(source.stem)}_{token}.mat", data, do_compression=True)
                count += 1
        print(f"Client {client}: exported {count} reconstructions to {output / 'results' / f'client{client}'}")
        del model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDIRCL T1/T6 dual-domain CT reconstruction")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--task", choices=("T1", "T6"), default="T1")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="ct_run")
    parser.add_argument("--folders", nargs=6, default=[f"client{i}" for i in range(1, 7)])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--geometry", choices=("parallel", "fanflat"), default="parallel")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--chunk-views", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--join-round", type=int, default=None, help="T6: default 2 from scratch, or 1 when initialized from T1")
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="0 uses all training data; never limits validation/test")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--score-target", type=float, default=98.5)
    parser.add_argument("--fixed-order", action="store_true", help="Disable adaptive reordering/extra epochs/withdrawal")
    parser.add_argument("--no-withdraw", action="store_true")
    parser.add_argument("--resume", help="Resume this task from last.pt, including all optimizer/control states")
    parser.add_argument("--initialize-from", help="Initialize T6 from a T1 last.pt; reset optimizer/control states")
    parser.add_argument("--clients", nargs="+", type=int, choices=range(1, 7), help="Test only; default all clients of the task")
    args = parser.parse_args(argv)
    if args.join_round is None and not args.resume:
        args.join_round = 1 if args.initialize_from else 2
    if min(args.width, args.chunk_views, args.batch_size, args.rounds, args.local_epochs) < 1:
        parser.error("Width, chunk size, batch size, rounds, and local epochs must be positive.")
    if min(args.workers, args.extra_epochs, args.max_batches, args.lr_patience) < 0 or args.lr <= 0:
        parser.error("Counts must be nonnegative and learning rate positive.")
    if args.resume and args.initialize_from:
        parser.error("Choose --resume or --initialize-from.")
    if args.initialize_from and (args.command != "train" or args.task != "T6"):
        parser.error("--initialize-from requires T6 training.")
    minimum_join = 1 if args.initialize_from or args.resume else 2
    if args.command == "train" and args.task == "T6" and args.join_round is not None and not minimum_join <= args.join_round <= args.rounds:
        parser.error(f"T6 requires {minimum_join} <= --join-round <= --rounds.")
    if not 0 <= args.score_target <= 100 or (args.clients and len(set(args.clients)) != len(args.clients)):
        parser.error("score-target must be in [0,100]; test clients must be unique.")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.command == "split":
        make_manifest(args)
    else:
        rows = read_manifest(args.manifest, args.data_root)
        (train if args.command == "train" else test)(args, rows)


if __name__ == "__main__":
    main()
