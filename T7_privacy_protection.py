import argparse
import csv
import hashlib
import json
import math
import random
import re
import secrets
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from opacus.accountants import RDPAccountant
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
from scipy.io import savemat
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLIENTS = {
    15: dict(name="BIDMC", vendor="GE", field_t=3, endorectal=1),
    16: dict(name="HK", vendor="Siemens", field_t=1.5, endorectal=1),
    17: dict(name="I2CVB", vendor="Siemens", field_t=3, endorectal=0),
    18: dict(name="ISBI", vendor="Siemens", field_t=3, endorectal=0),
    19: dict(name="ISBI15", vendor="Philips", field_t=1.5, endorectal=1),
    20: dict(name="MSD", vendor="unspecified", field_t=None, endorectal=None),
}


def condition_vector(client):
    vector = torch.zeros(8)
    vector[client - 15] = 1
    vector[6] = (CLIENTS[client]["field_t"] or 0) / 3
    vector[7] = CLIENTS[client]["endorectal"] or 0
    return vector


def source_path(value, root):
    return str((Path(root) / value).resolve()) if value else ""


def read_manifest(path, root, index=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"client", "case_id", "image"} | (set() if index else {"split"})
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows, groups, files = [], {}, set()
        for item in reader:
            row = {key: (value or "").strip() for key, value in item.items() if key}
            client, case, split = int(row["client"]), row["case_id"], row.get("split", "")
            if client not in CLIENTS or not case or (not index and split not in {"train", "val", "test"}):
                raise ValueError(f"Invalid client/case_id/split: {row}")
            if row.get("contrast", "T2").upper() not in {"", "T2"}:
                raise ValueError("T7 uses T2-weighted images only, without ADC channels.")
            row.update(client=client, image=source_path(row["image"], root),
                       mask=source_path(row.get("mask", ""), root))
            if row["image"] in files:
                raise ValueError(f"Duplicate source image: {row['image']}")
            files.add(row["image"])
            if not Path(row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            if (index or split in {"train", "val"}) and not Path(row["mask"]).is_file():
                raise FileNotFoundError(f"Training, validation and split preparation need a mask: {row['image']}")
            group = (client, case)
            if not index and group in groups and groups[group] != split:
                raise ValueError(f"Subject leakage across splits: {group}")
            groups[group] = split
            rows.append(row)
    if not rows:
        raise ValueError("The data manifest is empty.")
    return rows


def read_image(path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim not in {2, 3}:
        raise ValueError(f"Expected a preprocessed 8-bit grayscale slice: {path}")
    if values.ndim == 3:
        if values.shape[-1] not in {3, 4} or not np.array_equal(values[..., 0], values[..., 1]) or not np.array_equal(values[..., 0], values[..., 2]):
            raise ValueError(f"T7 requires grayscale MRI, not color or ADC data: {path}")
        values = values[..., 0]
    return np.array(values, dtype=np.float32, copy=True) / 255


def read_mask(path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        if values.shape[-1] not in {3, 4} or not np.array_equal(values[..., 0], values[..., 1]) or not np.array_equal(values[..., 0], values[..., 2]):
            raise ValueError(f"Expected a grayscale label mask: {path}")
        values = values[..., 0]
    if values.ndim != 2 or not np.isin(values, (0, 1, 2, 255)).all():
        raise ValueError(f"Mask labels must be 0/1, 0/255, or whole-prostate zones 0/1/2: {path}")
    return np.array(values > 0, dtype=np.int64)


def discover_index(args):
    rows, pattern = [], re.compile(args.case_pattern)
    for client, settings in CLIENTS.items():
        directory = Path(args.data_root) / settings["name"]
        found = sorted({path for folder in ("train", "val", "test")
                        for path in (directory / folder / "images").glob("*")
                        if path.suffix.lower() in {".png", ".tif", ".tiff", ".bmp"}})
        if not found:
            raise FileNotFoundError(f"No slices under {directory}/train|val|test/images; alternatively supply --index.")
        for image in found:
            match = pattern.fullmatch(image.stem)
            if not match or "case" not in match.groupdict():
                raise ValueError(f"Cannot identify subject in {image.name}; supply --case-pattern or an explicit --index CSV.")
            mask = image.parent.parent / "masks" / f"{image.stem}_mask.png"
            rows.append(dict(client=client, case_id=match["case"], image=str(image.resolve()), mask=str(mask.resolve())))
    return rows


def make_split(args):
    rows = read_manifest(args.index, args.data_root, index=True) if args.index else discover_index(args)
    retained = []
    for row in rows:
        mask = read_mask(row["mask"])
        if mask.shape != read_image(row["image"]).shape:
            raise ValueError(f"Image/mask shapes differ: {row['image']}")
        if mask.any():
            retained.append(row)
    rng = random.Random(args.seed)
    for client in CLIENTS:
        groups = defaultdict(list)
        for row in retained:
            if row["client"] == client:
                groups[row["case_id"]].append(row)
        if len(groups) < 3:
            raise ValueError(f"Client {client} needs at least three foreground-containing subjects for independent train/val/test sets.")
        subjects = sorted(groups)
        rng.shuffle(subjects)
        subjects.sort(key=lambda case: -len(groups[case]))
        target = np.asarray([.8, .05, .15]) * sum(map(len, groups.values()))
        used, counts = np.zeros(3), np.zeros(3, dtype=int)
        for index, case in enumerate(subjects):
            empty = np.flatnonzero(counts == 0)
            fold = int(empty[0] if len(subjects) - index == len(empty) else np.argmax(target - used))
            for row in groups[case]:
                row["split"] = ("train", "val", "test")[fold]
            used[fold] += len(groups[case])
            counts[fold] += 1
    destination = Path(args.manifest)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["client", "case_id", "split", "image", "mask"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in retained:
            writer.writerow({key: row[key] for key in fields})
    print(f"Kept {len(retained)} foreground slices; removed {len(rows)-len(retained)} background-only slices.")


class ProstateDataset(Dataset):
    def __init__(self, rows, client, split, size=384):
        self.rows = [row for row in rows if row["client"] == client and row["split"] == split]
        self.split, self.size, self.condition = split, size, condition_vector(client)
        if not self.rows:
            raise ValueError(f"No {split} images for client {client}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = torch.from_numpy(read_image(row["image"]))[None]
        original_shape = list(image.shape[-2:])
        image = F.interpolate(image[None], size=(self.size, self.size), mode="area")[0].repeat(3, 1, 1)
        metadata = dict(client=row["client"], dataset=CLIENTS[row["client"]]["name"], case_id=row["case_id"],
                        source=row["image"], original_shape=original_shape, model_shape=[self.size, self.size],
                        space="source 2D pixel grid", labels={"0": "background", "1": "prostate"}, contrast="T2")
        result = dict(image=image, condition=self.condition.clone(), metadata=json.dumps(metadata))
        if self.split != "test":
            mask = torch.from_numpy(read_mask(row["mask"]))
            if list(mask.shape) != original_shape:
                raise ValueError(f"Image/mask shape mismatch: {row['image']}")
            if not mask.any():
                raise ValueError("T7 uses foreground-containing slices; prepare the manifest with split.")
            result["target"] = F.interpolate(mask[None, None].float(), size=(self.size, self.size), mode="nearest")[0, 0].long()
        return result


class DoubleConv(nn.Module):
    def __init__(self, inputs, outputs, middle=None):
        super().__init__()
        middle = middle or outputs
        self.layers = nn.Sequential(nn.Conv2d(inputs, middle, 3, padding=1), nn.GroupNorm(1, middle), nn.ReLU(inplace=False),
                                    nn.Conv2d(middle, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        channels = [width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.blocks = nn.ModuleList(DoubleConv(inputs, outputs) for inputs, outputs in zip([3] + channels[:-1], channels))

    def forward(self, image):
        features = []
        for index, block in enumerate(self.blocks):
            image = block(F.max_pool2d(image, 2) if index else image)
            features.append(image)
        return features


class Up(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.block = DoubleConv(inputs, outputs, inputs // 2)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.block(torch.cat((skip, x), dim=1))


class Decoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.channels = [4 * width, 2 * width, width, width]
        self.up = nn.ModuleList(Up(inputs, outputs) for inputs, outputs in zip([16 * width, 8 * width, 4 * width, 2 * width], self.channels))
        self.head = nn.Conv2d(width, 2, 1)
        self.mlp = nn.Sequential(nn.Linear(8, 4 * width), nn.ReLU(inplace=False), nn.Dropout(.5),
                                 nn.Linear(4 * width, 8 * width), nn.ReLU(inplace=False), nn.Dropout(.5),
                                 nn.Linear(8 * width, 2 * sum(self.channels)))

    def forward(self, features, condition):
        modulations = self.mlp(condition).split([2 * channels for channels in self.channels], dim=1)
        value = features[-1]
        for block, skip, modulation in zip(self.up, reversed(features[:-1]), modulations):
            gamma, beta = modulation.chunk(2, dim=1)
            value = block(value, skip) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.head(value)


class SegmentationModel(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.encoder, self.decoder = Encoder(width), Decoder(width)

    def forward(self, image, condition):
        features = self.encoder(image)
        return self.decoder(features, condition), features


def task_loss(logits, target):
    foreground = logits.softmax(1)[:, 1].flatten(1)
    truth = target.flatten(1).float()
    dice = (2 * (foreground * truth).sum(1) + 1e-5) / (foreground.sum(1) + truth.sum(1) + 1e-5)
    return F.cross_entropy(logits, target) + 1 - dice.mean()


def kl_distance(student, teacher):
    log_p, log_q = student.log_softmax(1), teacher.detach().log_softmax(1)
    return (log_p.exp() * (log_p - log_q)).sum(1).mean()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss, dice_sum, count = 0., 0., 0
    for batch in loader:
        output, _ = model(batch["image"].to(device), batch["condition"].to(device))
        target = batch["target"].to(device)
        prediction, truth = output.argmax(1).flatten(1), target.flatten(1)
        denominator = prediction.sum(1) + truth.sum(1)
        dice = torch.where(denominator > 0, 2 * (prediction * truth).sum(1) / denominator.clamp_min(1), 1.)
        number = len(target)
        loss += task_loss(output, target).item() * number
        dice_sum += dice.sum().item()
        count += number
    result = dict(loss=loss / count, score=100 * dice_sum / count)
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite validation result.")
    return result


def loader_for(rows, client, split, args, size=None):
    return DataLoader(ProstateDataset(rows, client, split, size or args.size), batch_size=args.batch_size if split != "test" else 1,
                      shuffle=split == "train", num_workers=args.workers, drop_last=False,
                      pin_memory=str(args.device).startswith("cuda"), generator=torch.Generator().manual_seed(args.seed + client))


def manifest_signature(rows):
    records = sorted((row["client"], row["case_id"], row["split"], row["image"], row["mask"]) for row in rows)
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def model_config(args):
    return dict(width=args.width, size=args.size)


def checkpoint_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


ACCOUNTING_SCOPE = ("Configured sampled-Gaussian RDP monitor only. Client-local shuffling, non-DP teacher/features "
                    "and validation-driven scheduling are not certified. No end-to-end or patient-level DP bound.")


def noise_generator(device):
    return torch.Generator(device=device).manual_seed(secrets.randbits(63))


def make_proxy(args, device, dp, sample_rate):
    raw = Decoder(args.width).to(device)
    generator = noise_generator(device) if dp else None
    accountant = RDPAccountant() if dp else None
    if dp:
        ModuleValidator.validate(raw, strict=True)
    proxy = GradSampleModule(raw, batch_first=True, loss_reduction="mean", strict=True) if dp else raw
    optimizers = {}
    for client in CLIENTS:
        optimizer = torch.optim.Adam(raw.parameters(), lr=args.lr)
        if dp:
            optimizer = DPOptimizer(optimizer, noise_multiplier=args.noise_multiplier, max_grad_norm=args.clip_norm,
                                    expected_batch_size=args.batch_size, loss_reduction="mean", generator=generator,
                                    secure_mode=False)
            optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
        optimizers[client] = optimizer
    return raw, proxy, optimizers, accountant, generator


def train_batch(model, proxy, optimizer, proxy_optimizer, batch, device, physical_batch_size):
    number = len(batch["image"])
    dp = isinstance(proxy_optimizer, DPOptimizer)
    optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    if dp:
        proxy_optimizer.expected_batch_size = number
    totals = np.zeros(2)
    for start in range(0, number, physical_batch_size):
        stop = min(start + physical_batch_size, number)
        fraction = (stop - start) / number
        image, condition, target = [batch[key][start:stop].to(device) for key in ("image", "condition", "target")]
        local, features = model(image, condition)
        shared = proxy([feature.detach() for feature in features], condition)
        local_loss = task_loss(local, target) + kl_distance(local, shared)
        proxy_loss = task_loss(shared, target) + kl_distance(shared, local)
        if not torch.isfinite(local_loss + proxy_loss):
            raise FloatingPointError("Nonfinite T7 training loss.")
        (fraction * local_loss).backward()
        (proxy_loss if dp else fraction * proxy_loss).backward()
        if dp:
            proxy_optimizer.signal_skip_step(stop < number)
            proxy_optimizer.step()
            proxy_optimizer.zero_grad(set_to_none=True)
        totals += fraction * np.asarray([local_loss.item(), proxy_loss.item()])
    optimizer.step()
    if not dp:
        proxy_optimizer.step()
    return tuple(float(value) for value in totals)


def accounting_report(accountant, args, training_images, updates):
    result = dict(scope=ACCOUNTING_SCOPE, training_images=training_images, configured_q=args.batch_size / training_images,
                  configured_delta=1 / training_images, configured_logical_batch_size=args.batch_size, logical_proxy_updates=updates,
                  sampling="shuffled mini-batches within one client, without replacement per epoch",
                  clipping_unit="per-image proxy gradient" if accountant is not None else None, patient_level_bound=None,
                  certified_end_to_end_epsilon=None, joint_release_of_repeated_runs_certified_epsilon=None,
                  optimizer="Adam", noise_multiplier=args.noise_multiplier if accountant is not None else 0,
                  clip_norm=args.clip_norm if accountant is not None else None,
                  noise_std_on_gradient_sum=args.noise_multiplier * args.clip_norm if accountant is not None else 0,
                  physical_batch_size=args.physical_batch_size, secure_rng=False)
    if accountant is not None:
        epsilon, alpha = accountant.get_privacy_spent(delta=1 / training_images)
        result.update(configured_rdp_epsilon=epsilon, configured_rdp_alpha=alpha,
                      accounted_steps=sum(steps for _, _, steps in accountant.history))
        if result["accounted_steps"] != sum(updates.values()):
            raise RuntimeError("Accountant must count every logical proxy update exactly once.")
    else:
        result.update(configured_rdp_epsilon=None, configured_rdp_alpha=None, accounted_steps=0)
    return result


def train_run(args, rows, run, mode):
    dp = mode == "dp"
    device, output = torch.device(args.device), Path(args.output) / f"run{run}" / mode
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    last = output / "last_private.pt"
    if last.exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output directory; existing experimental runs are preserved.")
    seed = args.seed + 100000 * run
    torch.manual_seed(seed)
    models = {client: SegmentationModel(args.width).to(device) for client in CLIENTS}
    training = {client: loader_for(rows, client, "train", args) for client in CLIENTS}
    validation = {client: loader_for(rows, client, "val", args) for client in CLIENTS}
    training_images = sum(len(loader.dataset) for loader in training.values())
    if training_images < args.batch_size:
        raise ValueError("Total training images must be at least the logical batch size for configured q=B/N.")
    raw_proxy, proxy, proxy_optimizers, accountant, generator = make_proxy(args, device, dp, args.batch_size / training_images)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                  for client, optimizer in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                        for client, optimizer in proxy_optimizers.items()}
    best = {client: dict(score=-math.inf, loss=math.inf) for client in CLIENTS}
    updates, epochs_done = {client: 0 for client in CLIENTS}, {client: 0 for client in CLIENTS}
    order, retired, first_round = list(CLIENTS), set(), 1
    signature = manifest_signature(rows)
    if args.resume and last.exists():
        state = checkpoint_load(last)
        controlled = ("seed", "batch_size", "physical_batch_size", "local_epochs", "extra_epochs", "max_batches",
                      "noise_multiplier", "clip_norm", "lr", "lr_patience", "fixed_order", "no_withdraw", "score_target")
        if state["task"] != "T7" or state["run"] != run or state["mode"] != mode:
            raise ValueError("Resume requires the same T7 run and DP mode.")
        if state["model_config"] != model_config(args) or state["manifest_signature"] != signature:
            raise ValueError("Resume requires the same model and T4 subject/split assignments.")
        if state["training_images"] != training_images or any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume training or accounting settings differ from the saved run.")
        raw_proxy.load_state_dict(state["proxy"], strict=True)
        for client in CLIENTS:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
        if dp:
            accountant.load_state_dict(state["accountant"])
            generator.set_state(state["noise_rng"])
        best, order, retired = state["best"], state["order"], set(state["retired"])
        updates, epochs_done, first_round = state["updates"], state["epochs_done"], state["round"] + 1
        if any(not (output / "checkpoints" / f"client{client}_best_private.pt").exists() for client in CLIENTS):
            raise FileNotFoundError("Resume in the directory containing this run's local best checkpoints.")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config_private.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), clients=CLIENTS, run=run, mode=mode, model_config=model_config(args),
                       manifest_signature=signature, training_images=training_images, scope=ACCOUNTING_SCOPE,
                       checkpoint_scope="local-only: task models, optimizer states, training diagnostics and RNG states"), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        if not order:
            break
        extra_client = order[-1] if round_number > 1 and not args.fixed_order else None
        losses, scores = {}, {}
        print(f"Run {run}, {mode}, round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()
            training[client].generator.manual_seed(seed + 10000 * round_number + client)
            total, count = np.zeros(2), 0
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    values = train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client],
                                         batch, device, args.physical_batch_size)
                    number = len(batch["image"])
                    total += np.asarray(values) * number
                    count += number
                    updates[client] += 1
            epochs_done[client] += epochs
            losses[client] = dict(local=float(total[0] / count), proxy=float(total[1] / count))
        for client in CLIENTS:
            result = validate(models[client], validation[client], device)
            scores[client] = result
            if client in order:
                schedulers[client].step(result["loss"])
                proxy_schedulers[client].step(result["loss"])
            if (result["score"], -result["loss"]) > (best[client]["score"], -best[client]["loss"]):
                best[client] = result
                torch.save(dict(task="T7", run=run, mode=mode, client=client, round=round_number,
                                epochs_done=epochs_done[client], model_config=model_config(args), manifest_signature=signature,
                                test_cases=sorted({row["case_id"] for row in rows if row["client"] == client and row["split"] == "test"}),
                                model=models[client].state_dict(), scope="local task model; not a DP release"),
                           output / "checkpoints" / f"client{client}_best_private.pt")
            print(f"Client {client}: validation loss={result['loss']:.6f}, score={result['score']:.4f}")
        if not args.fixed_order:
            if not args.no_withdraw:
                retired.update(client for client in CLIENTS if scores[client]["score"] >= args.score_target)
            order = sorted((client for client in CLIENTS if client not in retired), key=lambda client: (-scores[client]["score"], client))
        report = accounting_report(accountant, args, training_images, updates)
        with (output / "training_private.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, training=losses, validation=scores,
                                         next_order=order, retired=sorted(retired), accounting=report)) + "\n")
        with (output / "accounting_private.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        torch.save(dict(args=vars(args), task="T7", run=run, mode=mode, round=round_number, model_config=model_config(args),
                        manifest_signature=signature, training_images=training_images,
                        models={c: model.state_dict() for c, model in models.items()}, proxy=raw_proxy.state_dict(),
                        optimizers={c: optimizer.state_dict() for c, optimizer in optimizers.items()},
                        proxy_optimizers={c: optimizer.state_dict() for c, optimizer in proxy_optimizers.items()},
                        schedulers={c: scheduler.state_dict() for c, scheduler in schedulers.items()},
                        proxy_schedulers={c: scheduler.state_dict() for c, scheduler in proxy_schedulers.items()},
                        accountant=accountant.state_dict() if dp else None, noise_rng=generator.get_state() if dp else None,
                        updates=updates, epochs_done=epochs_done, best=best, order=order, retired=sorted(retired),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        scope="private recovery state; do not exchange local models, diagnostics or RNG states"), last)
        torch.save(dict(task="T7", mode=mode, run=run, round=round_number, model_config=model_config(args),
                        proxy=raw_proxy.state_dict(), scope=ACCOUNTING_SCOPE), output / "proxy_only.pt")
    if dp:
        proxy.remove_hooks()


@torch.no_grad()
def test_run(args, rows, run, mode):
    device = torch.device(args.device)
    for client in args.clients:
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output) / f"run{run}" / mode / "checkpoints" / f"client{client}_best_private.pt"
        state = checkpoint_load(checkpoint)
        if state.get("task") != "T7" or state["run"] != run or state["mode"] != mode or state["client"] != client:
            raise ValueError("Testing requires a T7 local best checkpoint for this run, mode and client.")
        if "test_cases" in state:
            if not {row["case_id"] for row in rows if row["client"] == client and row["split"] == "test"} <= set(state["test_cases"]):
                raise ValueError("Test subjects must belong to this checkpoint's held-out cases.")
        elif state["manifest_signature"] != manifest_signature(rows):
            raise ValueError("Legacy checkpoints require the complete original held-out manifest.")
        model = SegmentationModel(state["model_config"]["width"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        destination = Path(args.output) / f"run{run}" / mode / "results" / f"client{client}"
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in loader_for(rows, client, "test", args, state["model_config"]["size"]):
            logits, _ = model(batch["image"].to(device), batch["condition"].to(device))
            metadata = json.loads(batch["metadata"][0])
            probability = F.interpolate(logits.softmax(1), size=metadata["original_shape"], mode="bilinear", align_corners=False)[0].cpu().numpy()
            segmentation = probability.argmax(0).astype(np.uint8)
            metadata.update(task="T7", run=run, mode=mode, checkpoint=str(checkpoint.resolve()), round=state["round"],
                            epochs_done=state["epochs_done"], model_scope="local unperturbed task model; outputs are not a DP release")
            name = hashlib.sha256(metadata["source"].encode()).hexdigest()[:16]
            savemat(destination / f"{name}.mat", dict(output=segmentation, probability=probability,
                                                     metadata=json.dumps(metadata, ensure_ascii=False)), do_compression=True)
            Image.fromarray(segmentation * 255).save(destination / f"{name}.png")
            count += 1
        print(f"Run {run}, {mode}, client {client}: exported {count} segmentation slices.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDIRCL T7 proxy perturbation experiments with nominal RDP monitoring")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True, help="Use the same T4 subject/split manifest for every DP mode and run")
    parser.add_argument("--index", default="")
    parser.add_argument("--case-pattern", default=r"(?P<case>.+)_(?P<slice>\d+)")
    parser.add_argument("--output", default="privacy_run")
    parser.add_argument("--mode", choices=("dp", "no_dp", "both"), default="both")
    parser.add_argument("--runs", type=int, nargs="+", choices=range(5), default=list(range(5)), help="Five independent repetitions using the same subject split")
    parser.add_argument("--clients", type=int, nargs="+", choices=tuple(CLIENTS), default=list(CLIENTS), help="Clients to export in test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=1, help="Physical microbatch; clipping/noise/Adam use one logical batch update")
    parser.add_argument("--noise-multiplier", type=float, default=1.2)
    parser.add_argument("--clip-norm", type=float, default=.5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=20, help="Logical training batches per epoch; 0 traverses all training images")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--score-target", type=float, default=92)
    parser.add_argument("--no-withdraw", action="store_true")
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume private local states separately for every selected run/mode")
    parser.add_argument("--checkpoint", default="", help="One local T7 best checkpoint for single-run single-mode single-client test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    if min(args.width, args.batch_size, args.physical_batch_size, args.rounds, args.local_epochs) < 1 or args.size < 16:
        parser.error("Model sizes, logical/physical batches and training epochs must be positive; size must be at least 16")
    if min(args.extra_epochs, args.max_batches, args.workers, args.lr_patience) < 0 or min(args.lr, args.noise_multiplier, args.clip_norm) <= 0:
        parser.error("Invalid training budget, workers, learning rate, noise or clipping norm")
    if args.physical_batch_size > args.batch_size or not 0 <= args.score_target <= 100:
        parser.error("Physical batch cannot exceed logical batch; score-target must be in [0,100]")
    if len(set(args.runs)) != len(args.runs) or len(set(args.clients)) != len(args.clients):
        parser.error("Run and client selections must be unique")
    if args.command == "train" and sorted(args.clients) != list(CLIENTS):
        parser.error("T7 training uses all six clients; --clients selects test exports")
    if args.checkpoint and (args.command != "test" or len(args.runs) != 1 or args.mode == "both" or len(args.clients) != 1):
        parser.error("--checkpoint requires one test run, one DP mode and one client")
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
        return
    rows = read_manifest(args.manifest, args.data_root)
    if args.command == "train":
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        protocol = dict(task="T7", model_config=model_config(args), manifest_signature=manifest_signature(rows),
                        shared_settings={key: getattr(args, key) for key in ("seed", "batch_size", "physical_batch_size", "local_epochs",
                                         "extra_epochs", "max_batches", "lr", "lr_patience", "fixed_order", "no_withdraw", "score_target",
                                         "noise_multiplier", "clip_norm")})
        path = root / "protocol_private.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("All DP modes and repetitions in one output directory must share the same data split and training protocol.")
        if not path.exists():
            path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    modes = ("no_dp", "dp") if args.mode == "both" else (args.mode,)
    for run in args.runs:
        for mode in modes:
            (train_run if args.command == "train" else test_run)(args, rows, run, mode)


if __name__ == "__main__":
    main()
