import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
                raise ValueError("T4 uses T2-weighted images only, without ADC channels.")
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
            raise ValueError(f"T4 requires grayscale MRI, not color or ADC data: {path}")
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
                raise ValueError("T4 uses foreground-containing slices; prepare the manifest with split.")
            result["target"] = F.interpolate(mask[None, None].float(), size=(self.size, self.size), mode="nearest")[0, 0].long()
        return result


class DoubleConv(nn.Module):
    def __init__(self, inputs, outputs, middle=None):
        super().__init__()
        middle = middle or outputs
        self.layers = nn.Sequential(nn.Conv2d(inputs, middle, 3, padding=1), nn.GroupNorm(1, middle), nn.ReLU(inplace=True),
                                    nn.Conv2d(middle, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.ReLU(inplace=True))

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
        self.mlp = nn.Sequential(nn.Linear(8, 4 * width), nn.ReLU(inplace=True), nn.Dropout(.5),
                                 nn.Linear(4 * width, 8 * width), nn.ReLU(inplace=True), nn.Dropout(.5),
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


def train_batch(model, proxy, optimizer, proxy_optimizer, batch, device):
    image, condition, target = [batch[key].to(device) for key in ("image", "condition", "target")]
    optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    local, features = model(image, condition)
    shared = proxy([feature.detach() for feature in features], condition)
    local_loss = task_loss(local, target) + kl_distance(local, shared)
    proxy_loss = task_loss(shared, target) + kl_distance(shared, local)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite segmentation training loss.")
    local_loss.backward()
    proxy_loss.backward()
    optimizer.step()
    proxy_optimizer.step()
    return local_loss.item(), proxy_loss.item()


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


def train(args, rows):
    output, device = Path(args.output), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use --resume or a new output directory.")
    models = {client: SegmentationModel(args.width).to(device) for client in CLIENTS}
    proxy = Decoder(args.width).to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in CLIENTS}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                  for client, optimizer in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                        for client, optimizer in proxy_optimizers.items()}
    training = {client: loader_for(rows, client, "train", args) for client in CLIENTS}
    validation = {client: loader_for(rows, client, "val", args) for client in CLIENTS}
    best, epochs_done = {client: -math.inf for client in CLIENTS}, {client: 0 for client in CLIENTS}
    order, first_phase = args.initial_clients.copy(), 1
    signature = manifest_signature(rows)
    if args.resume:
        state = checkpoint_load(args.resume)
        controlled = ("initial_clients", "initial_epochs", "additional_epochs", "batch_size", "max_batches", "lr", "lr_patience", "seed", "fixed_order")
        if state["model_config"] != model_config(args) or state["manifest_signature"] != signature:
            raise ValueError("Resume requires the same model and subject/split manifest.")
        if any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume training settings differ from the saved run.")
        proxy.load_state_dict(state["proxy"], strict=True)
        for client in CLIENTS:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
        best, epochs_done, order = state["best"], state["epochs_done"], state["order"]
        first_phase = state["phase"] + 1
        if any(math.isfinite(best[c]) and not (output / "checkpoints" / f"client{c}_best.pt").exists() for c in CLIENTS):
            raise FileNotFoundError("Resume in the run directory containing its best checkpoints.")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), clients=CLIENTS, manifest_signature=signature,
                       protocol="four clients, then six clients; two phases without repeated cycles"),
                  handle, indent=2)
    for phase in range(first_phase, args.stop_after_phase + 1):
        joined = args.initial_clients if phase == 1 else list(CLIENTS)
        arrivals = [client for client in joined if client not in order]
        order = arrivals + [client for client in order if client in joined]
        epochs = args.initial_epochs if phase == 1 else args.additional_epochs
        losses, scores = {}, {}
        print(f"Phase {phase}: clients={order}, local_epochs={epochs}")
        for client in order:
            models[client].train()
            proxy.train()
            training[client].generator.manual_seed(args.seed + 10000 * phase + client)
            totals, count = np.zeros(2), 0
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    number = len(batch["image"])
                    totals += np.asarray(train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client], batch, device)) * number
                    count += number
            epochs_done[client] += epochs
            losses[client] = dict(local=float(totals[0] / count), proxy=float(totals[1] / count))
        for client in joined:
            result = validate(models[client], validation[client], device)
            scores[client] = result
            schedulers[client].step(result["loss"])
            proxy_schedulers[client].step(result["loss"])
            if result["score"] > best[client]:
                best[client] = result["score"]
                torch.save(dict(task="T4", client=client, phase=phase, epochs_done=epochs_done[client],
                                model_config=model_config(args), manifest_signature=signature,
                                test_cases=sorted({row["case_id"] for row in rows if row["client"] == client and row["split"] == "test"}),
                                model=models[client].state_dict()),
                           output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation loss={result['loss']:.6f}, score={result['score']:.4f}")
        if not args.fixed_order:
            order = sorted(joined, key=lambda client: (-scores[client]["score"], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(phase=phase, training=losses, validation=scores,
                                         next_order=order, epochs_done=epochs_done)) + "\n")
        torch.save(dict(args=vars(args), model_config=model_config(args), manifest_signature=signature, phase=phase,
                        models={client: model.state_dict() for client, model in models.items()}, proxy=proxy.state_dict(),
                        optimizers={client: optimizer.state_dict() for client, optimizer in optimizers.items()},
                        proxy_optimizers={client: optimizer.state_dict() for client, optimizer in proxy_optimizers.items()},
                        schedulers={client: scheduler.state_dict() for client, scheduler in schedulers.items()},
                        proxy_schedulers={client: scheduler.state_dict() for client, scheduler in proxy_schedulers.items()},
                        best=best, epochs_done=epochs_done, order=order, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), output / "last.pt")


@torch.no_grad()
def test(args, rows):
    device = torch.device(args.device)
    for client in args.clients:
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output) / "checkpoints" / f"client{client}_best.pt"
        state = checkpoint_load(checkpoint)
        if state.get("task") != "T4" or state["client"] != client:
            raise ValueError("Testing requires a T4 best checkpoint for the requested client.")
        if "test_cases" in state:
            if not {row["case_id"] for row in rows if row["client"] == client and row["split"] == "test"} <= set(state["test_cases"]):
                raise ValueError("Test subjects must belong to this checkpoint's held-out cases.")
        elif state["manifest_signature"] != manifest_signature(rows):
            raise ValueError("Legacy checkpoints require the complete original held-out manifest.")
        model = SegmentationModel(state["model_config"]["width"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        destination = Path(args.output) / "results" / f"client{client}"
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for batch in loader_for(rows, client, "test", args, state["model_config"]["size"]):
            logits, _ = model(batch["image"].to(device), batch["condition"].to(device))
            metadata = json.loads(batch["metadata"][0])
            probability = F.interpolate(logits.softmax(1), size=metadata["original_shape"], mode="bilinear", align_corners=False)[0].cpu().numpy()
            output = probability.argmax(0).astype(np.uint8)
            metadata.update(checkpoint=str(checkpoint.resolve()), phase=state["phase"], epochs_done=state["epochs_done"])
            name = hashlib.sha256(metadata["source"].encode()).hexdigest()[:16]
            savemat(destination / f"{name}.mat", dict(output=output, probability=probability,
                                                     metadata=json.dumps(metadata, ensure_ascii=False)), do_compression=True)
            Image.fromarray(output * 255).save(destination / f"{name}.png")
            count += 1
        print(f"Client {client}: exported {count} segmentation slices to {destination}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDIRCL T4 two-stage prostate MRI segmentation")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", default="", help="Optional source CSV with client,case_id,image,mask columns")
    parser.add_argument("--case-pattern", default=r"(?P<case>.+)_(?P<slice>\d+)", help="Subject and slice identification for original cohort_subject_slice.png files")
    parser.add_argument("--output", default="mri_segmentation_run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-clients", nargs=4, type=int, choices=tuple(CLIENTS), default=[15, 16, 17, 18])
    parser.add_argument("--initial-epochs", type=int, default=10)
    parser.add_argument("--additional-epochs", type=int, default=5)
    parser.add_argument("--stop-after-phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-batches", type=int, default=0, help="Training-only epoch limit; 0 traverses all training images")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="", help="Single-client T4 best checkpoint for test")
    parser.add_argument("--clients", type=int, nargs="+", choices=tuple(CLIENTS), default=list(CLIENTS), help="Clients to export in test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    if min(args.width, args.batch_size, args.initial_epochs, args.additional_epochs) < 1 or args.size < 16:
        parser.error("Model sizes, epochs and batch size must be positive; image size must be at least 16")
    if min(args.workers, args.max_batches, args.lr_patience) < 0 or args.lr <= 0:
        parser.error("Invalid workers, batch limit, scheduler patience or learning rate")
    if len(set(args.initial_clients)) != 4:
        parser.error("The initial phase requires four distinct clients")
    if len(set(args.clients)) != len(args.clients):
        parser.error("Test clients must be unique")
    if args.command == "train" and sorted(args.clients) != list(CLIENTS):
        parser.error("Training uses the specified four-then-six protocol; --clients selects test exports")
    if args.checkpoint and (args.command != "test" or len(args.clients) != 1):
        parser.error("--checkpoint requires test with one --clients value")
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
