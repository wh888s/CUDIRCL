import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


HU_MIN, HU_MAX = -1024.0, 2500.0
CLIENTS = {
    25: dict(folder="CT_1/abdomen/1_4/kernel1", vendor=1, anatomy="abdomen", kernel="sharp", kvp=120, mas=75),
    26: dict(folder="CT_2/abdomen/1_8/kernel1", vendor=2, anatomy="abdomen", kernel="sharp", kvp=120, mas=40),
    27: dict(folder="CT_1/chest/1_10/kernel3", vendor=1, anatomy="chest", kernel="smooth", kvp=100, mas=28),
    28: dict(folder="CT_1/head/1_6/kernel1", vendor=1, anatomy="brain", kernel="sharp", kvp=120, mas=48),
}


def conditioning(client):
    profile = CLIENTS[client]
    return torch.tensor([profile["kvp"] / 140, profile["mas"] / 100,
                         profile["vendor"] == 1, profile["vendor"] == 2,
                         profile["anatomy"] == "abdomen", profile["anatomy"] == "chest",
                         profile["anatomy"] == "brain", profile["kernel"] == "sharp"], dtype=torch.float32)


def source_key(path, root):
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path, root):
    rows, groups, sources = [], {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"client", "case_id", "split", "path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs client,case_id,split,path columns.")
        for item in reader:
            client, case, split = int(item["client"]), item["case_id"].strip(), item["split"].strip()
            path = (Path(root) / item["path"]).resolve()
            if client not in CLIENTS or not case or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid manifest row: {item}")
            if (client, case) in groups and groups[client, case] != split:
                raise ValueError(f"Case leakage across splits: client {client}, {case}")
            if path in sources:
                raise ValueError(f"Duplicate input file: {path}")
            groups[client, case] = split
            sources.add(path)
            rows.append(dict(client=client, case_id=case, split=split, path=path,
                             source_key=source_key(path, root)))
    return rows


def case_id(path, folder):
    value = loadmat(path, variable_names=["case_id"]).get("case_id")
    if value is not None:
        flat = np.asarray(value).reshape(-1)
        if flat.size:
            return "".join(str(item) for item in flat) if flat.dtype.kind in "US" else str(flat[0])
    parts = path.relative_to(folder).parts
    if parts[0] in {"train", "val", "test"}:
        parts = parts[1:]
    if len(parts) == 2:
        return parts[0]
    raise ValueError(f"Missing subject ID in {path}; provide case_id metadata, case folders, or a CSV manifest.")


def partition_cases(groups, ratios, rng):
    if len(groups) < len(ratios):
        raise ValueError("Not enough subjects for independent nonempty splits.")
    cases = list(groups)
    rng.shuffle(cases)
    cases.sort(key=lambda case: len(groups[case]), reverse=True)
    target = np.asarray(ratios) * sum(map(len, groups.values()))
    counts, allocation = np.zeros(len(ratios)), [[] for _ in ratios]
    for index, case in enumerate(cases):
        empty = [i for i, items in enumerate(allocation) if not items]
        chosen = empty[0] if len(cases) - index == len(empty) else int(np.argmax(target - counts))
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
    for client, name in zip(CLIENTS, args.folders):
        folder = root / name
        paths = sorted(folder.rglob("*.mat"))
        if not paths:
            raise ValueError(f"No MAT files found in {folder}")
        rng = random.Random(args.seed + client)
        candidates, preset = [], {}
        for path in paths:
            case = case_id(path, folder)
            first = path.relative_to(folder).parts[0]
            split = first if first in {"train", "val", "test"} else "unsplit"
            if case in preset and preset[case] != split:
                raise ValueError(f"Subject {case} occurs in multiple source splits in {folder}")
            preset[case] = split
            candidates.append((path, case))
        if args.samples_per_client:
            if len(candidates) < args.samples_per_client:
                raise ValueError(f"Client {client}: need {args.samples_per_client} images, found {len(candidates)}.")
            candidates = sorted(rng.sample(candidates, args.samples_per_client))
        groups = defaultdict(list)
        for path, case in candidates:
            groups[case].append((path, case))
        source_splits = set(preset.values())
        if source_splits == {"unsplit"}:
            allocation = partition_cases(groups, (.80, .05, .15), rng)
        elif source_splits == {"train", "test"}:
            allocation = legacy_partition(groups, preset, rng)
        elif source_splits == {"train", "val", "test"}:
            allocation = [[item for case, items in groups.items() if preset[case] == split for item in items]
                          for split in ("train", "val", "test")]
        else:
            raise ValueError(f"Use unsplit cases, train/test, or train/val/test in {folder}.")
        if not all(allocation):
            raise ValueError(f"Client {client}: selection produced an empty split; supply a case-grouped manifest.")
        for split, items in zip(("train", "val", "test"), allocation):
            records.extend(dict(client=client, case_id=case, split=split, path=path.relative_to(root).as_posix())
                           for path, case in items)
        print(f"Client {client}: train/val/test = {[len(items) for items in allocation]}")
    target = Path(args.manifest)
    if target.exists():
        raise FileExistsError(f"Preserve existing manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client", "case_id", "split", "path"])
        writer.writeheader()
        writer.writerows(records)
    read_manifest(target, root)


def split_signatures(rows):
    return {client: hashlib.sha256(json.dumps(sorted(
        (row["case_id"], row["split"], row["source_key"]) for row in rows if row["client"] == client)).encode()).hexdigest()
        for client in CLIENTS}


class DenoisingDataset(Dataset):
    def __init__(self, rows, client, split, image_key="low_dose"):
        self.rows = [row for row in rows if row["client"] == client and row["split"] == split]
        self.client, self.image_key = client, image_key
        if not self.rows:
            raise ValueError(f"Missing {split} data for client {client}.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mat = loadmat(row["path"], variable_names=[self.image_key])
        if self.image_key not in mat:
            raise ValueError(f"Missing {self.image_key} in {row['path']}")
        image = np.asarray(mat[self.image_key], dtype=np.float32).squeeze()
        if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
            raise ValueError(f"Expected a finite 2D HU image in {row['path']}")
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        return dict(image=(image - HU_MIN) / (HU_MAX - HU_MIN), low_dose=image,
                    parameters=conditioning(self.client), case_id=row["case_id"],
                    source_key=row["source_key"], source_name=row["path"].name)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(1, out_channels, affine=False), nn.LeakyReLU(.2, inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(1, out_channels, affine=False), nn.LeakyReLU(.2, inplace=True))


class Encoder(nn.Module):
    def __init__(self, width=32, depth=5):
        super().__init__()
        self.factor = 2 ** (depth - 1)
        channels = [width * 2 ** index for index in range(depth)]
        self.blocks = nn.ModuleList([ConvBlock(1 if index == 0 else channels[index - 1], channels[index])
                                     for index in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(channel, channel, 4, stride=2, padding=1, bias=False)
                                  for channel in channels[:-1]])

    def forward(self, image):
        height, width = image.shape[-2:]
        image = F.pad(image, (0, -width % self.factor, 0, -height % self.factor), mode="replicate")
        skips = []
        for index, block in enumerate(self.blocks):
            image = block(image)
            if index < len(self.down):
                skips.append(image)
                image = self.down[index](image)
        return image, tuple(skips), (height, width)


class Decoder(nn.Module):
    def __init__(self, width=32, depth=5, conditioned=False):
        super().__init__()
        self.channels = [width * 2 ** index for index in reversed(range(depth - 1))]
        self.up = nn.ModuleList([nn.ConvTranspose2d(2 * channel, channel, 2, stride=2, bias=False)
                                 for channel in self.channels])
        self.blocks = nn.ModuleList([ConvBlock(2 * channel, channel) for channel in self.channels])
        self.output = nn.Conv2d(width, 1, 3, padding=1, bias=False)
        total = sum(self.channels)
        self.film = (nn.Sequential(nn.Linear(8, max(1, total // 2)), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(max(1, total // 2), total), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(total, 2 * total)) if conditioned else None)

    def forward(self, features, parameters=None):
        image, skips, shape = features
        if self.film is not None:
            gamma, beta = self.film(parameters).chunk(2, dim=1)
        offset = 0
        for index, (up, block) in enumerate(zip(self.up, self.blocks)):
            image = block(torch.cat((up(image), skips[-index - 1]), dim=1))
            if self.film is not None:
                channels = self.channels[index]
                scale = gamma[:, offset:offset + channels, None, None]
                shift = beta[:, offset:offset + channels, None, None]
                image = scale * image + shift
                offset += channels
        return self.output(image)[..., :shape[0], :shape[1]]


class DenoisingModel(nn.Module):
    def __init__(self, width=32, depth=5):
        super().__init__()
        self.encoder = Encoder(width, depth)
        self.decoder = Decoder(width, depth)

    def forward(self, image):
        features = self.encoder(image)
        return features, self.decoder(features)


def detached_features(features):
    image, skips, shape = features
    return image.detach(), tuple(skip.detach() for skip in skips), shape


def self_supervised_loss(prediction, noisy):
    tv = ((prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean()
          + (prediction[..., :, 1:] - prediction[..., :, :-1]).abs().mean())
    return F.mse_loss(prediction, noisy) + .3 * tv


def train_batch(model, proxy, local_optimizer, proxy_optimizer, batch, device):
    image, parameters = batch["image"].to(device), batch["parameters"].to(device)
    local_optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    features, local = model(image)
    shared = proxy(detached_features(features), parameters)
    local_loss = self_supervised_loss(local, image) + F.huber_loss(local, shared.detach(), delta=.1)
    proxy_loss = self_supervised_loss(shared, image) + F.huber_loss(shared, local.detach(), delta=.1)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite loss; check HU units and input data.")
    local_loss.backward()
    proxy_loss.backward()
    local_optimizer.step()
    proxy_optimizer.step()
    return local_loss.item(), proxy_loss.item()


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, samples = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        _, prediction = model(image)
        total += self_supervised_loss(prediction, image).item() * image.shape[0]
        samples += image.shape[0]
    result = total / samples
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite validation loss.")
    return result


def loader_for(rows, client, split, args, image_key=None):
    generator = torch.Generator().manual_seed(args.seed + client)
    return DataLoader(DenoisingDataset(rows, client, split, image_key or args.image_key),
                      batch_size=args.batch_size, shuffle=split == "train", num_workers=args.workers,
                      pin_memory=str(args.device).startswith("cuda"), generator=generator)


def save_checkpoint(state, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def model_config(args):
    return dict(width=args.width, depth=args.depth, image_key=args.image_key,
                hu_min=HU_MIN, hu_max=HU_MAX, profiles=CLIENTS)


def train(args, rows):
    for client in CLIENTS:
        count = sum(row["client"] == client for row in rows)
        if args.samples_per_client and count != args.samples_per_client:
            raise ValueError(f"Client {client}: manifest has {count} images; expected {args.samples_per_client}.")
    output, device = Path(args.output), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output already contains a run; use --resume or a new --output.")
    models = {client: DenoisingModel(args.width, args.depth).to(device) for client in CLIENTS}
    proxy = Decoder(args.width, args.depth, conditioned=True).to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in CLIENTS}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                  for client, opt in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=args.lr_patience)
                        for client, opt in proxy_optimizers.items()}
    training = {client: loader_for(rows, client, "train", args) for client in CLIENTS}
    validation = {client: loader_for(rows, client, "val", args) for client in CLIENTS}
    signatures, best, first_round = split_signatures(rows), {client: math.inf for client in CLIENTS}, 1
    order = list(CLIENTS)
    random.Random(args.seed).shuffle(order)
    if args.resume:
        state = load_checkpoint(args.resume)
        if state["model_config"] != model_config(args) or state["split_signatures"] != signatures:
            raise ValueError("Resume requires the same model configuration and subject/split assignments.")
        controlled = ("seed", "batch_size", "local_epochs", "extra_epochs", "fixed_order",
                      "lr", "lr_patience", "max_batches", "samples_per_client")
        if any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume training settings differ from the saved run.")
        for client in CLIENTS:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
            if not (output / "checkpoints" / f"client{client}_best.pt").exists():
                raise FileNotFoundError("Resume in the run directory containing its best checkpoints.")
        proxy.load_state_dict(state["proxy"], strict=True)
        best, order, first_round = state["best"], state["order"], state["round"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), model_config=model_config(args)), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        used_order = list(order)
        extra_client = order[-1] if round_number > 1 and not args.fixed_order else None
        losses = {}
        print(f"Round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()
            training[client].generator.manual_seed(args.seed + 10000 * round_number + client)
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            totals, samples = np.zeros(2), 0
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    values = train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client], batch, device)
                    number = batch["image"].shape[0]
                    totals += np.asarray(values) * number
                    samples += number
            losses[client] = dict(local=float(totals[0] / samples), proxy=float(totals[1] / samples))
        values = {}
        for client in CLIENTS:
            values[client] = validation_loss(models[client], validation[client], device)
            schedulers[client].step(values[client])
            proxy_schedulers[client].step(values[client])
            if values[client] < best[client]:
                best[client] = values[client]
                save_checkpoint(dict(client=client, round=round_number, model_config=model_config(args),
                                     split_signatures=signatures, model=models[client].state_dict()),
                                output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation loss={values[client]:.6f}")
        if not args.fixed_order:
            order = sorted(CLIENTS, key=lambda client: (values[client], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, order=used_order, extra_client=extra_client,
                                         training=losses, validation_loss=values, next_order=order)) + "\n")
        save_checkpoint(dict(args=vars(args), model_config=model_config(args), round=round_number,
                             split_signatures=signatures, models={c: model.state_dict() for c, model in models.items()},
                             proxy=proxy.state_dict(), optimizers={c: opt.state_dict() for c, opt in optimizers.items()},
                             proxy_optimizers={c: opt.state_dict() for c, opt in proxy_optimizers.items()},
                             schedulers={c: sch.state_dict() for c, sch in schedulers.items()},
                             proxy_schedulers={c: sch.state_dict() for c, sch in proxy_schedulers.items()},
                             best=best, order=order, torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), output / "last.pt")


def safe_name(value):
    clean = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)[:80]
    return clean or "image"


@torch.inference_mode()
def test(args, rows):
    output, device, signatures = Path(args.output), torch.device(args.device), split_signatures(rows)
    model_clients = list(dict.fromkeys(args.model_clients or CLIENTS))
    data_clients = list(dict.fromkeys(args.data_clients or CLIENTS))
    for model_client in model_clients:
        selected = [client for client in data_clients if not args.same_client_only or client == model_client]
        if not selected:
            continue
        state = load_checkpoint(output / "checkpoints" / f"client{model_client}_best.pt")
        config = state["model_config"]
        if state["client"] != model_client or any(state["split_signatures"][client] != signatures[client] for client in selected):
            raise ValueError("Test manifest or checkpoint client does not match the training run.")
        if (config["hu_min"], config["hu_max"], config["profiles"]) != (HU_MIN, HU_MAX, CLIENTS):
            raise ValueError("Checkpoint normalization or acquisition profiles differ.")
        model = DenoisingModel(config["width"], config["depth"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        for data_client in selected:
            loader = loader_for(rows, data_client, "test", args, image_key=config["image_key"])
            count = 0
            for batch in loader:
                _, prediction = model(batch["image"].to(device))
                prediction = (prediction * (HU_MAX - HU_MIN) + HU_MIN).cpu().numpy()
                for index, image in enumerate(prediction):
                    case, name = batch["case_id"][index], batch["source_name"][index]
                    key = batch["source_key"][index]
                    token = hashlib.sha256(key.encode()).hexdigest()[:10]
                    folder = output / "results" / f"model{model_client}" / f"data{data_client}" / safe_name(case)
                    folder.mkdir(parents=True, exist_ok=True)
                    data = dict(output=image[0], low_dose=batch["low_dose"][index, 0].numpy(),
                                model_client=model_client, data_client=data_client, case_id=case,
                                source_name=name, source_path=key, checkpoint_round=state["round"],
                                units="HU", acquisition=CLIENTS[data_client])
                    savemat(folder / f"{safe_name(Path(name).stem)}_{token}.mat", data, do_compression=True)
                    count += 1
            print(f"Model {model_client}, data {data_client}: exported {count} denoised images.")
        del model


def main():
    parser = argparse.ArgumentParser(description="CUDIRCL real-data image-domain CT denoising")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True, help="CSV columns: client,case_id,split,path; clients 25-28")
    parser.add_argument("--output", default="ct_denoising_run")
    parser.add_argument("--folders", nargs=4, default=[profile["folder"] for profile in CLIENTS.values()])
    parser.add_argument("--image-key", default="low_dose", help="MAT field containing a 2D low-dose image in HU")
    parser.add_argument("--samples-per-client", type=int, default=5000, help="0 uses all images; split selects and train checks this count")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--extra-epochs", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0, help="Training only; 0 uses all training images")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--resume", help="Resume from last.pt in the same run directory")
    parser.add_argument("--model-clients", nargs="+", type=int, choices=CLIENTS, help="Test model IDs; default all four")
    parser.add_argument("--data-clients", nargs="+", type=int, choices=CLIENTS, help="Test data IDs; default all four")
    parser.add_argument("--same-client-only", action="store_true", help="Export diagonal pairs instead of the full 4x4 model/data combinations")
    args = parser.parse_args()
    if min(args.width, args.batch_size, args.rounds, args.local_epochs) < 1 or args.depth < 2:
        parser.error("Widths, batch size, rounds, and epochs must be positive; depth must be at least 2.")
    if min(args.workers, args.extra_epochs, args.max_batches, args.samples_per_client, args.lr_patience) < 0 or args.lr <= 0:
        parser.error("Counts must be nonnegative and learning rate positive.")
    if args.image_key in {"ndct", "normal_dose", "reference", "label"}:
        parser.error("This experiment requires low-dose input, not paired normal-dose labels.")
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
