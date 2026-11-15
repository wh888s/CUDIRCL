import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLASSES = ("neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte",
           "immature_granulocyte", "erythroblast", "platelet")
CLIENT_CLASSES = {21: (0, 1), 22: (2, 3), 23: (4, 5), 24: (6, 7)}


def class_id(value):
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {name + "s": index for index, name in enumerate(CLASSES)}
    aliases.update({name: index for index, name in enumerate(CLASSES)})
    aliases.update(ig=5, immature_granulocytes=5)
    if text in aliases:
        return aliases[text]
    if text in {str(index) for index in range(8)}:
        return int(text)
    raise ValueError(f"Unknown PBC class {value}; numeric labels must follow this script's fixed CLASSES order.")


def condition_vector(client):
    vector = torch.zeros(4)
    vector[client - 21] = 1
    return vector


def read_manifest(path, root, index=False, allow_unlabeled=False):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "path"} | (set() if index else {"fold"})
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}; supply labels by cell-type name or fixed global ID.")
        rows, identifiers, files = [], set(), set()
        for item in reader:
            row = {key: (value or "").strip() for key, value in item.items() if key}
            if row.get("include", "1").lower() in {"0", "false"}:
                continue
            identifier = row["image_id"]
            source = str((Path(root) / row["path"]).resolve())
            if not identifier or identifier in identifiers or source in files:
                raise ValueError(f"Duplicate/empty original image ID or duplicate source file: {identifier}")
            if not Path(source).is_file():
                raise FileNotFoundError(source)
            identifiers.add(identifier)
            files.add(source)
            label = class_id(row["label"]) if row.get("label", "") else None
            if label is None and (index or not allow_unlabeled):
                raise ValueError("Split preparation and training require the original global cell-type label.")
            client = int(row["client"]) if row.get("client", "") else 21 + label // 2 if label is not None else None
            if client not in CLIENT_CLASSES or (label is not None and label not in CLIENT_CLASSES[client]):
                raise ValueError(f"Class/client mismatch: image {identifier}, label={label}, client={client}")
            fold = int(row["fold"]) if row.get("fold", "") else None
            if (fold is None and not index) or (fold is not None and fold not in range(5)):
                raise ValueError("Each original image must have exactly one fold ID in 0,1,2,3,4.")
            rows.append(dict(image_id=identifier, path=source, label=label, client=client, fold=fold))
    if not rows:
        raise ValueError("The original-image manifest is empty.")
    return rows


def make_split(args):
    if not args.index:
        raise ValueError("split requires --index with original image_id,path,label, optionally client and fold.")
    rows = read_manifest(args.index, args.data_root, index=True)
    assigned = [row["fold"] is not None for row in rows]
    if any(assigned) and not all(assigned):
        raise ValueError("Provide all original fold assignments or none; do not mix assigned and unassigned images.")
    if not any(assigned):
        rng = random.Random(args.seed)
        for label in range(8):
            selected = sorted((row for row in rows if row["label"] == label), key=lambda row: row["image_id"])
            if len(selected) < 5:
                raise ValueError(f"At least five original images of {CLASSES[label]} are needed for stratified five-fold splitting.")
            rng.shuffle(selected)
            for index, row in enumerate(selected):
                row["fold"] = index % 5
    for label in range(8):
        if {row["fold"] for row in rows if row["label"] == label} != set(range(5)):
            raise ValueError(f"All five folds must contain {CLASSES[label]}.")
    destination = Path(args.manifest)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "path", "label", "client", "fold"))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Assigned {len(rows)} original images to five folds before augmentation.")


def split_for(row, test_fold):
    return "test" if row["fold"] == test_fold else "val" if row["fold"] == (test_fold + 1) % 5 else "train"


def center_crop_pad(image, size):
    if not size:
        return image
    dw, dh = max(0, size - image.width), max(0, size - image.height)
    image = ImageOps.expand(image, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill=0)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


class PBCDataset(Dataset):
    def __init__(self, rows, client, fold, split, size=384, crop_size=0, augment=True):
        self.rows = [row for row in rows if row["client"] == client and split_for(row, fold) == split]
        self.client, self.fold, self.split = client, fold, split
        self.size, self.crop_size, self.augment = size, crop_size, augment and split == "train"
        self.pair, self.condition = CLIENT_CLASSES[client], condition_vector(client)
        if not self.rows:
            raise ValueError(f"No {split} images for client {client}, outer fold {fold}.")
        if split != "test" and any(row["label"] is None for row in self.rows):
            raise ValueError("Training and validation need global labels.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            if source.mode != "RGB":
                raise ValueError(f"Expected an original RGB PBC image: {row['path']}")
            image = source.copy()
        original_shape = [image.height, image.width]
        if self.augment:
            if torch.rand(()) < .5:
                image = ImageOps.mirror(image)
            if torch.rand(()) < .5:
                image = ImageOps.flip(image)
            turn = int(torch.randint(0, 4, ()))
            if turn:
                image = image.transpose((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)[turn - 1])
        image = center_crop_pad(image, self.crop_size).resize((self.size, self.size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255
        metadata = dict(image_id=row["image_id"], source=row["path"], original_shape=original_shape,
                        client=self.client, fold=self.fold, global_target=row["label"])
        result = dict(image=(tensor - .5) / .5, condition=self.condition.clone(), metadata=json.dumps(metadata))
        if self.split != "test":
            result.update(target=torch.tensor(self.pair.index(row["label"])), global_target=torch.tensor(row["label"]))
        return result


class DoubleConv(nn.Module):
    def __init__(self, inputs, outputs, middle=None):
        super().__init__()
        middle = middle or outputs
        self.layers = nn.Sequential(nn.Conv2d(inputs, middle, 3, padding=1), nn.GroupNorm(1, middle), nn.ReLU(inplace=True),
                                    nn.Conv2d(middle, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.block = DoubleConv(inputs, outputs, inputs // 2)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.block(torch.cat((skip, x), dim=1))


class FeatureExtractor(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        channels = [width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.encoder = nn.ModuleList(DoubleConv(inputs, outputs) for inputs, outputs in zip([3] + channels[:-1], channels))
        self.decoder = nn.ModuleList(Up(inputs, outputs) for inputs, outputs in zip([16 * width, 8 * width, 4 * width, 2 * width],
                                                                                 [4 * width, 2 * width, width, width]))

    def forward(self, image):
        features = []
        for index, block in enumerate(self.encoder):
            image = block(F.max_pool2d(image, 2) if index else image)
            features.append(image)
        value = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            value = block(value, skip)
        return F.adaptive_avg_pool2d(value, 1).flatten(1)


class ClassificationModel(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.features, self.head = FeatureExtractor(width), nn.Linear(width, 2)

    def forward(self, image):
        features = self.features(image)
        return self.head(features), features


class Proxy(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.modulation = nn.Sequential(nn.Linear(4, width), nn.ReLU(inplace=True), nn.Linear(width, 2 * width))
        self.head = nn.Linear(width, 8)

    def forward(self, features, condition):
        gamma, beta = self.modulation(condition).chunk(2, dim=1)
        return self.head(features.detach() * (1 + gamma) + beta)


def paired_logits(proxy_logits, client):
    return proxy_logits[:, list(CLIENT_CLASSES[client])]


def kl_distance(student, teacher):
    log_p, log_q = student.log_softmax(1), teacher.detach().log_softmax(1)
    return (log_p.exp() * (log_p - log_q)).sum(1).mean()


def train_batch(model, proxy, optimizer, proxy_optimizer, batch, client, device):
    image, condition, target, global_target = [batch[key].to(device) for key in ("image", "condition", "target", "global_target")]
    optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    local, features = model(image)
    shared = proxy(features, condition)
    pair = paired_logits(shared, client)
    local_loss = F.cross_entropy(local, target) + kl_distance(local, pair)
    proxy_loss = F.cross_entropy(shared, global_target) + kl_distance(pair, local)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite classification training loss.")
    local_loss.backward()
    proxy_loss.backward()
    optimizer.step()
    proxy_optimizer.step()
    return local_loss.item(), proxy_loss.item()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, correct, count = 0., 0, 0
    for batch in loader:
        logits, _ = model(batch["image"].to(device))
        target = batch["target"].to(device)
        total += F.cross_entropy(logits, target, reduction="sum").item()
        correct += (logits.argmax(1) == target).sum().item()
        count += len(target)
    result = dict(loss=total / count, score=100 * correct / count)
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite validation result.")
    return result


def loader_for(rows, client, fold, split, args, config=None):
    settings = config or model_config(args)
    dataset = PBCDataset(rows, client, fold, split, settings["size"], settings["crop_size"], not args.no_augment)
    generator = torch.Generator().manual_seed(args.seed + 1000000 * fold + client)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=args.workers,
                      drop_last=False, pin_memory=str(args.device).startswith("cuda"), generator=generator)


def model_config(args):
    return dict(width=args.width, size=args.size, crop_size=args.crop_size)


def manifest_signature(rows):
    records = sorted((r["image_id"], r["path"], r["label"], r["client"], r["fold"]) for r in rows)
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def checkpoint_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def train_fold(args, rows, fold):
    device, output = torch.device(args.device), Path(args.output) / f"fold{fold}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    last = output / "last.pt"
    if last.exists() and not args.resume:
        raise FileExistsError(f"Fold {fold} already has a run; use --resume or a new output directory.")
    torch.manual_seed(args.seed + 1000000 * fold)
    models = {client: ClassificationModel(args.width).to(device) for client in CLIENT_CLASSES}
    proxy = Proxy(args.width).to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in CLIENT_CLASSES}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                  for client, optimizer in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=args.lr_patience)
                        for client, optimizer in proxy_optimizers.items()}
    training = {client: loader_for(rows, client, fold, "train", args) for client in CLIENT_CLASSES}
    validation = {client: loader_for(rows, client, fold, "val", args) for client in CLIENT_CLASSES}
    for client in CLIENT_CLASSES:
        if any({row["label"] for row in loader.dataset.rows} != set(CLIENT_CLASSES[client]) for loader in (training[client], validation[client])):
            raise ValueError("Every client needs both local classes in training and validation.")
    best = {client: dict(score=-math.inf, loss=math.inf) for client in CLIENT_CLASSES}
    retired, order, first_round = set(), args.initial_order.copy(), 1
    signature = manifest_signature(rows)
    if args.resume and last.exists():
        state = checkpoint_load(last)
        controlled = ("seed", "initial_order", "batch_size", "local_epochs", "extra_epochs", "max_batches",
                      "lr", "lr_patience", "fixed_order", "score_target", "no_augment")
        if state["fold"] != fold or state["model_config"] != model_config(args) or state["manifest_signature"] != signature:
            raise ValueError("Resume requires the same outer fold, model and original-image assignments.")
        if state["classes"] != list(CLASSES) or any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume label mapping or training settings differ from the saved run.")
        proxy.load_state_dict(state["proxy"], strict=True)
        for client in CLIENT_CLASSES:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
        best, retired, order = state["best"], set(state["retired"]), state["order"]
        first_round = state["round"] + 1
        if any(not (output / "checkpoints" / f"client{client}_best.pt").exists() for client in CLIENT_CLASSES):
            raise FileNotFoundError("Resume in the directory containing this fold's best checkpoints.")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), fold=fold, validation_fold=(fold + 1) % 5,
                       classes=CLASSES, client_classes=CLIENT_CLASSES, manifest_signature=signature), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        if not order:
            break
        extra_client = order[-1] if round_number > 1 and not args.fixed_order else None
        losses, scores = {}, {}
        print(f"Fold {fold}, round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()
            training[client].generator.manual_seed(args.seed + 1000000 * fold + 1000 * round_number + client)
            total, count = np.zeros(2), 0
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            for _ in range(epochs):
                for index, batch in enumerate(training[client]):
                    if args.max_batches and index >= args.max_batches:
                        break
                    number = len(batch["image"])
                    total += np.asarray(train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client], batch, client, device)) * number
                    count += number
            losses[client] = dict(local=float(total[0] / count), proxy=float(total[1] / count))
        for client in CLIENT_CLASSES:
            result = validate(models[client], validation[client], device)
            scores[client] = result
            if client in order:
                schedulers[client].step(result["loss"])
                proxy_schedulers[client].step(result["loss"])
            if (result["score"], -result["loss"]) > (best[client]["score"], -best[client]["loss"]):
                best[client] = result
                torch.save(dict(task="T5", fold=fold, client=client, round=round_number, model_config=model_config(args),
                                classes=list(CLASSES), pair=list(CLIENT_CLASSES[client]), manifest_signature=signature,
                                test_ids=sorted(row["image_id"] for row in rows if row["client"] == client and row["fold"] == fold),
                                model=models[client].state_dict()), output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation loss={result['loss']:.6f}, score={result['score']:.4f}")
        if not args.fixed_order:
            if args.score_target is not None:
                retired.update(client for client in CLIENT_CLASSES if scores[client]["score"] >= args.score_target)
            order = sorted((client for client in CLIENT_CLASSES if client not in retired), key=lambda client: (-scores[client]["score"], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, training=losses, validation=scores,
                                         next_order=order, retired=sorted(retired))) + "\n")
        torch.save(dict(args=vars(args), task="T5", fold=fold, round=round_number, model_config=model_config(args),
                        manifest_signature=signature, classes=list(CLASSES), models={c: model.state_dict() for c, model in models.items()},
                        proxy=proxy.state_dict(), optimizers={c: optimizer.state_dict() for c, optimizer in optimizers.items()},
                        proxy_optimizers={c: optimizer.state_dict() for c, optimizer in proxy_optimizers.items()},
                        schedulers={c: scheduler.state_dict() for c, scheduler in schedulers.items()},
                        proxy_schedulers={c: scheduler.state_dict() for c, scheduler in proxy_schedulers.items()},
                        best=best, order=order, retired=sorted(retired), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), last)


@torch.no_grad()
def test_fold(args, rows, fold):
    device = torch.device(args.device)
    destination = Path(args.output) / f"fold{fold}" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    for client in args.clients:
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.output) / f"fold{fold}" / "checkpoints" / f"client{client}_best.pt"
        state = checkpoint_load(checkpoint)
        if state.get("task") != "T5" or state["fold"] != fold or state["client"] != client:
            raise ValueError("Testing requires a T5 best checkpoint for the requested fold and client.")
        if state["classes"] != list(CLASSES) or state["pair"] != list(CLIENT_CLASSES[client]):
            raise ValueError("Checkpoint class semantics differ from the fixed eight-class mapping.")
        model = ClassificationModel(state["model_config"]["width"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        loader = loader_for(rows, client, fold, "test", args, state["model_config"])
        if not {row["image_id"] for row in loader.dataset.rows} <= set(state["test_ids"]):
            raise ValueError("Test contains original image IDs outside this checkpoint's held-out fold.")
        pair, count = CLIENT_CLASSES[client], 0
        fields = ("image_id", "source", "fold", "client", "global_target", "local_target", "predicted_local", "predicted_global",
                  "predicted_name", "probability_0", "probability_1", "source_height", "source_width")
        with (destination / f"client{client}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for batch in loader:
                logits, _ = model(batch["image"].to(device))
                probability = logits.softmax(1).cpu().numpy()
                for values, text in zip(probability, batch["metadata"]):
                    metadata = json.loads(text)
                    prediction = int(values.argmax())
                    target = metadata["global_target"]
                    writer.writerow(dict(image_id=metadata["image_id"], source=metadata["source"], fold=fold, client=client,
                                         global_target="" if target is None else target, local_target="" if target is None else pair.index(target),
                                         predicted_local=prediction, predicted_global=pair[prediction], predicted_name=CLASSES[pair[prediction]],
                                         probability_0=float(values[0]), probability_1=float(values[1]),
                                         source_height=metadata["original_shape"][0], source_width=metadata["original_shape"][1]))
                    count += 1
        with (destination / f"client{client}.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(task="T5", fold=fold, client=client, checkpoint=str(checkpoint.resolve()), round=state["round"],
                           local_classes={i: dict(global_id=label, name=CLASSES[label]) for i, label in enumerate(pair)},
                           global_classes=CLASSES, model_config=state["model_config"],
                           normalization=dict(mean=[.5, .5, .5], std=[.5, .5, .5]), augmentation=False,
                           classifier="local two-class task head", image_id_source="author-supplied original image IDs"), handle, indent=2)
        print(f"Fold {fold}, client {client}: exported {count} classification predictions.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CUDIRCL T5 five-fold PBC classification")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", default="", help="Original image_id,path,label CSV; use cell-type names to avoid numeric mapping ambiguity")
    parser.add_argument("--output", default="pathology_run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)), help="Outer test folds; the next fold is validation and the other three are training")
    parser.add_argument("--clients", nargs="+", type=int, choices=tuple(CLIENT_CLASSES), default=list(CLIENT_CLASSES), help="Clients to export in test")
    parser.add_argument("--initial-order", nargs=4, type=int, choices=tuple(CLIENT_CLASSES), default=list(CLIENT_CLASSES))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=0, help="Optional center crop/pad in source pixels; 0 keeps the entire image")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="Training-only epoch limit; 0 uses all training images")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--score-target", type=float, default=None, help="Optional validation ACC threshold for client withdrawal; disabled by default")
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume requested folds from their own last.pt; missing folds start fresh")
    parser.add_argument("--checkpoint", default="", help="Single-fold single-client T5 best checkpoint for test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=65)
    args = parser.parse_args(argv)
    if min(args.width, args.batch_size, args.rounds, args.local_epochs) < 1 or args.size < 16:
        parser.error("Widths, batches, rounds and epochs must be positive; image size must be at least 16")
    if min(args.crop_size, args.extra_epochs, args.max_batches, args.lr_patience, args.workers) < 0 or args.lr <= 0:
        parser.error("Invalid crop, training budget, scheduler, workers or learning rate")
    if args.score_target is not None and not 0 <= args.score_target <= 100:
        parser.error("score-target must be in [0,100]")
    if len(set(args.initial_order)) != 4 or len(set(args.folds)) != len(args.folds) or len(set(args.clients)) != len(args.clients):
        parser.error("Client order must contain all four clients once; fold/client selections must be unique")
    if args.command == "train" and sorted(args.clients) != list(CLIENT_CLASSES):
        parser.error("Training uses all four clients; --clients selects test exports")
    if args.checkpoint and (args.command != "test" or len(args.folds) != 1 or len(args.clients) != 1):
        parser.error("--checkpoint requires test with one fold and one client")
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
        rows = read_manifest(args.manifest, args.data_root, allow_unlabeled=args.command == "test")
        for fold in args.folds:
            (train_fold if args.command == "train" else test_fold)(args, rows, fold)


if __name__ == "__main__":
    main()
