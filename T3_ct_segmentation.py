import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import random
import shutil
import tempfile
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform


CLIENTS = OrderedDict([
    (11, dict(task="Task532_CHAOSCT", dataset="CHAOS-CT", reported_cases=16)),
    (12, dict(task="Task533_BTCV", dataset="BTCV", reported_cases=24)),
    (13, dict(task="Task534_WORD", dataset="WORD", reported_cases=96)),
    (14, dict(task="Task535_AMOS", dataset="AMOS", reported_cases=160)),
    (3, dict(task="Task536_ABD1K", dataset="AbdomenCT-1K", reported_cases=80)),
])
ORGANS = ("kidney_left", "kidney_right", "liver", "spleen", "pancreas", "stomach")
ATLAS_LABELS = (3, 4, 5, 8, 6, 9)
POOL_KERNELS = ((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2))
CONV_KERNELS = ((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3))


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def data_root(path):
    root = Path(path).resolve()
    return root / "nnUNet_preprocessed" if (root / "nnUNet_preprocessed").is_dir() else root


def source_key(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def label_ids(task, override):
    if override is not None:
        result = tuple(override)
    elif (task / "dataset.json").exists():
        labels = json.loads((task / "dataset.json").read_text(encoding="utf-8"))["labels"]
        names = {}
        for key, value in labels.items():
            number, name = (int(key), value) if str(key).isdigit() else (int(value), key)
            names["".join(char for char in str(name).lower() if char.isalnum())] = number
        aliases = (("kidneyleft", "leftkidney"), ("kidneyright", "rightkidney"),
                   ("liver",), ("spleen",), ("pancreas",), ("stomach",))
        result = tuple(next((names[name] for name in options if name in names), -1) for options in aliases)
    else:
        result = ATLAS_LABELS
    if len(set(result)) != 6 or any(not 0 < value < 256 for value in result):
        raise ValueError(f"Need distinct source IDs for {ORGANS} in {task}; provide --label-ids.")
    return result


def load_contexts(args):
    root, contexts = data_root(args.data_root), {}
    for client, task_name in zip(CLIENTS, args.tasks):
        task = root / task_name
        plans_file = task / args.plans_name
        if not plans_file.exists():
            candidates = sorted(task.glob("*plans_3D.pkl"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot uniquely find nnU-Net v1 3D plans in {task}.")
            plans_file = candidates[0]
        plans = load_pickle(plans_file)
        stages = {int(key): value for key, value in plans["plans_per_stage"].items()}
        stage = max(stages) if args.stage is None else args.stage
        folder = task / f"{plans['data_identifier']}_stage{stage}"
        if stage not in stages or not folder.is_dir():
            raise FileNotFoundError(f"Missing stage {stage} preprocessing in {task}.")
        forward = tuple(int(value) for value in plans.get("transpose_forward", ()))
        backward = tuple(int(value) for value in plans.get("transpose_backward", ()))
        if sorted(forward) != [0, 1, 2] or backward != tuple(np.argsort(forward)):
            raise ValueError(f"Invalid/missing nnU-Net axis transpose metadata in {plans_file}.")
        if int(plans.get("num_modalities", 1)) != 1:
            raise ValueError("T3 requires single-channel CT preprocessing.")
        ids = label_ids(task, args.label_ids)
        allowed = sorted({0, *map(int, plans.get("all_classes", range(1, 10)))})
        if not set(ids) <= set(allowed):
            raise ValueError(f"Selected organ IDs disagree with the nnU-Net plans in {task}.")
        contexts[client] = dict(task=task, folder=folder, plans=plans, stage=stage,
                                transpose_backward=backward, label_ids=ids, allowed_labels=allowed,
                                plans_sha256=hashlib.sha256(plans_file.read_bytes()).hexdigest())
    return contexts


def read_manifest(path, root, contexts):
    rows, groups, files = [], {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"client", "case_id", "split", "path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs client,case_id,split,path columns.")
        for entry in reader:
            client, case, split = int(entry["client"]), entry["case_id"].strip(), entry["split"].strip()
            file = (root / entry["path"]).resolve()
            if client not in CLIENTS or not case or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid manifest row: {entry}")
            if (client, case) in groups and groups[client, case] != split:
                raise ValueError(f"Subject leakage across splits: client {client}, {case}")
            identity = file.with_suffix("")
            if identity in files:
                raise ValueError(f"Duplicate input volume or nnU-Net npz/npy cache alias: {file}")
            if file.parent != contexts[client]["folder"].resolve() or file.suffix not in {".npz", ".npy"}:
                raise ValueError(f"Volume must belong to its configured nnU-Net stage: {file}")
            if not file.exists() or not file.with_suffix(".pkl").exists():
                raise FileNotFoundError(f"Missing preprocessed volume/properties: {file}")
            groups[client, case] = split
            files.add(identity)
            rows.append(dict(client=client, case_id=case, split=split, path=file, source_key=source_key(file, root)))
    for client in CLIENTS:
        if not any(row["client"] == client for row in rows):
            raise ValueError(f"Manifest does not contain client {client}.")
    return rows


def partition_cases(cases, ratios, weights, rng):
    if len(cases) < len(ratios):
        raise ValueError("Not enough independent cases; supply a three-way split manifest.")
    order = list(cases)
    rng.shuffle(order)
    order.sort(key=lambda case: weights[case], reverse=True)
    target = np.asarray(ratios) * sum(weights[case] for case in cases)
    counts, allocation = np.zeros(len(ratios)), [[] for _ in ratios]
    for index, case in enumerate(order):
        empty = [i for i, items in enumerate(allocation) if not items]
        chosen = empty[0] if len(order) - index == len(empty) else int(np.argmax(target - counts))
        allocation[chosen].append(case)
        counts[chosen] += weights[case]
    return allocation


def make_manifest(args, contexts):
    root, records = data_root(args.data_root), []
    for client, context in contexts.items():
        files = {file.stem: file for file in sorted(context["folder"].glob("*.npz"))
                 if "segFromPrevStage" not in file.name}
        for file in sorted(context["folder"].glob("*.npy")):
            if "segFromPrevStage" not in file.name:
                files.setdefault(file.stem, file)
        if not files:
            raise ValueError(f"No preprocessed cases found in {context['folder']}.")
        weights = {}
        for key, file in files.items():
            properties = load_pickle(file.with_suffix(".pkl"))
            weights[key] = int(properties["original_size_of_raw_data"][0])
        rng = random.Random(args.seed + client)
        splits_file = context["task"] / "splits_final.pkl"
        if splits_file.exists():
            folds = load_pickle(splits_file)
            if not 0 <= args.fold < len(folds):
                raise ValueError(f"Fold {args.fold} is not available in {splits_file}.")
            selected = folds[args.fold]
            train_keys, val_keys = list(map(str, selected["train"])), list(map(str, selected["val"]))
            test_keys = list(map(str, selected.get("test", [])))
            used = train_keys + val_keys + test_keys
            if len(set(used)) != len(used) or set(used) != set(files):
                raise ValueError(f"Fold keys must cover each preprocessed case exactly once: {splits_file}")
            if test_keys:
                allocation = [train_keys, val_keys, test_keys]
            else:
                validation, testing = partition_cases(val_keys, (.25, .75), weights, rng)
                allocation = [train_keys, validation, testing]
        else:
            allocation = partition_cases(list(files), (.80, .05, .15), weights, rng)
        if not all(allocation):
            raise ValueError(f"Client {client} has an empty data split.")
        for split, keys in zip(("train", "val", "test"), allocation):
            records.extend(dict(client=client, case_id=key, split=split, path=source_key(files[key], root)) for key in keys)
        print(f"Client {client}: train/val/test cases = {[len(items) for items in allocation]}")
    target = Path(args.manifest)
    if target.exists():
        raise FileExistsError(f"Preserve existing manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client", "case_id", "split", "path"])
        writer.writeheader()
        writer.writerows(records)
    read_manifest(target, root, contexts)


def split_signature(rows):
    values = sorted((row["client"], row["case_id"], row["split"], row["source_key"]) for row in rows)
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def read_case(row, with_target=True):
    path = row["path"]
    unpacked = path.with_suffix(".npy")
    if unpacked.exists():
        array = np.load(unpacked, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            array = archive["data"]
    else:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 4 or array.shape[0] not in {1, 2} or (with_target and array.shape[0] != 2):
        raise ValueError(f"Expected nnU-Net [CT,segmentation] or image-only test data in {path}.")
    image = np.asarray(array[:1], dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite preprocessed image in {path}.")
    properties = load_pickle(path.with_suffix(".pkl"))
    if tuple(properties["size_after_resampling"]) != image.shape[1:]:
        raise ValueError(f"Preprocessed shape/properties disagree in {path}.")
    return image, np.asarray(array[1]) if with_target else None, properties


def remap_target(target, context):
    if not np.isfinite(target).all() or not np.array_equal(target, np.rint(target)):
        raise ValueError("Segmentation labels must be finite integers.")
    target = target.astype(np.int64)
    found = set(map(int, np.unique(target)))
    if not found <= set(context["allowed_labels"]) | {-1}:
        raise ValueError(f"Unrecognized source label IDs: {found - set(context['allowed_labels']) - {-1}}")
    result = np.zeros_like(target)
    result[target == -1] = -1
    for index, source in enumerate(context["label_ids"], 1):
        result[target == source] = index
    return result


def conditioning(client, properties):
    spacing = np.asarray(properties["spacing_after_resampling"], dtype=np.float32)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid preprocessed voxel spacing.")
    return np.asarray([float(client == key) for key in CLIENTS] + list(spacing / 5), dtype=np.float32)


def training_dataset(rows, client, context):
    dataset = OrderedDict()
    for row in rows:
        if row["client"] != client or row["split"] != "train":
            continue
        properties = deepcopy(load_pickle(row["path"].with_suffix(".pkl")))
        if "class_locations" not in properties:
            _, target, _ = read_case(row)
            properties["class_locations"] = {source: np.argwhere(target == source) for source in context["label_ids"]}
        properties["class_locations"] = {key: value for key, value in properties["class_locations"].items()
                                          if int(key) in context["label_ids"]}
        dataset[row["path"].stem] = dict(data_file=str(row["path"]), properties=properties,
                                          properties_file=str(row["path"].with_suffix(".pkl")))
    if not dataset:
        raise ValueError(f"Missing training data for client {client}.")
    return dataset


def augmentation(patch):
    return Compose([
        Convert3DTo2DTransform(),
        SpatialTransform(tuple(patch[1:]), do_elastic_deform=False, do_rotation=True,
                         angle_x=(-math.pi, math.pi), do_scale=True, scale=(.7, 1.4),
                         border_mode_data="constant", border_cval_data=0, order_data=3,
                         border_mode_seg="constant", border_cval_seg=-1, order_seg=0,
                         random_crop=False, p_scale_per_sample=.2, p_rot_per_sample=.2),
        Convert2DTo3DTransform(),
        GaussianNoiseTransform(p_per_sample=.1),
        GaussianBlurTransform((.5, 1.), different_sigma_per_channel=True, p_per_sample=.2, p_per_channel=.5),
        BrightnessMultiplicativeTransform((.75, 1.25), p_per_sample=.15),
        ContrastAugmentationTransform(p_per_sample=.15),
        SimulateLowResolutionTransform(zoom_range=(.5, 1.), per_channel=True, p_per_channel=.5,
                                      order_downsample=0, order_upsample=3, p_per_sample=.25, ignore_axes=(0,)),
        GammaTransform((.7, 1.5), True, True, retain_stats=True, p_per_sample=.1),
        GammaTransform((.7, 1.5), False, True, retain_stats=True, p_per_sample=.3),
        MirrorTransform((0, 1, 2)),
    ])


class GroupNorm3D(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(math.gcd(8, num_features), num_features, eps=eps, affine=affine)


def build_nnunet(config):
    return Generic_UNet(input_channels=1, base_num_features=config["width"], num_classes=7, num_pool=5,
                        num_conv_per_stage=2, conv_op=nn.Conv3d, norm_op=GroupNorm3D,
                        norm_op_kwargs=dict(eps=1e-5, affine=True), dropout_op=nn.Dropout3d,
                        dropout_op_kwargs=dict(p=0, inplace=True), nonlin=nn.LeakyReLU,
                        nonlin_kwargs=dict(negative_slope=.01, inplace=True), deep_supervision=True,
                        final_nonlin=nn.Identity(), weightInitializer=InitWeights_He(1e-2),
                        pool_op_kernel_sizes=[list(kernel) for kernel in POOL_KERNELS],
                        conv_kernel_sizes=[list(kernel) for kernel in CONV_KERNELS],
                        convolutional_pooling=True, convolutional_upsampling=True,
                        max_num_features=config["max_features"])


class Encoder(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.blocks = network.conv_blocks_context

    def forward(self, image):
        skips = []
        for block in self.blocks[:-1]:
            image = block(image)
            skips.append(image)
        return self.blocks[-1](image), tuple(skips)


class Decoder(nn.Module):
    def __init__(self, network, conditioned=False):
        super().__init__()
        self.up = network.tu
        self.blocks = network.conv_blocks_localization
        self.outputs = network.seg_outputs
        self.channels = [block[-1].output_channels for block in self.blocks]
        total = sum(self.channels)
        self.film = (nn.Sequential(nn.Linear(8, max(1, total // 2)), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(max(1, total // 2), total), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(total, 2 * total)) if conditioned else None)

    def forward(self, features, parameters=None):
        image, skips = features
        if self.film is not None:
            gamma, beta = self.film(parameters).chunk(2, dim=1)
        outputs, offset = [], 0
        for index, (up, block, head) in enumerate(zip(self.up, self.blocks, self.outputs)):
            image = block(torch.cat((up(image), skips[-index - 1]), dim=1))
            if self.film is not None:
                count = self.channels[index]
                image = (gamma[:, offset:offset + count, None, None, None] * image
                         + beta[:, offset:offset + count, None, None, None])
                offset += count
            outputs.append(head(image))
        return tuple(reversed(outputs))


class SegmentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        network = build_nnunet(config)
        self.encoder, self.decoder = Encoder(network), Decoder(network)

    def forward(self, image):
        features = self.encoder(image)
        return features, self.decoder(features)


def detached_features(features):
    image, skips = features
    return image.detach(), tuple(skip.detach() for skip in skips)


def dice_ce_loss(logits, target):
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0
    target = target.long()
    logits = logits.float()
    ce = F.cross_entropy(logits, target, ignore_index=-1, reduction="sum") / valid.sum()
    probability = logits.softmax(1) * valid[:, None]
    truth = F.one_hot(target.clamp_min(0), 7).movedim(-1, 1).float() * valid[:, None]
    axes = (2, 3, 4)
    dice = (2 * (probability * truth).sum(axes) + 1e-5) / (probability.sum(axes) + truth.sum(axes) + 1e-5)
    return ce + 1 - dice[valid.flatten(1).any(1), 1:].mean()


def directed_kl(student, teacher, target):
    valid = target >= 0
    if not valid.any():
        return student.sum() * 0
    log_student, log_teacher = student.float().log_softmax(1), teacher.detach().float().log_softmax(1)
    divergence = (log_student.exp() * (log_student - log_teacher)).sum(1)
    return (divergence * valid).sum() / valid.sum()


def mutual_losses(local, shared, target):
    weights = np.asarray([.5 ** index for index in range(len(local))])
    weights[-1] = 0
    weights /= weights.sum()
    local_loss, proxy_loss = local[0].new_zeros(()), shared[0].new_zeros(())
    for weight, left, right in zip(weights, local, shared):
        if weight == 0:
            continue
        scaled = F.interpolate(target[:, None].float(), size=left.shape[2:], mode="nearest")[:, 0].long()
        local_loss = local_loss + float(weight) * (dice_ce_loss(left, scaled) + directed_kl(left, right, scaled))
        proxy_loss = proxy_loss + float(weight) * (dice_ce_loss(right, scaled) + directed_kl(right, left, scaled))
    return local_loss, proxy_loss


def train_batch(model, proxy, local_optimizer, proxy_optimizer, batch, scaler, device, amp):
    image, target, parameters = (batch[key].to(device) for key in ("image", "target", "parameters"))
    local_optimizer.zero_grad(set_to_none=True)
    proxy_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp):
        features, local = model(image)
        shared = proxy(detached_features(features), parameters)
        local_loss, proxy_loss = mutual_losses(local, shared, target)
    if not torch.isfinite(local_loss + proxy_loss):
        raise FloatingPointError("Nonfinite segmentation training loss.")
    scaler.scale(local_loss).backward()
    scaler.scale(proxy_loss).backward()
    scaler.step(local_optimizer)
    scaler.step(proxy_optimizer)
    scaler.update()
    return local_loss.item(), proxy_loss.item()


@lru_cache(maxsize=8)
def gaussian_weights(patch):
    weights = np.zeros(patch, dtype=np.float32)
    weights[tuple(size // 2 for size in patch)] = 1
    weights = gaussian_filter(weights, [size / 8 for size in patch], mode="constant")
    return np.maximum(weights / weights.max(), 1e-7)


@torch.inference_mode()
def predict_volume(model, image, patch, device, step=.5, amp=False):
    model.eval()
    patch, shape = tuple(patch), image.shape[1:]
    padding = [(max(size - old, 0) // 2, max(size - old, 0) - max(size - old, 0) // 2)
               for size, old in zip(patch, shape)]
    padded = np.pad(image, [(0, 0)] + padding, mode="constant")
    axes = []
    for old, size in zip(padded.shape[1:], patch):
        count = math.ceil((old - size) / (size * step)) + 1
        axes.append([round(value) for value in np.linspace(0, old - size, count)])
    total = np.zeros((7, *padded.shape[1:]), dtype=np.float32)
    normalization = np.zeros(padded.shape[1:], dtype=np.float32)
    importance = gaussian_weights(patch)
    for starts in itertools.product(*axes):
        slices = tuple(slice(start, start + size) for start, size in zip(starts, patch))
        tile = torch.from_numpy(np.ascontiguousarray(padded[(slice(None), *slices)]))[None].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            _, outputs = model(tile)
        probability = outputs[0].float().softmax(1)[0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError("Nonfinite sliding-window prediction.")
        total[(slice(None), *slices)] += probability * importance
        normalization[slices] += importance
    total /= normalization[None]
    crop = tuple(slice(before, before + size) for size, (before, _) in zip(shape, padding))
    return total[(slice(None), *crop)]


def foreground_dice(prediction, target):
    valid = target >= 0
    histogram = np.bincount((7 * target[valid] + prediction[valid]).astype(np.int64), minlength=49).reshape(7, 7)
    denominator = histogram.sum(0) + histogram.sum(1)
    return np.divide(2 * histogram.diagonal()[1:], denominator[1:],
                     out=np.ones(6, dtype=float), where=denominator[1:] > 0)


def validate(model, rows, client, context, args, device, amp):
    values = []
    for row in rows:
        if row["client"] != client or row["split"] != "val":
            continue
        image, target, _ = read_case(row)
        probability = predict_volume(model, image, args.patch_size, device, args.step_size, amp)
        values.append(foreground_dice(probability.argmax(0), remap_target(target, context)))
    if not values:
        raise ValueError(f"Missing independent validation cases for client {client}.")
    classes = np.mean(values, axis=0)
    organs = np.asarray([(classes[0] + classes[1]) / 2, *classes[2:]])
    return dict(score=float(organs.mean()), class_dice=classes.tolist())


def model_config(args):
    return dict(width=args.width, max_features=args.max_features,
                pool_kernels=POOL_KERNELS, conv_kernels=CONV_KERNELS, organs=ORGANS)


def context_signature(contexts):
    return {client: dict(label_ids=context["label_ids"], transpose_backward=context["transpose_backward"],
                         stage=context["stage"], allowed_labels=context["allowed_labels"], plans_sha256=context["plans_sha256"])
            for client, context in contexts.items()}


def save_checkpoint(state, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def train(args, rows, contexts):
    output, device = Path(args.output), torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output already contains a run; use --resume or a new --output.")
    for client in CLIENTS:
        if not all(any(row["client"] == client and row["split"] == split for row in rows) for split in ("train", "val", "test")):
            raise ValueError(f"Client {client} requires independent train/val/test cases.")
    config, context_info = model_config(args), context_signature(contexts)
    models = {client: SegmentationModel(config).to(device) for client in CLIENTS}
    proxy = Decoder(build_nnunet(config), conditioned=True).to(device)
    optimizers = {client: torch.optim.Adam(model.parameters(), lr=args.lr) for client, model in models.items()}
    proxy_optimizers = {client: torch.optim.Adam(proxy.parameters(), lr=args.lr) for client in CLIENTS}
    schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=args.lr_patience)
                  for client, opt in optimizers.items()}
    proxy_schedulers = {client: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=args.lr_patience)
                        for client, opt in proxy_optimizers.items()}
    amp = device.type == "cuda" and not args.no_amp
    scalers = {client: torch.amp.GradScaler("cuda", enabled=amp) for client in CLIENTS}
    datasets = {client: training_dataset(rows, client, contexts[client]) for client in CLIENTS}
    basic_patch = tuple(args.patch_size) if args.no_augment else (
        args.patch_size[0], math.ceil(args.patch_size[1] * 225 / 192), math.ceil(args.patch_size[2] * 225 / 192))
    loaders = {client: DataLoader3D(dataset, basic_patch, args.patch_size, args.batch_size, False,
                                  oversample_foreground_percent=.33, pad_mode="constant", memmap_mode="r")
               for client, dataset in datasets.items()}
    transform = None if args.no_augment else augmentation(args.patch_size)
    first_round, best, retired, order = 1, {client: -1. for client in CLIENTS}, set(), list(CLIENTS)
    random.Random(args.seed).shuffle(order)
    if args.resume:
        state = load_checkpoint(args.resume)
        if state["model_config"] != config or state["context_signature"] != context_info or state["split_signature"] != split_signature(rows):
            raise ValueError("Resume requires the same model, preprocessing, labels, and subject split.")
        controlled = ("seed", "patch_size", "batch_size", "steps_per_epoch", "local_epochs", "extra_epochs",
                      "fixed_order", "target_dice", "no_withdraw", "no_augment", "no_amp", "lr", "lr_patience", "step_size")
        if any(state["args"][key] != getattr(args, key) for key in controlled):
            raise ValueError("Resume training settings differ from the saved run.")
        proxy.load_state_dict(state["proxy"], strict=True)
        for client in CLIENTS:
            models[client].load_state_dict(state["models"][client], strict=True)
            optimizers[client].load_state_dict(state["optimizers"][client])
            proxy_optimizers[client].load_state_dict(state["proxy_optimizers"][client])
            schedulers[client].load_state_dict(state["schedulers"][client])
            proxy_schedulers[client].load_state_dict(state["proxy_schedulers"][client])
            scalers[client].load_state_dict(state["scalers"][client])
            if not (output / "checkpoints" / f"client{client}_best.pt").exists():
                raise FileNotFoundError("Resume in the run directory containing its best checkpoints.")
        best, order, retired = state["best"], state["order"], set(state["retired"])
        first_round = state["round"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), model_config=config, clients=CLIENTS, preprocessing=context_info), handle, indent=2)
    for round_number in range(first_round, args.rounds + 1):
        order = [client for client in order if client not in retired]
        if not order:
            break
        used_order, losses = list(order), {}
        extra_client = order[-1] if round_number > 1 and not args.fixed_order else None
        print(f"Round {round_number}: order={order}, extra_client={extra_client}")
        for client in order:
            models[client].train()
            proxy.train()
            epochs = args.local_epochs + (args.extra_epochs if client == extra_client else 0)
            totals = np.zeros(2)
            for epoch in range(epochs):
                epoch_seed = (args.seed + 100000 * round_number + 1000 * client + epoch) % (2 ** 32)
                np.random.seed(epoch_seed)
                random.seed(epoch_seed)
                for _ in range(args.steps_per_epoch):
                    raw = next(loaders[client])
                    raw["seg"] = remap_target(raw["seg"], contexts[client]).astype(np.float32)
                    batch = transform(**raw) if transform is not None else raw
                    parameters = np.stack([conditioning(client, properties) for properties in raw["properties"]])
                    tensors = dict(image=torch.from_numpy(np.ascontiguousarray(batch["data"])),
                                   target=torch.from_numpy(np.ascontiguousarray(batch["seg"][:, 0])).long(),
                                   parameters=torch.from_numpy(parameters))
                    totals += train_batch(models[client], proxy, optimizers[client], proxy_optimizers[client],
                                          tensors, scalers[client], device, amp)
            count = epochs * args.steps_per_epoch
            losses[client] = dict(local=float(totals[0] / count), proxy=float(totals[1] / count))
        values = {}
        for client in CLIENTS:
            values[client] = validate(models[client], rows, client, contexts[client], args, device, amp)
            score = values[client]["score"]
            if client in order:
                schedulers[client].step(score)
                proxy_schedulers[client].step(score)
            if score > best[client]:
                best[client] = score
                save_checkpoint(dict(client=client, round=round_number, model_config=config,
                                     context_signature=context_info, split_signature=split_signature(rows),
                                     patch_size=list(args.patch_size),
                                     model=models[client].state_dict()), output / "checkpoints" / f"client{client}_best.pt")
            print(f"Client {client}: validation organ Dice={100 * score:.2f}%")
        if not args.fixed_order:
            if not args.no_withdraw:
                retired.update(client for client in CLIENTS if values[client]["score"] >= args.target_dice)
            order = sorted((client for client in CLIENTS if client not in retired),
                           key=lambda client: (-values[client]["score"], client))
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(round=round_number, order=used_order, extra_client=extra_client, training=losses,
                                         validation=values, next_order=order, retired=sorted(retired))) + "\n")
        save_checkpoint(dict(args=vars(args), model_config=config, context_signature=context_info,
                             split_signature=split_signature(rows), round=round_number,
                             models={c: model.state_dict() for c, model in models.items()}, proxy=proxy.state_dict(),
                             optimizers={c: opt.state_dict() for c, opt in optimizers.items()},
                             proxy_optimizers={c: opt.state_dict() for c, opt in proxy_optimizers.items()},
                             schedulers={c: scheduler.state_dict() for c, scheduler in schedulers.items()},
                             proxy_schedulers={c: scheduler.state_dict() for c, scheduler in proxy_schedulers.items()},
                             scalers={c: scaler.state_dict() for c, scaler in scalers.items()},
                             best=best, order=order, retired=sorted(retired), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), output / "last.pt")


def restore_label_ids(segmentation, ids):
    return np.asarray((0, *ids), dtype=np.uint8)[segmentation.astype(np.int64)]


def export_native(probability, path, properties, ids):
    with tempfile.TemporaryDirectory(prefix="cudircl_seg_") as temporary:
        native = Path(temporary) / "segmentation.nii.gz"
        save_segmentation_nifti_from_softmax(probability, str(native), deepcopy(properties),
                                            order=1, seg_postprogess_fn=restore_label_ids,
                                            seg_postprocess_args=(ids,), verbose=False)
        shutil.copyfile(native, path)


def safe_name(value):
    clean = "".join(char if char.isalnum() or char in "_-" else "_" for char in value)[:80]
    return clean or "case"


def test(args, rows, contexts):
    output, device = Path(args.output), torch.device(args.device)
    signature, preprocessing = split_signature(rows), context_signature(contexts)
    amp = device.type == "cuda" and not args.no_amp
    for client in list(dict.fromkeys(args.clients or CLIENTS)):
        state = load_checkpoint(output / "checkpoints" / f"client{client}_best.pt")
        if state["client"] != client or state["split_signature"] != signature or state["context_signature"] != preprocessing:
            raise ValueError("Test manifest/preprocessing does not match the training run.")
        model = SegmentationModel(state["model_config"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        cases = [row for row in rows if row["client"] == client and row["split"] == "test"]
        if not cases:
            raise ValueError(f"Missing independent test cases for client {client}.")
        context = contexts[client]
        folder = output / "results" / f"client{client}"
        folder.mkdir(parents=True, exist_ok=True)
        for row in cases:
            image, _, properties = read_case(row, with_target=False)
            required = {"crop_bbox", "size_after_cropping", "original_size_of_raw_data", "original_spacing",
                        "spacing_after_resampling", "itk_spacing", "itk_origin", "itk_direction"}
            if not required <= properties.keys():
                raise ValueError(f"Missing original-space export properties in {row['path']}.")
            probability = predict_volume(model, image, state["patch_size"], device, args.step_size, amp)
            token = hashlib.sha256(row["source_key"].encode()).hexdigest()[:10]
            name = f"{safe_name(row['case_id'])}_{token}"
            source_labels = restore_label_ids(probability.argmax(0), context["label_ids"])
            np.savez_compressed(folder / f"{name}.npz", segmentation=source_labels,
                                spacing=np.asarray(properties["spacing_after_resampling"], dtype=np.float32),
                                label_ids=np.asarray(context["label_ids"], dtype=np.uint8))
            native_probability = probability.transpose((0, *[axis + 1 for axis in context["transpose_backward"]]))
            export_native(native_probability, folder / f"{name}.nii.gz", properties, context["label_ids"])
            information = dict(client=client, case_id=row["case_id"], source_path=row["source_key"],
                               checkpoint_round=state["round"], label_map=dict(zip(ORGANS, context["label_ids"])),
                               npz_space="nnUNet_preprocessed", nifti_space="original", shape=list(image.shape[1:]),
                               spacing=np.asarray(properties["spacing_after_resampling"]).tolist(),
                               transpose_backward=context["transpose_backward"],
                               crop_bbox=np.asarray(properties["crop_bbox"]).tolist() if properties["crop_bbox"] is not None else None,
                               original_shape=np.asarray(properties["original_size_of_raw_data"]).tolist())
            with (folder / f"{name}.json").open("w", encoding="utf-8") as handle:
                json.dump(information, handle, indent=2)
        print(f"Client {client}: exported {len(cases)} complete segmentation volumes.")
        del model


def main():
    parser = argparse.ArgumentParser(description="CUDIRCL T3 segmentation using nnU-Net v1 preprocessed volumes and model")
    parser.add_argument("command", choices=("split", "train", "test"))
    parser.add_argument("--data-root", required=True, help="nnUNet_preprocessed or its parent directory")
    parser.add_argument("--manifest", required=True, help="CSV columns: client,case_id,split,path; clients 11,12,13,14,3")
    parser.add_argument("--output", default="ct_segmentation_run")
    parser.add_argument("--tasks", nargs=5, default=[profile["task"] for profile in CLIENTS.values()])
    parser.add_argument("--plans-name", default="nnUNetPlansv2.1_plans_3D.pkl")
    parser.add_argument("--stage", type=int, help="Default: highest-resolution stage in each client's plans")
    parser.add_argument("--fold", type=int, default=0, help="split only; a legacy held-out fold is separated into validation/test")
    parser.add_argument("--label-ids", nargs=6, type=int, help="Source IDs: left kidney,right kidney,liver,spleen,pancreas,stomach")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[48, 192, 192])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--extra-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--target-dice", type=float, default=1.)
    parser.add_argument("--step-size", type=float, default=.5, help="Sliding-window stride as a fraction of patch size")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixed-order", action="store_true")
    parser.add_argument("--no-withdraw", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", help="Resume from last.pt in the same run directory")
    parser.add_argument("--clients", nargs="+", type=int, choices=CLIENTS, help="Test client IDs; default all five")
    args = parser.parse_args()
    divisibility = np.prod(np.asarray(POOL_KERNELS), axis=0)
    if any(size < factor or size % factor for size, factor in zip(args.patch_size, divisibility)):
        parser.error(f"Patch dimensions must be multiples of {divisibility.tolist()}.")
    if min(args.width, args.max_features, args.batch_size, args.steps_per_epoch, args.rounds, args.local_epochs) < 1:
        parser.error("Network dimensions and training counts must be positive.")
    if args.max_features < args.width or min(args.extra_epochs, args.fold, args.lr_patience) < 0 or args.lr <= 0:
        parser.error("Invalid channel limit, count, or learning rate.")
    if not 0 < args.step_size <= 1 or not 0 < args.target_dice <= 1:
        parser.error("Sliding-window step and target Dice must be in (0,1].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    contexts = load_contexts(args)
    if args.command == "split":
        make_manifest(args, contexts)
    else:
        rows = read_manifest(args.manifest, data_root(args.data_root), contexts)
        (train if args.command == "train" else test)(args, rows, contexts)


if __name__ == "__main__":
    main()
