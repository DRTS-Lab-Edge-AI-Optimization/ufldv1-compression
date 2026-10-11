"""TuSimple datasets.

Training labels are produced on-the-fly from `label_data_*.json` with the exact logic of the original
`scripts/convert_tusimple.py` + `LaneClsDataset`, so you do NOT need to run the conversion script.

Expected layout (same as the official UFLD instructions):

    data_root/
        clips/...
        label_data_0313.json  label_data_0531.json  label_data_0601.json
        test_label.json        (or point cfg.test_root to the test-set folder)
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .lane_utils import get_lane_index, grid_points, render_seg_label

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def image_transform(size=(288, 800)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


def _read_json_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def subset_indices(n_total: int, n: int | None, seed: int = 0, shuffle: bool = True):
    idx = list(range(n_total))
    if n is None or n >= n_total:
        return idx
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    return sorted(idx[:n])


# ---- augmentation (same as mytransforms.RandomRotate / RandomUDoffsetLABEL / RandomLROffsetLABEL) -----

def _rotate(img, label, max_angle=6):
    angle = random.randint(0, max_angle * 2) - max_angle
    return img.rotate(angle, resample=Image.BILINEAR), label.rotate(angle, resample=Image.NEAREST)


def _shift(img, label, max_offset, axis):
    offset = np.random.randint(-max_offset, max_offset)
    img = np.array(img)
    label = np.array(label)
    for arr in (img, label):
        n = arr.shape[axis]
        sl = [slice(None)] * arr.ndim
        if offset > 0:
            dst, src, zero = slice(offset, None), slice(0, n - offset), slice(0, offset)
        elif offset < 0:
            o = -offset
            dst, src, zero = slice(0, n - o), slice(o, None), slice(n - o, None)
        else:
            continue
        d, s, z = list(sl), list(sl), list(sl)
        d[axis], s[axis], z[axis] = dst, src, zero
        arr[tuple(d)] = arr[tuple(s)].copy()
        arr[tuple(z)] = 0
    return Image.fromarray(img), Image.fromarray(label)


def _worker_init(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)
    random.seed(torch.initial_seed() + worker_id)


# ---------------------------------------------------------------------------------------------


class TuSimpleTrainDataset(Dataset):
    """Returns (image_tensor, cls_label[rows, lanes]) - the UFLD classification target."""

    def __init__(self, cfg, subset: int | None = None, augment: bool = True, seed: int = 0):
        self.cfg = cfg
        self.root = cfg.data_root
        self.items = []
        for j in cfg.train_jsons:
            self.items += _read_json_lines(os.path.join(self.root, j))
        self.indices = subset_indices(len(self.items), subset, seed)
        self.augment = augment
        self.transform = image_transform((cfg.input_h, cfg.input_w))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        item = self.items[self.indices[k]]
        img = Image.open(os.path.join(self.root, item["raw_file"])).convert("RGB")
        w, h = img.size
        label = Image.fromarray(render_seg_label(item["lanes"], item["h_samples"], h, w))
        if self.augment:
            img, label = _rotate(img, label, 6)
            img, label = _shift(img, label, 100, axis=0)   # up/down
            img, label = _shift(img, label, 200, axis=1)   # left/right
        pts = get_lane_index(np.asarray(label), self.cfg.row_anchor, self.cfg.num_lanes)
        cls_label = grid_points(pts, self.cfg.griding_num, w)
        return self.transform(img), torch.from_numpy(cls_label).long()


class TuSimpleTestDataset(Dataset):
    """Returns (image_tensor, global_index); ground truth is in `self.items[global_index]`."""

    def __init__(self, cfg, subset: int | None = None, seed: int = 0):
        self.cfg = cfg
        self.root = cfg.test_dir
        self.items = _read_json_lines(os.path.join(self.root, cfg.test_json))
        self.indices = subset_indices(len(self.items), subset, seed)
        self.transform = image_transform((cfg.input_h, cfg.input_w))

    def __len__(self):
        return len(self.indices)

    def image_path(self, i):
        return os.path.join(self.root, self.items[i]["raw_file"])

    def __getitem__(self, k):
        i = self.indices[k]
        img = Image.open(self.image_path(i)).convert("RGB")
        return self.transform(img), i


# ---- synthetic data: runs the whole pipeline without the dataset (dry run) -----------------------


class SyntheticTrainDataset(Dataset):
    def __init__(self, cfg, n: int = 64):
        self.cfg, self.n = cfg, n

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        g = torch.Generator().manual_seed(k)
        x = torch.randn(3, self.cfg.input_h, self.cfg.input_w, generator=g)
        y = torch.randint(0, self.cfg.griding_num + 1, (self.cfg.num_row_anchors, self.cfg.num_lanes), generator=g)
        return x, y


class SyntheticTestDataset(Dataset):
    def __init__(self, cfg, n: int = 32):
        self.cfg, self.n = cfg, n
        rng = np.random.RandomState(0)
        hs = cfg.h_samples
        self.items = []
        for i in range(n):
            lanes = []
            for _ in range(rng.randint(2, 5)):
                x0, slope = rng.uniform(100, 1180), rng.uniform(-1.5, 1.5)
                lanes.append([int(x0 + slope * (y - 400)) if 0 <= x0 + slope * (y - 400) < 1280 else -2 for y in hs])
            self.items.append({"raw_file": f"synthetic/{i}.jpg", "lanes": lanes, "h_samples": hs})
        self.indices = list(range(n))

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        g = torch.Generator().manual_seed(10_000 + k)
        return torch.randn(3, self.cfg.input_h, self.cfg.input_w, generator=g), k


# ---------------------------------------------------------------------------------------------


def make_loader(ds, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0, drop_last: bool = False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, worker_init_fn=_worker_init,
                      persistent_workers=num_workers > 0, pin_memory=False)


def build_datasets(cfg, train_subset=None, test_subset=None, augment=True):
    """(train_ds, test_ds). In dry-run mode synthetic data is returned."""
    if cfg.dry_run:
        return SyntheticTrainDataset(cfg, train_subset or 64), SyntheticTestDataset(cfg, test_subset or 32)
    return (TuSimpleTrainDataset(cfg, train_subset, augment, cfg.seed),
            TuSimpleTestDataset(cfg, test_subset, cfg.seed))


def images_only(loader, max_batches=None):
    """Yield only the image tensors of a loader (for calibration / unlabeled distillation)."""
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch[0] if isinstance(batch, (list, tuple)) else batch
