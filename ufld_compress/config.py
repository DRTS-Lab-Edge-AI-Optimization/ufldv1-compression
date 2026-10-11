"""Experiment configuration for UFLD-v1 / TuSimple compression experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

# Row anchors (in the 288-px resized image) used by UFLD for TuSimple: 64, 68, ..., 284 (56 rows).
TUSIMPLE_ROW_ANCHOR = list(range(64, 288, 4))
# TuSimple evaluation rows in the original 720-px image: 160, 170, ..., 710 (56 rows).
TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))
TUSIMPLE_TRAIN_JSONS = ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json")


@dataclass
class Config:
    # ---- paths -----------------------------------------------------------------------------
    data_root: str = "data/tusimple"          # folder with clips/ and label_data_*.json
    test_root: Optional[str] = None           # folder with clips/ + test_label.json (defaults to data_root)
    checkpoint: str = "weights/tusimple_18.pth"
    output_dir: str = "outputs"
    train_jsons: Tuple[str, ...] = TUSIMPLE_TRAIN_JSONS
    test_json: str = "test_label.json"

    # ---- model (must match the released checkpoint) -----------------------------------------
    backbone: str = "18"
    griding_num: int = 100
    num_lanes: int = 4
    input_h: int = 288
    input_w: int = 800
    img_w: int = 1280
    img_h: int = 720

    # ---- runtime -----------------------------------------------------------------------------
    num_threads: int = 6        # Ryzen 5 5500U: 6 physical cores
    num_workers: int = 2        # DataLoader workers (they compete with compute threads)
    seed: int = 0
    dry_run: bool = False       # synthetic data + random weights: checks the pipeline end-to-end

    @property
    def row_anchor(self):
        return TUSIMPLE_ROW_ANCHOR

    @property
    def h_samples(self):
        return TUSIMPLE_H_SAMPLES

    @property
    def num_row_anchors(self) -> int:
        return len(TUSIMPLE_ROW_ANCHOR)

    @property
    def cls_dim(self) -> Tuple[int, int, int]:
        return (self.griding_num + 1, self.num_row_anchors, self.num_lanes)

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        return (1, 3, self.input_h, self.input_w)

    @property
    def test_dir(self) -> str:
        return self.test_root or self.data_root

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cls_dim"] = list(self.cls_dim)
        return d
