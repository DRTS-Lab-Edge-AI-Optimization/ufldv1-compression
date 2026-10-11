"""NumPy-only lane utilities (no PyTorch import, so they also run on the Raspberry Pi).

* `logits_to_lanes`     - UFLD post-processing exactly as in evaluation/eval_wrapper.py (generate_tusimple_lines)
* `LaneEval`            - TuSimple benchmark (evaluation/tusimple/lane.py) without the sklearn dependency
* `render_seg_label`    - lane ordering + segmentation label, as in scripts/convert_tusimple.py (done in memory)
* `get_lane_index` / `grid_points` - classification targets, as in data/dataset.py (LaneClsDataset)
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def expected_locations(out: np.ndarray, griding_num: int) -> np.ndarray:
    """out: (griding_num+1, rows, lanes) logits -> (rows, lanes) float grid locations; `griding_num` = no lane."""
    out_loc = np.argmax(out, axis=0)
    prob = softmax(out[:-1], axis=0)
    idx = np.arange(griding_num, dtype=np.float64).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    loc[out_loc == griding_num] = griding_num
    return loc


def logits_to_lanes(out: np.ndarray, griding_num: int = 100, img_w: int = 1280) -> list:
    """TuSimple format: one list of x coordinates per lane (len = rows), -2 where the lane is absent."""
    loc = expected_locations(out, griding_num)
    lanes = []
    for i in range(loc.shape[1]):
        lanes.append([int(round((l + 0.5) * float(img_w) / (griding_num - 1))) if l != griding_num else -2
                      for l in loc[:, i]])
    return lanes


def lanes_to_points(lanes, h_samples):
    """Convert TuSimple lanes into lists of (x, y) points for drawing."""
    return [[(x, y) for x, y in zip(lane, h_samples) if x >= 0] for lane in lanes]


# --------------------------------------------------------------------------------------------
# TuSimple benchmark (same maths as evaluation/tusimple/lane.py)
# --------------------------------------------------------------------------------------------


class LaneEval:
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            k = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 1)[0]  # == sklearn LinearRegression slope
            return np.arctan(k)
        return 0.0

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time=10):
        if any(len(p) != len(y_samples) for p in pred):
            raise ValueError("Format of lanes error.")
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fn = 0.0
        matched = 0.0
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return (s / max(min(4.0, len(gt)), 1.0),
                fp / len(pred) if len(pred) > 0 else 0.0,
                fn / max(min(len(gt), 4.0), 1.0))


# --------------------------------------------------------------------------------------------
# Training labels (identical logic to convert_tusimple.py + LaneClsDataset, without writing PNGs)
# --------------------------------------------------------------------------------------------


def _calc_k(line):
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 90:
        return -10  # too short -> skipped
    p = np.polyfit(line_x, line_y, deg=1)
    return np.arctan(p[0])


def _draw(im, line, idx):
    import cv2

    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]), int(line_y[0]))
    for i in range(len(line_x) - 1):
        pt1 = (int(line_x[i + 1]), int(line_y[i + 1]))
        cv2.line(im, pt0, pt1, (idx,), thickness=16)
        pt0 = pt1


def render_seg_label(lanes, h_samples, img_h=720, img_w=1280) -> np.ndarray:
    """Lanes ordered left->right as 1..4 (CULane convention) exactly like convert_tusimple.py."""
    h_samples = np.asarray(h_samples)
    lines = []
    for lane in lanes:
        lane = np.asarray(lane)
        if np.all(lane == -2):
            continue
        valid = lane != -2
        line = np.empty(2 * int(valid.sum()), dtype=np.float64)
        line[::2] = lane[valid]
        line[1::2] = h_samples[valid]
        lines.append(line)

    label = np.zeros((img_h, img_w), dtype=np.uint8)
    if not lines:
        return label
    ks = np.array([_calc_k(line) for line in lines])
    k_neg = ks[ks < 0].copy()
    k_pos = ks[ks > 0].copy()
    k_neg = k_neg[k_neg != -10]
    k_pos = k_pos[k_pos != -10]
    k_neg.sort()
    k_pos.sort()

    def which(k):
        return np.where(ks == k)[0][0]

    if len(k_neg) == 1:
        _draw(label, lines[which(k_neg[0])], 2)
    elif len(k_neg) >= 2:  # == 2 and > 2 are handled identically in the original script
        _draw(label, lines[which(k_neg[1])], 1)
        _draw(label, lines[which(k_neg[0])], 2)

    if len(k_pos) == 1:
        _draw(label, lines[which(k_pos[0])], 3)
    elif len(k_pos) == 2:
        _draw(label, lines[which(k_pos[1])], 3)
        _draw(label, lines[which(k_pos[0])], 4)
    elif len(k_pos) > 2:
        _draw(label, lines[which(k_pos[-1])], 3)
        _draw(label, lines[which(k_pos[-2])], 4)
    return label


def _find_start_pos(row_sample, start_line):
    l, r = 0, len(row_sample) - 1
    while True:
        mid = int((l + r) / 2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid


def get_lane_index(label: np.ndarray, row_anchor, num_lanes: int = 4) -> np.ndarray:
    """Lane x-positions at the row anchors (+ linear extension to the image bottom), from a seg label."""
    h, w = label.shape[:2]
    sample_tmp = [int((r * 1.0 / 288) * h) for r in sorted(row_anchor)]
    all_idx = np.zeros((num_lanes, len(sample_tmp), 2))
    for i, r in enumerate(sample_tmp):
        label_r = label[int(round(r))]
        for lane_idx in range(1, num_lanes + 1):
            pos = np.where(label_r == lane_idx)[0]
            all_idx[lane_idx - 1, i, 0] = r
            all_idx[lane_idx - 1, i, 1] = np.mean(pos) if len(pos) else -1

    all_idx_cp = all_idx.copy()
    for i in range(num_lanes):
        if np.all(all_idx_cp[i, :, 1] == -1):
            continue
        valid = all_idx_cp[i, :, 1] != -1
        valid_idx = all_idx_cp[i, valid, :]
        if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
            continue  # lane already reaches the bottom row
        if len(valid_idx) < 6:
            continue
        valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
        p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
        start_line = valid_idx_half[-1, 0]
        pos = _find_start_pos(all_idx_cp[i, :, 0], start_line) + 1
        fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
        fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])
        all_idx_cp[i, pos:, 1] = fitted
    return all_idx_cp


def grid_points(pts: np.ndarray, num_cols: int, w: int) -> np.ndarray:
    """(lanes, rows, 2) -> (rows, lanes) int class labels; num_cols = 'no lane'."""
    num_lane, n, _ = pts.shape
    col_sample = np.linspace(0, w - 1, num_cols)
    step = col_sample[1] - col_sample[0]
    to_pts = np.zeros((n, num_lane))
    for i in range(num_lane):
        to_pts[:, i] = [int(pt // step) if pt != -1 else num_cols for pt in pts[i, :, 1]]
    return to_pts.astype(int)
