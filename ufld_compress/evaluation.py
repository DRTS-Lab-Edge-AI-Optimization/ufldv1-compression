"""TuSimple evaluation (Accuracy / FP / FN / F1) for any runner + agreement with the teacher."""
from __future__ import annotations

import time

import numpy as np

from .lane_utils import LaneEval, logits_to_lanes


def evaluate(runner, loader, dataset, cfg, reference_argmax: dict | None = None,
             collect_argmax: bool = False, max_batches: int | None = None, progress: bool = True) -> dict:
    """Run the model on the TuSimple test loader and compute the official metrics.

    reference_argmax: {image_index: argmax(rows, lanes)} of the teacher -> 'teacher_agreement'
                      (fraction of row/lane cells whose predicted grid cell is identical to the teacher's).
    """
    it = loader
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(loader, desc="eval", leave=False)
        except ImportError:
            pass
    acc = fp = fn = 0.0
    n = 0
    agree, n_agree = 0.0, 0
    argmaxes = {}
    t_model = 0.0
    for bi, (x, idx) in enumerate(it):
        if max_batches is not None and bi >= max_batches:
            break
        t0 = time.perf_counter()
        out = runner(x)
        t_model += time.perf_counter() - t0
        for b in range(out.shape[0]):
            i = int(idx[b])
            gt = dataset.items[i]
            lanes = logits_to_lanes(out[b], cfg.griding_num, cfg.img_w)
            a, p, q = LaneEval.bench(lanes, gt["lanes"], gt["h_samples"], 10)
            acc, fp, fn, n = acc + a, fp + p, fn + q, n + 1
            if collect_argmax or reference_argmax is not None:
                am = out[b].argmax(0).astype(np.int16)
                if collect_argmax:
                    argmaxes[i] = am
                if reference_argmax is not None and i in reference_argmax:
                    agree += float((am == reference_argmax[i]).mean())
                    n_agree += 1
    n = max(n, 1)
    res = {"accuracy": acc / n, "fp": fp / n, "fn": fn / n, "n_images": n,
           "eval_ms_per_image": 1000 * t_model / n}
    prec, rec = 1 - res["fp"], 1 - res["fn"]
    res["f1"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    if n_agree:
        res["teacher_agreement"] = agree / n_agree
    if collect_argmax:
        res["argmax"] = argmaxes
    return res
