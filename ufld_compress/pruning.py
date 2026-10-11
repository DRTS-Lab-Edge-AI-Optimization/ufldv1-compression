"""Pruning methods.

1. unstructured_prune   - one-shot magnitude (or random) pruning, global or layer-wise.
                          Smaller compressed storage; no dense-CPU speed-up (zeros are still multiplied).
2. AGPPruner            - gradual magnitude pruning (Zhu & Gupta 2017, cubic schedule) during (KD) fine-tuning.
3. channel_prune        - structured channel/filter pruning that physically removes channels:
                          * block-internal channels of every BasicBlock (conv1 -> conv2)
                          * optionally the residual (stage) channels, shared by all blocks of a stage
                          * the 2048-unit hidden layer of the FC head (holds most of the parameters)
                          importance: l1 | l2 | bn_gamma (network slimming) | taylor (first-order) | random
                          -> real FLOPs/latency/memory reduction on any CPU, then recover with KD.
4. recalibrate_bn       - re-estimate BN statistics after pruning (cheap, often recovers a lot of accuracy).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .model import ResNetBackbone

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def prunable_layers(model, skip=(), include_fc: bool = True, include_conv: bool = True):
    out = []
    for name, m in model.named_modules():
        if name in skip:
            continue
        if (include_conv and isinstance(m, nn.Conv2d)) or (include_fc and isinstance(m, nn.Linear)):
            out.append((name, m))
    return out


def _current_weight(m):
    if hasattr(m, "weight_orig"):
        return (m.weight_orig * m.weight_mask).detach()
    return m.weight.detach()


def _apply_mask(m, mask):
    if hasattr(m, "weight_mask"):
        m.weight_mask.data.copy_(mask)
        m.weight = m.weight_orig * m.weight_mask  # refresh immediately (also recomputed by the hook)
    else:
        prune.custom_from_mask(m, "weight", mask)


def make_pruning_permanent(model):
    for m in model.modules():
        if hasattr(m, "weight_orig") and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")
    return model


def sparsity_report(model) -> dict:
    rep, tot, zeros = {}, 0, 0
    for name, m in prunable_layers(model):
        w = _current_weight(m)
        z = int((w == 0).sum())
        rep[name] = z / w.numel()
        tot += w.numel()
        zeros += z
    rep["__global__"] = zeros / max(tot, 1)
    return rep


def _global_threshold(weights, sparsity):
    scores = torch.cat([w.abs().flatten() for w in weights])
    k = int(round(sparsity * scores.numel()))
    if k <= 0:
        return None
    return torch.kthvalue(scores, k).values


# ---------------------------------------------------------------------------------------------
# 1) unstructured one-shot
# ---------------------------------------------------------------------------------------------


def unstructured_prune(model, amount: float, scope: str = "global", method: str = "l1", skip=(),
                       include_fc: bool = True, include_conv: bool = True, permanent: bool = True,
                       inplace: bool = False):
    """scope: 'global' (one threshold for all layers) or 'layer' (same sparsity in every layer).
    method: 'l1' (magnitude) or 'random' (baseline)."""
    if not inplace:
        model = copy.deepcopy(model)
    layers = [m for _, m in prunable_layers(model, skip, include_fc, include_conv)]
    if method == "random":
        for m in layers:
            prune.random_unstructured(m, "weight", amount=amount)
    elif scope == "global":
        th = _global_threshold([_current_weight(m) for m in layers], amount)
        if th is not None:
            for m in layers:
                _apply_mask(m, (_current_weight(m).abs() > th).float())
    else:
        for m in layers:
            w = _current_weight(m)
            th = _global_threshold([w], amount)
            if th is not None:
                _apply_mask(m, (w.abs() > th).float())
    if permanent:
        make_pruning_permanent(model)
    return model


# ---------------------------------------------------------------------------------------------
# 2) AGP gradual pruning (use as a Trainer step callback)
# ---------------------------------------------------------------------------------------------


class AGPPruner:
    def __init__(self, model, final_sparsity: float, start_step: int = 0, end_step: int = 100,
                 frequency: int = 10, initial_sparsity: float = 0.0, skip=(), include_fc: bool = True,
                 include_conv: bool = True):
        self.layers = [m for _, m in prunable_layers(model, skip, include_fc, include_conv)]
        self.sf, self.si = final_sparsity, initial_sparsity
        self.t0, self.t1, self.freq = start_step, max(end_step, start_step + 1), frequency
        self.current = 0.0
        self.history = []

    def target(self, step: int) -> float:
        if step < self.t0:
            return 0.0
        t = min(1.0, (step - self.t0) / (self.t1 - self.t0))
        return self.sf + (self.si - self.sf) * (1 - t) ** 3

    @torch.no_grad()
    def apply(self, sparsity: float):
        weights = [_current_weight(m) for m in self.layers]
        th = _global_threshold(weights, sparsity)
        if th is None:
            return
        for m, w in zip(self.layers, weights):
            _apply_mask(m, (w.abs() > th).float())
        self.current = sparsity

    def __call__(self, model, step: int):
        if step < self.t0:
            return
        if (step - self.t0) % self.freq == 0 or (step >= self.t1 and self.current < self.sf):
            s = self.target(step)
            if s > self.current + 1e-6:
                self.apply(s)
                self.history.append((step, s))


# ---------------------------------------------------------------------------------------------
# 3) structured channel pruning
# ---------------------------------------------------------------------------------------------


@dataclass
class ChannelGroup:
    name: str
    size: int
    outs: list = field(default_factory=list)   # (module, kind): 'conv_out' | 'bn' | 'linear_out'
    ins: list = field(default_factory=list)    # (module, kind): 'conv_in' | 'linear_in'
    kind: str = "conv"                          # 'conv' | 'fc'


def resnet_ufld_groups(model, prune_internal: bool = True, prune_residual: bool = False,
                       prune_fc: bool = True) -> list:
    bb = model.model
    if not isinstance(bb, ResNetBackbone):
        raise NotImplementedError("channel_prune supports the ResNet backbones (the UFLD teacher)")
    layers = [bb.layer1, bb.layer2, bb.layer3, bb.layer4]
    groups = []
    if prune_internal:
        for li, layer in enumerate(layers, start=1):
            for bi, blk in enumerate(layer):
                groups.append(ChannelGroup(f"model.layer{li}.{bi}.conv1", blk.conv1.out_channels,
                                           [(blk.conv1, "conv_out"), (blk.bn1, "bn")], [(blk.conv2, "conv_in")]))
    if prune_residual:
        for li, layer in enumerate(layers):
            outs, ins = [], []
            if li == 0:  # the stem feeds the identity path of layer1
                outs += [(bb.conv1, "conv_out"), (bb.bn1, "bn")]
            for bi, blk in enumerate(layer):
                outs += [(blk.conv2, "conv_out"), (blk.bn2, "bn")]
                if blk.downsample is not None:
                    outs += [(blk.downsample[0], "conv_out"), (blk.downsample[1], "bn")]
                if bi > 0 or li == 0:
                    ins.append((blk.conv1, "conv_in"))
            if li + 1 < len(layers):
                nxt = layers[li + 1][0]
                ins.append((nxt.conv1, "conv_in"))
                if nxt.downsample is not None:
                    ins.append((nxt.downsample[0], "conv_in"))
            else:
                ins.append((model.pool, "conv_in"))
            size = layer[-1].conv2.out_channels
            groups.append(ChannelGroup(f"model.stage{li + 1}.residual", size, outs, ins))
    if prune_fc:
        fc0, fc2 = model.cls[0], model.cls[2]
        groups.append(ChannelGroup("cls.hidden", fc0.out_features, [(fc0, "linear_out")], [(fc2, "linear_in")],
                                   kind="fc"))
    return groups


def _slice_scores(t, kind):
    """Per-channel score of a tensor slice (sum of |x| along all dims except the channel dim)."""
    if kind in ("conv_out", "linear_out", "bn"):
        return t.reshape(t.shape[0], -1).sum(1)
    return t.transpose(0, 1).reshape(t.shape[1], -1).sum(1)  # *_in


def group_importance(g: ChannelGroup, criterion: str = "l1", taylor: dict | None = None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    if criterion == "random":
        return torch.rand(g.size, generator=generator)
    parts = []
    for mod, kind in g.outs + g.ins:
        if criterion == "bn_gamma":
            if kind == "bn":
                parts.append(mod.weight.detach().abs())
            continue
        if criterion == "taylor":
            if taylor is None:
                raise ValueError("taylor scores required (compute_taylor_scores)")
            names = ["weight", "bias"] if kind in ("bn", "linear_out") else ["weight"]
            for pn in names:
                p = getattr(mod, pn, None)
                if p is not None and id(p) in taylor:
                    parts.append(_slice_scores(taylor[id(p)], kind))
            continue
        if kind == "bn":
            continue
        w = mod.weight.detach()
        parts.append(_slice_scores(w.abs() if criterion == "l1" else w.pow(2), kind))
    if not parts:  # e.g. bn_gamma on the FC group -> fall back to L2
        return group_importance(g, "l2")
    # normalise each part so coupled layers contribute equally, then average
    parts = [p / (p.mean() + 1e-12) for p in parts]
    return torch.stack(parts).mean(0)


def compute_taylor_scores(model, batches, loss_fn, n_batches: int = 8) -> dict:
    """Accumulated (w * dL/dw)^2 per parameter (Molchanov et al. 2019). loss_fn(model, batch) -> scalar."""
    model.eval()  # keep BN statistics fixed
    params = list(model.parameters())
    req = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    scores = {id(p): torch.zeros_like(p) for p in params}
    for i, batch in enumerate(batches):
        if i >= n_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        for p in params:
            if p.grad is not None:
                scores[id(p)] += (p.detach() * p.grad.detach()).pow(2)
    model.zero_grad(set_to_none=True)
    for p, r in zip(params, req):
        p.requires_grad_(r)
    return scores


def pseudo_label_loss(model, batch, teacher=None, temperature: float = 1.0):
    """Label-free loss for Taylor importance: KD to the teacher if given, else CE to own argmax."""
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    out = model(x)
    if teacher is not None:
        with torch.no_grad():
            t = teacher(x)
        return nn.functional.kl_div(nn.functional.log_softmax(out / temperature, 1),
                                    nn.functional.softmax(t / temperature, 1), reduction="batchmean")
    return nn.functional.cross_entropy(out, out.argmax(1).detach())


def _n_keep(size, ratio, round_to, min_keep_frac=0.1):
    n = size * (1 - ratio)
    n = int(math.ceil(n / round_to) * round_to) if round_to > 1 else int(round(n))
    lo = max(round_to, int(math.ceil(size * min_keep_frac)))
    return int(min(size, max(lo, n)))


@torch.no_grad()
def _apply_keep(g: ChannelGroup, keep: torch.Tensor):
    k = len(keep)
    for mod, kind in g.outs:
        if kind in ("conv_out", "linear_out"):
            mod.weight = nn.Parameter(mod.weight.data.index_select(0, keep).clone())
            if mod.bias is not None:
                mod.bias = nn.Parameter(mod.bias.data.index_select(0, keep).clone())
            if kind == "conv_out":
                mod.out_channels = k
            else:
                mod.out_features = k
        elif kind == "bn":
            mod.weight = nn.Parameter(mod.weight.data.index_select(0, keep).clone())
            mod.bias = nn.Parameter(mod.bias.data.index_select(0, keep).clone())
            mod.running_mean = mod.running_mean.index_select(0, keep).clone()
            mod.running_var = mod.running_var.index_select(0, keep).clone()
            mod.num_features = k
    for mod, kind in g.ins:
        if kind == "conv_in":
            assert mod.groups == 1
            mod.weight = nn.Parameter(mod.weight.data.index_select(1, keep).clone())
            mod.in_channels = k
        else:
            mod.weight = nn.Parameter(mod.weight.data.index_select(1, keep).clone())
            mod.in_features = k


def channel_prune(model, ratio: float = 0.3, criterion: str = "l1", prune_internal: bool = True,
                  prune_residual: bool = False, fc_ratio: float | None = None, strategy: str = "uniform",
                  layer_ratios: dict | None = None, round_to: int = 8, min_keep_frac: float = 0.1,
                  taylor_scores: dict | None = None, seed: int = 0, inplace: bool = False,
                  input_shape=(1, 3, 288, 800)):
    """Structured pruning that physically shrinks the network.

    ratio        fraction of channels removed from each conv group (uniform) / overall (global)
    fc_ratio     fraction of the 2048 hidden FC units removed (None -> same as ratio, 0 -> keep FC)
    strategy     'uniform' | 'global' (one threshold on normalised importance across conv groups)
    layer_ratios {group-name-prefix: ratio} overrides, e.g. {'model.layer4': 0.6, 'model.layer1': 0.1}
    round_to     keep channel counts multiples of 8 (SIMD friendly on AVX2 / NEON)
    taylor_scores from compute_taylor_scores(model, ...) when criterion='taylor' (same model object!)
    """
    if any(hasattr(m, "weight_orig") for m in model.modules()):
        raise ValueError("model has unstructured pruning masks; call make_pruning_permanent() first")
    if criterion == "taylor" and inplace is False:
        # taylor scores are keyed by parameter identity -> compute on `model` and prune a copy: remap ids
        src_params = list(model.parameters())
        model = copy.deepcopy(model)
        if taylor_scores is not None:
            taylor_scores = {id(p_new): taylor_scores[id(p_old)]
                             for p_old, p_new in zip(src_params, model.parameters()) if id(p_old) in taylor_scores}
    elif not inplace:
        model = copy.deepcopy(model)
    fc_ratio = ratio if fc_ratio is None else fc_ratio
    groups = resnet_ufld_groups(model, prune_internal, prune_residual, prune_fc=fc_ratio > 0)
    gen = torch.Generator().manual_seed(seed)
    imps = {g.name: group_importance(g, criterion, taylor_scores, gen) for g in groups}

    def ratio_for(g):
        if g.kind == "fc":
            return fc_ratio
        for prefix, r in (layer_ratios or {}).items():
            if g.name.startswith(prefix):
                return r
        return ratio

    keep_counts = {g.name: _n_keep(g.size, ratio_for(g), round_to, min_keep_frac) for g in groups}
    if strategy == "global":
        conv = [g for g in groups if g.kind == "conv"]
        allv = torch.cat([imps[g.name] for g in conv])
        k = int(round(ratio * allv.numel()))
        if k > 0:
            th = torch.kthvalue(allv, k).values
            for g in conv:
                n_above = int((imps[g.name] > th).sum())
                keep_counts[g.name] = _n_keep(g.size, 1 - n_above / g.size, round_to, min_keep_frac)

    report = {}
    for g in groups:
        k = keep_counts[g.name]
        keep = torch.sort(torch.topk(imps[g.name], k).indices).values
        _apply_keep(g, keep)
        report[g.name] = (g.size, k)

    with torch.no_grad():  # sanity check
        model.eval()(torch.randn(*input_shape))
    model.prune_report = report
    return model


# ---------------------------------------------------------------------------------------------
# 4) BN recalibration
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def recalibrate_bn(model, batches, n_batches: int = 20):
    """Reset BN running stats and re-estimate them (cumulative average) on unlabeled images."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model
    moms = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    model.train()
    for i, x in enumerate(batches):
        if i >= n_batches:
            break
        model(x)
    for bn, mom in zip(bns, moms):
        bn.momentum = mom
    return model.eval()
