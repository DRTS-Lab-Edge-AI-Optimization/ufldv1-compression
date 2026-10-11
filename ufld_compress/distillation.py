"""Knowledge distillation for UFLD + a CPU-friendly Trainer (also used for pruning recovery and QAT).

KD terms (combine freely through TrainConfig weights):
  * logit KD      - KL(softmax(t/T) || softmax(s/T)) * T^2 over the (griding_num+1) cells of every row/lane
  * location KD   - L1 between the expected lane x-position of student and teacher (UFLD-specific,
                    only on rows where the teacher sees a lane)
  * feature KD    - FitNets: MSE between 1x1-adapted student features and teacher features (layer2/3/4)
  * attention KD  - Attention Transfer (Zagoruyko & Komodakis): channel-pooled spatial attention maps
Hard-label terms (when labels are available): UFLD focal loss (gamma=2) + row relation (similarity) loss.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------------------


def focal_loss(logits, labels, gamma: float = 2.0):
    """UFLD SoftmaxFocalLoss: logits (N,C,R,L), labels (N,R,L)."""
    log_p = F.log_softmax(logits, dim=1)
    factor = torch.pow(1.0 - log_p.exp(), gamma)
    return F.nll_loss(factor * log_p, labels)


def relation_loss(logits):
    """UFLD ParsingRelationLoss: neighbouring row anchors should have similar distributions."""
    d = logits[:, :, :-1, :] - logits[:, :, 1:, :]
    return F.smooth_l1_loss(d, torch.zeros_like(d))


def logit_kd_loss(s, t, temperature: float = 4.0):
    log_ps = F.log_softmax(s / temperature, dim=1)
    pt = F.softmax(t / temperature, dim=1)
    kl = F.kl_div(log_ps, pt, reduction="none").sum(1)  # per (N,R,L) cell
    return kl.mean() * temperature ** 2


def location_kd_loss(s, t):
    """L1 between expected grid locations (normalised to [0,1]) where the teacher predicts a lane."""
    g = s.shape[1] - 1
    idx = torch.arange(g, device=s.device, dtype=s.dtype).view(1, -1, 1, 1)
    loc_s = (F.softmax(s[:, :g], 1) * idx).sum(1) / g
    loc_t = (F.softmax(t[:, :g], 1) * idx).sum(1) / g
    mask = (t.argmax(1) != g).float()
    return (mask * (loc_s - loc_t).abs()).sum() / mask.sum().clamp(min=1.0)


def _norm_feat(f):
    n = f.shape[0]
    flat = f.reshape(n, -1)
    return ((flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-6)).reshape_as(f)


def attention_map(f, p: int = 2):
    return F.normalize(f.pow(p).mean(1).flatten(1), dim=1)


def at_loss(fs, ft):
    if fs.shape[-2:] != ft.shape[-2:]:
        fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
    return (attention_map(fs) - attention_map(ft)).pow(2).mean()


class FeatureHooks:
    """Capture outputs of named sub-modules (e.g. 'model.layer2')."""

    def __init__(self, model, names):
        mods = dict(model.named_modules())
        missing = [n for n in names if n not in mods]
        if missing:
            raise KeyError(f"modules {missing} not found (feature KD is not supported on FX/QAT graphs - "
                           f"use logit KD there)")
        self.names = list(names)
        self.feats = {}
        self.handles = [mods[n].register_forward_hook(self._hook(n)) for n in self.names]

    def _hook(self, name):
        def fn(m, i, o):
            self.feats[name] = o
        return fn

    def get(self):
        return [self.feats[n] for n in self.names]

    def remove(self):
        for h in self.handles:
            h.remove()


class FeatureAdapters(nn.Module):
    def __init__(self, s_channels, t_channels):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Conv2d(cs, ct, 1, bias=False) for cs, ct in zip(s_channels, t_channels)])

    def forward(self, feats):
        return [a(f) for a, f in zip(self.adapters, feats)]


# ---------------------------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    max_steps: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 20
    hard_weight: float = 1.0          # focal loss with ground-truth labels (0 if no labels)
    relation_weight: float = 1.0      # UFLD sim_loss_w
    kd_weight: float = 0.0            # logit KD
    temperature: float = 4.0
    loc_weight: float = 0.0           # location KD
    feat_weight: float = 0.0          # FitNets feature KD
    at_weight: float = 0.0            # attention transfer (typ. 100-1000)
    feat_layers: tuple = ("model.layer2", "model.layer3", "model.layer4")
    freeze_bn: bool = True            # small CPU batches -> keep BN statistics fixed
    grad_clip: float = 5.0
    log_every: int = 10
    num_threads: int | None = None

    def to_dict(self):
        return asdict(self)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class Trainer:
    """Fine-tune `student` (optionally against `teacher`) for a fixed number of steps on the CPU.

    callbacks: list of fn(model, step) called after every optimizer step (AGP pruning, QAT freezing ...)
    """

    def __init__(self, student, train_loader, cfg: TrainConfig, teacher=None, tb_logdir: str | None = None,
                 tracker=None, callbacks=(), metric_prefix: str = "train/"):
        self.s, self.t, self.loader, self.cfg = student, teacher, train_loader, cfg
        self.callbacks = list(callbacks)
        self.tracker = tracker
        self.prefix = metric_prefix
        self.writer = None
        if tb_logdir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(tb_logdir)
            except ImportError:
                pass
        needs_teacher = cfg.kd_weight or cfg.loc_weight or cfg.feat_weight or cfg.at_weight
        if needs_teacher and teacher is None:
            raise ValueError("KD weights > 0 require a teacher")
        for p in student.parameters():  # copies of a frozen teacher inherit requires_grad=False
            p.requires_grad_(True)
        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        self.adapters = None
        self.s_hooks = self.t_hooks = None
        if cfg.feat_weight or cfg.at_weight:
            self.s_hooks = FeatureHooks(student, cfg.feat_layers)
            self.t_hooks = FeatureHooks(teacher, cfg.feat_layers)
            if cfg.feat_weight:
                x = torch.randn(1, 3, 288, 800)
                with torch.no_grad():
                    student.eval()(x)
                    teacher(x)
                sc = [f.shape[1] for f in self.s_hooks.get()]
                tc = [f.shape[1] for f in self.t_hooks.get()]
                self.adapters = FeatureAdapters(sc, tc)
        params = [p for p in student.parameters() if p.requires_grad]
        if self.adapters is not None:
            params += list(self.adapters.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return (step + 1) / max(cfg.warmup_steps, 1)
            p = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.history = []

    def _losses(self, x, y):
        cfg = self.cfg
        out = self.s(x)
        L = {}
        if y is not None and cfg.hard_weight:
            L["hard"] = cfg.hard_weight * focal_loss(out, y)
        if cfg.relation_weight:
            L["relation"] = cfg.relation_weight * relation_loss(out)
        if self.t is not None:
            with torch.no_grad():
                t_out = self.t(x)
            if cfg.kd_weight:
                L["kd"] = cfg.kd_weight * logit_kd_loss(out, t_out, cfg.temperature)
            if cfg.loc_weight:
                L["loc"] = cfg.loc_weight * location_kd_loss(out, t_out)
            if self.s_hooks is not None:
                fs, ft = self.s_hooks.get(), [f.detach() for f in self.t_hooks.get()]
                if cfg.feat_weight:
                    fa = self.adapters(fs)
                    L["feat"] = cfg.feat_weight * sum(F.mse_loss(_norm_feat(a), _norm_feat(b)) for a, b in zip(fa, ft)) / len(ft)
                if cfg.at_weight:
                    L["at"] = cfg.at_weight * sum(at_loss(a, b) for a, b in zip(fs, ft))
        return L

    def fit(self):
        cfg = self.cfg
        if cfg.num_threads:
            torch.set_num_threads(cfg.num_threads)
        it = _cycle(self.loader)
        try:
            from tqdm import tqdm

            bar = tqdm(range(cfg.max_steps), desc="train", leave=False)
        except ImportError:
            bar = range(cfg.max_steps)
        t_last = time.perf_counter()
        for step in bar:
            self.s.train()
            if cfg.freeze_bn:
                _set_bn_eval(self.s)
            batch = next(it)
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) and batch[1].dim() == 3 else None
            else:
                x, y = batch, None
            L = self._losses(x, y)
            loss = sum(L.values())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], cfg.grad_clip)
            self.opt.step()
            self.sched.step()
            for cb in self.callbacks:
                cb(self.s, step + 1)
            rec = {k: float(v.detach()) for k, v in L.items()}
            rec["loss"] = float(loss.detach())
            rec["lr"] = self.sched.get_last_lr()[0]
            self.history.append(rec)
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.max_steps:
                now = time.perf_counter()
                rec["sec_per_step"] = (now - t_last) / cfg.log_every
                t_last = now
                avg = {k: sum(h.get(k, 0.0) for h in self.history[-cfg.log_every:]) / min(cfg.log_every, len(self.history))
                       for k in rec if k not in ("lr", "sec_per_step")}
                avg.update(lr=rec["lr"], sec_per_step=rec["sec_per_step"])
                if self.writer is not None:
                    for k, v in avg.items():
                        self.writer.add_scalar(f"{self.prefix}{k}", v, step + 1)
                if self.tracker is not None:
                    self.tracker.log_metrics({f"{self.prefix}{k}": v for k, v in avg.items()}, step=step + 1)
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(loss=f"{avg['loss']:.4f}")
        self.close()
        self.s.eval()
        return self.history

    def close(self):
        for h in (self.s_hooks, self.t_hooks):
            if h is not None:
                h.remove()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
