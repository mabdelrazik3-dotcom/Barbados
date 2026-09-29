"""CRNN line reader with CTC: training, greedy / prefix-beam decoding, candidate scoring.

The network reads a grayscale line rescaled to a fixed height: a VGG-style CNN (width / 4,
height pooled away) feeds a 2-layer BiLSTM whose per-column outputs go through CTC.
Its two roles in approach B:
    - generator: the beam-search n-best ("CTC beam-8 strings") join the candidate pool;
    - reader:    the CTC negative log-likelihood of every pooled candidate is a ranker feature.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ..config import Cfg
from ..metric import line_edits
from ..text.clean import ALLOWED
from ..utils import log, set_seed


# ------------------------------------------------------------------ charset
class Charset:
    """Index 0 is the CTC blank."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.index = {c: i + 1 for i, c in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> tuple[list[int], int]:
        ids = [self.index[c] for c in text if c in self.index]
        return ids, len(text) - len(ids)

    def decode(self, ids) -> str:
        return "".join(self.chars[i - 1] for i in ids if 0 < i <= len(self.chars))

    @classmethod
    def from_texts(cls, texts) -> "Charset":
        chars = set(ALLOWED)
        for t in texts:
            chars.update(t)
        return cls(chars)


# ------------------------------------------------------------------ images
def load_line(path: str, height: int, max_width: int, augment: bool = False, rng: random.Random | None = None):
    img = ImageOps.autocontrast(Image.open(path).convert("L"), cutoff=1)
    if augment and rng is not None:
        img = _augment(img, rng)
    w, h = img.size
    new_w = max(16, min(max_width, round(w * height / h)))
    img = img.resize((new_w, height), Image.BILINEAR)
    arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0  # ink -> high values, background -> 0
    return arr


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    shear = rng.uniform(-0.25, 0.25)
    scale_x = rng.uniform(0.9, 1.1)
    img = img.transform((int(w * scale_x), h), Image.AFFINE, (1 / scale_x, shear, -shear * h / 2, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        lo, hi = rng.uniform(0, 40), rng.uniform(215, 255)
        img = img.point(lambda v: max(0, min(255, int(lo + v * (hi - lo) / 255))))
    return img


# ------------------------------------------------------------------ model
def build_model(n_classes: int, hidden: int):
    import torch.nn as nn

    def block(cin, cout, pool):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        return layers

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                *block(1, 64, (2, 2)), *block(64, 128, (2, 2)),
                *block(128, 256, None), *block(256, 256, (2, 1)),
                *block(256, 384, None), *block(384, 384, (2, 1)),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            self.rnn = nn.LSTM(384, hidden, num_layers=2, bidirectional=True, dropout=0.2)
            self.fc = nn.Linear(2 * hidden, n_classes)

        def forward(self, x):                       # x: [B, 1, H, W]
            f = self.cnn(x).squeeze(2).permute(2, 0, 1)  # [T, B, 384], T = W // 4
            out, _ = self.rnn(f)
            return self.fc(out)                     # [T, B, C]

    return CRNN()


def out_len(width: int) -> int:
    return (width // 2) // 2


def _collate(batch):
    import torch

    width = max(a.shape[1] for a, _ in batch)
    height = batch[0][0].shape[0]
    x = torch.zeros(len(batch), 1, height, width)
    targets, tlens, ilens = [], [], []
    for i, (arr, ids) in enumerate(batch):
        x[i, 0, :, : arr.shape[1]] = torch.from_numpy(arr)
        targets.extend(ids)
        tlens.append(len(ids))
        ilens.append(out_len(arr.shape[1]))
    return x, torch.tensor(targets, dtype=torch.long), torch.tensor(tlens), torch.tensor(ilens)


class _LineSet:
    def __init__(self, paths, texts, charset, height, max_width, augment, seed):
        self.paths, self.texts, self.charset = list(paths), list(texts), charset
        self.height, self.max_width, self.augment = height, max_width, augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = load_line(self.paths[i], self.height, self.max_width, self.augment, self.rng)
        ids, _ = self.charset.encode(" ".join(self.texts[i].split()))
        return arr, ids


# ------------------------------------------------------------------ training
def train_crnn(cfg: Cfg, rows, out_path: str | Path, seed: int, force: bool = False) -> Path:
    import torch
    from torch.utils.data import DataLoader

    out_path = Path(out_path)
    if out_path.exists() and not force:
        log.info("crnn exists: %s", out_path)
        return out_path
    c = cfg.approach_b.ctc
    set_seed(seed, bool(cfg.deterministic))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    charset = Charset.from_texts(rows.Target)
    groups = rows.group.unique()
    rng = np.random.default_rng(seed)
    valid_groups = set(rng.choice(groups, size=max(1, int(len(groups) * float(c.valid_frac))), replace=False))
    is_val = rows.group.isin(valid_groups).values
    tr, va = rows[~is_val], rows[is_val]
    mk = lambda df, aug: _LineSet(df.image_path, df.Target, charset, int(c.height), int(c.max_width), aug, seed)
    dl_tr = DataLoader(mk(tr, bool(c.augment)), batch_size=int(c.batch_size), shuffle=True,
                       num_workers=int(c.num_workers), collate_fn=_collate)
    dl_va = DataLoader(mk(va, False), batch_size=int(c.batch_size), shuffle=False,
                       num_workers=int(c.num_workers), collate_fn=_collate)
    model = build_model(charset.size, int(c.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    epochs = int(c.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(c.lr), total_steps=max(1, epochs * len(dl_tr)))
    ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    best = math.inf
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, tgt, tlen, ilen in dl_tr:
            x = x.to(device)
            logp = model(x).log_softmax(-1).float()
            loss = ctc(logp, tgt.to(device), ilen.to(device), tlen.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss)
        cer = _valid_cer(model, dl_va, charset, device)
        log.info("crnn epoch %d/%d: train ctc %.3f, valid char edits/line %.2f", epoch + 1, epochs,
                 total / max(1, len(dl_tr)), cer)
        if cer < best:
            best = cer
            torch.save({"state": model.state_dict(), "chars": charset.chars, "hidden": int(c.hidden),
                        "height": int(c.height), "max_width": int(c.max_width)}, out_path)
    return out_path


def _valid_cer(model, dl, charset, device) -> float:
    import torch

    model.eval()
    edits, n = 0.0, 0
    with torch.no_grad():
        for x, tgt, tlen, ilen in dl:
            logp = model(x.to(device)).log_softmax(-1).cpu().numpy()
            offset = 0
            for b in range(x.shape[0]):
                ref = charset.decode(tgt[offset : offset + int(tlen[b])].tolist())
                offset += int(tlen[b])
                hyp = greedy_decode(logp[: int(ilen[b]), b], charset)
                edits += line_edits(ref, hyp)[1]
                n += 1
    return edits / max(n, 1)


# ------------------------------------------------------------------ inference
class CTCReader:
    def __init__(self, path: str | Path):
        import torch

        ckpt = torch.load(str(path), map_location="cpu")
        self.torch = torch
        self.charset = Charset(ckpt["chars"])
        self.height, self.max_width = ckpt["height"], ckpt["max_width"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(self.charset.size, ckpt["hidden"])
        self.model.load_state_dict(ckpt["state"])
        self.model.to(self.device).eval()

    def log_probs(self, path: str) -> np.ndarray:
        """[T, C] log-probabilities for one line image."""
        arr = load_line(path, self.height, self.max_width)
        x = self.torch.from_numpy(arr)[None, None].to(self.device)
        with self.torch.no_grad():
            return self.model(x).log_softmax(-1)[:, 0].float().cpu().numpy()

    def score(self, logp: np.ndarray, texts: list[str]) -> list[dict]:
        """CTC negative log-likelihood of each text given the line's log-probs."""
        import torch.nn.functional as F

        torch = self.torch
        t = logp.shape[0]
        out = []
        lp = torch.from_numpy(logp)[:, None, :]
        for text in texts:
            ids, unknown = self.charset.encode(" ".join(text.split()))
            if len(ids) > t:
                out.append({"nll": 1e4, "unknown": unknown, "n": len(ids)})
                continue
            nll = F.ctc_loss(lp, torch.tensor([ids], dtype=torch.long), torch.tensor([t]), torch.tensor([len(ids)]),
                             blank=0, reduction="sum", zero_infinity=True)
            out.append({"nll": float(nll), "unknown": unknown, "n": len(ids)})
        return out


def greedy_decode(logp: np.ndarray, charset: Charset) -> str:
    best = logp.argmax(-1)
    ids, prev = [], 0
    for i in best:
        if i != prev and i != 0:
            ids.append(int(i))
        prev = i
    return charset.decode(ids)


def prefix_beam_search(logp: np.ndarray, charset: Charset, beam_width: int = 8, prune: float = 1e-4,
                       lm=None, lm_weight: float = 0.0) -> list[tuple[str, float]]:
    """CTC prefix beam search in log space, with optional character-LM shallow fusion."""
    neg = -math.inf
    lse = np.logaddexp
    log_prune = math.log(prune)
    beams: dict[tuple, list[float]] = {(): [0.0, neg]}  # prefix -> [log p(blank end), log p(non-blank end)]
    lm_cache: dict[tuple, float] = {}

    def lm_bonus(prefix: tuple, c: int) -> float:
        if lm is None or lm_weight == 0:
            return 0.0
        key = (prefix[-(lm.order - 1):], c)
        if key not in lm_cache:
            ctx = charset.decode(list(key[0]))
            lm_cache[key] = lm.char_logprob(charset.chars[c - 1], ctx)
        return lm_weight * lm_cache[key]

    for t in range(logp.shape[0]):
        row = logp[t]
        chars = [int(c) for c in np.nonzero(row > log_prune)[0] if c != 0]
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [neg, neg])
        for prefix, (pb, pnb) in beams.items():
            total = lse(pb, pnb)
            nb = nxt[prefix]
            nb[0] = lse(nb[0], total + row[0])
            for c in chars:
                p = row[c]
                new = prefix + (c,)
                if prefix and prefix[-1] == c:
                    n = nxt[new]
                    n[1] = lse(n[1], pb + p + lm_bonus(prefix, c))
                    s = nxt[prefix]
                    s[1] = lse(s[1], pnb + p)
                else:
                    n = nxt[new]
                    n[1] = lse(n[1], total + p + lm_bonus(prefix, c))
        beams = dict(sorted(nxt.items(), key=lambda kv: -lse(*kv[1]))[:beam_width])
    ranked = sorted(beams.items(), key=lambda kv: -lse(*kv[1]))
    return [(charset.decode(list(p)), float(lse(*s))) for p, s in ranked]
