"""How much of the gap between pooled MBR (0.9175) and the pool oracle (0.950) can candidate SELECTION recover?
CPU only, on the finished EXP_002 GPU outputs (7B + 8B beam-5 lists of the 819 Fold-0 validation lines).

Every rule with a tuned knob is scored honestly: the knob is chosen on 4/5 of the lines and applied to the
held-out 1/5 (5-fold CV over lines, 3 seeds). Rules without knobs are scored directly.
  S0  pooled MBR, system weights 1/lost-points, T=1                              (current champion)
  S1  posterior temperature T                                                    (knob: T)
  S2  per-class system weights                                                   (knob: w8 per class)
  S3  character 6-gram LM prior on the evidence, trained on Fold-0 TRAIN labels   (knob: beta)
  S4  out-of-vocabulary penalty on the chosen line (vocabulary = Fold-0 TRAIN)   (knob: gamma)
  S5  MBR over an EXTENDED hypothesis space: greedy word-level median search     (no knob)
      (start from the MBR pick, apply single-word edits taken from other candidates while the
       expected official cost under the pooled posterior keeps falling)
  S6  S5 on top of the CV-chosen S1/S3 settings
Also: where the oracle candidate sits when MBR misses it.
usage: python work/selection_experiments.py [trils]
"""
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(__file__))
from road_eval import cost, norm, official, paired_delta  # noqa: E402

ROOT = "D:/HANAFY/ROAD"
S = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/trils"
M7, M8 = "qwen2_5_vl_transformers_7b_instruct_2", "qwen_3_vl_transformers_8b_instruct_1"
OUT = f"{ROOT}/work/selection_experiments.json"

fa = pd.read_csv(f"{ROOT}/work/fold_assignments.csv")
train_txt = [norm(t) for t in fa[fa.fold != 0].Target]
F = pd.read_csv(f"{ROOT}/work/image_features_plus.csv")[["ID", "height", "bg_mean"]]
F["group"] = np.where(F.height > 150, "B", np.where(F.height <= 58, "A1", np.where(F.bg_mean <= 194, "A2", "A3")))
GRP = F.set_index("ID").group

a = pd.read_csv(f"{S}/EXP_002__{M7}__D2_b5nb__val.csv")
b = pd.read_csv(f"{S}/EXP_002__{M8}__D2_b5nb__val.csv").set_index("ID").loc[a.ID].reset_index()
ids = a.ID.tolist()
gt = [norm(t) for t in a.GroundTruth]
grp = np.array([GRP.get(i, "A1") for i in ids])
cls = np.array([g[0] for g in grp])
N = len(ids)
NB7 = [[(norm(t), float(s)) for t, s in json.loads(x)] for x in a.nbest]
NB8 = [[(norm(t), float(s)) for t, s in json.loads(x)] for x in b.nbest]
print(f"lines {N} | classes {dict(collections.Counter(cls))}")

# ------------------------------------------------------------------ character 6-gram LM (Witten-Bell), Fold-0 TRAIN only
class CharLM:
    def __init__(self, texts, n=6):
        self.n = n
        self.c = [collections.Counter() for _ in range(n + 1)]      # c[k][(ctx, ch)] for context length k
        self.ctx = [collections.Counter() for _ in range(n + 1)]
        self.types = [collections.defaultdict(set) for _ in range(n + 1)]
        V = set()
        for t in texts:
            s = "\x02" * (n - 1) + t + "\x03"
            for i in range(n - 1, len(s)):
                ch = s[i]; V.add(ch)
                for k in range(n):
                    h = s[i - k:i]
                    self.c[k][(h, ch)] += 1; self.ctx[k][h] += 1; self.types[k][h].add(ch)
        self.V = len(V) + 1
        self.cache = {}

    def p(self, h, ch):
        key = (h, ch)
        if key in self.cache:
            return self.cache[key]
        pr = 1.0 / self.V
        for k in range(0, len(h) + 1):
            hk = h[len(h) - k:] if k else ""
            n_h = self.ctx[k].get(hk, 0)
            if n_h == 0:
                continue
            t_h = len(self.types[k][hk])
            lam = n_h / (n_h + t_h)
            pr = lam * self.c[k].get((hk, ch), 0) / n_h + (1 - lam) * pr
        self.cache[key] = pr
        return pr

    def logprob(self, t):
        n = self.n
        s = "\x02" * (n - 1) + t + "\x03"
        return sum(math.log(self.p(s[i - n + 1:i], s[i])) for i in range(n - 1, len(s)))


lm = CharLM(train_txt)
vocab = collections.Counter(w for t in train_txt for w in t.split())
dev = [lm.logprob(g) / (len(g) + 1) for g in gt[:200]]
print(f"char LM trained on {len(train_txt)} Fold-0 TRAIN lines | val GT per-char log-prob {np.mean(dev):.3f} "
      f"(perplexity {math.exp(-np.mean(dev)):.2f})")

# ------------------------------------------------------------------ per-line candidate tables
LINES = []
for i in range(N):
    cands = list(dict.fromkeys([t for t, _ in NB8[i]] + [t for t, _ in NB7[i]]))
    lp7 = {t: s for t, s in reversed(NB7[i])}; lp8 = {t: s for t, s in reversed(NB8[i])}
    C = np.array([[cost(c, h) for h in cands] for c in cands])     # C[e, h] = cost(evidence e, hypothesis h)
    LINES.append({
        "cands": cands, "C": C,
        "lp7": np.array([lp7.get(c, np.nan) for c in cands]), "lp8": np.array([lp8.get(c, np.nan) for c in cands]),
        "lm": np.array([lm.logprob(c) for c in cands]),
        "oov": np.array([sum(w not in vocab for w in c.split()) for c in cands], float),
        "true": np.array([cost(gt[i], c) for c in cands]),
    })


def post(lp, T):
    m = ~np.isnan(lp)
    out = np.zeros(len(lp))
    if m.any():
        s = lp[m] / T; s = s - s.max(); p = np.exp(s); out[m] = p / p.sum()
    return out


lost = {"7": 1 - official(gt, [x[0][0] for x in NB7])["score"], "8": 1 - official(gt, [x[0][0] for x in NB8])["score"]}
W8Q = (1 / lost["8"]) / (1 / lost["8"] + 1 / lost["7"])
print(f"quality weight of 8B (1/lost-points, fixed a priori): {W8Q:.3f}")


def weights(L, T=1.0, w8=W8Q, beta=0.0):
    w = w8 * post(L["lp8"], T) + (1 - w8) * post(L["lp7"], T)
    if beta:
        w = w * np.exp(beta * (L["lm"] - L["lm"].max()))
    return w / w.sum()


def pick_cost(L, w, gamma=0.0):
    risk = w @ L["C"] + gamma * L["oov"]
    return L["true"][int(np.argmin(risk))], int(np.argmin(risk))


def line_costs(**kw):
    gamma = kw.pop("gamma", 0.0)
    return np.array([pick_cost(L, weights(L, **kw), gamma)[0] for L in LINES])


def cv(grid_costs, seeds=(0, 1, 2), classwise=None):
    """grid_costs: {param: per-line cost array}. Choose the param with the lowest mean cost on 4 folds, apply to
    the 5th. classwise: array of class labels -> the param is chosen separately per class."""
    keys = list(grid_costs)
    M = np.stack([grid_costs[k] for k in keys])
    scores, chosen = [], collections.Counter()
    for sd in seeds:
        perm = np.random.RandomState(sd).permutation(N)
        fold = np.empty(N, int); fold[perm] = np.arange(N) % 5
        held = np.empty(N)
        for f in range(5):
            tr, te = fold != f, fold == f
            groups = [None] if classwise is None else sorted(set(classwise))
            for g in groups:
                gtr = tr if g is None else tr & (classwise == g)
                gte = te if g is None else te & (classwise == g)
                k = int(np.argmin(M[:, gtr].mean(1)))
                chosen[(g, keys[k])] += 1
                held[gte] = M[k, gte]
        scores.append(1 - held.mean())
        if sd == seeds[0]:
            held0 = held.copy()
    return float(np.mean(scores)), float(np.std(scores)), chosen.most_common(6), held0


base = line_costs()
R = {"S0_pooled_mbr_quality_T1": {"score": 1 - base.mean()}}
oracle = np.array([L["true"].min() for L in LINES])
print(f"\nS0 pooled MBR (quality weights, T=1): {1 - base.mean():.5f} | pool oracle {1 - oracle.mean():.5f} | "
      f"gap {base.mean() - oracle.mean():.5f}")


def report(name, costs_heldout, extra=""):
    diff = base - costs_heldout
    rng = np.random.default_rng(0)
    boots = diff[rng.integers(0, N, (5000, N))].mean(1)
    r = {"score": float(1 - costs_heldout.mean()), "delta_vs_S0": float(diff.mean()),
         "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
         "p_better": float((boots > 0).mean()), "improved": int((diff > 1e-12).sum()), "worsened": int((diff < -1e-12).sum())}
    R[name] = {**R.get(name, {}), **r}
    print(f"{name:42s} {r['score']:.5f}  delta {r['delta_vs_S0']:+.5f} CI95 [{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}] "
          f"P={r['p_better']:.3f}  +{r['improved']}/-{r['worsened']} {extra}")


# S1 temperature
TGRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
gT = {T: line_costs(T=T) for T in TGRID}
print("\nS1 temperature, in-sample (optimistic):", {T: round(1 - v.mean(), 5) for T, v in gT.items()})
m, s, ch, held = cv(gT)
report("S1_temperature_cv", held, f"| CV mean {m:.5f}±{s:.5f} chosen {ch[:3]}")

# S2 per-class system weight
WGRID = [0.3, 0.4, 0.5, W8Q, 0.6, 0.7, 0.8]
gW = {w: line_costs(w8=w) for w in WGRID}
print("\nS2 8B weight, in-sample:", {round(w, 3): round(1 - v.mean(), 5) for w, v in gW.items()})
for c in ("A", "B"):
    print(f"   class {c}:", {round(w, 3): round(1 - v[cls == c].mean(), 5) for w, v in gW.items()})
m, s, ch, held = cv(gW, classwise=cls)
report("S2_classwise_system_weight_cv", held, f"| CV mean {m:.5f}±{s:.5f} chosen {ch[:4]}")

# S3 char-LM prior on the evidence
BGRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
gB = {bt: line_costs(beta=bt) for bt in BGRID}
print("\nS3 char-LM beta, in-sample:", {bt: round(1 - v.mean(), 5) for bt, v in gB.items()})
m, s, ch, held = cv(gB)
report("S3_charLM_prior_cv", held, f"| CV mean {m:.5f}±{s:.5f} chosen {ch[:3]}")

# S4 OOV penalty on the hypothesis
GGRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
gG = {g: line_costs(gamma=g) for g in GGRID}
print("\nS4 OOV gamma, in-sample:", {g: round(1 - v.mean(), 5) for g, v in gG.items()})
m, s, ch, held = cv(gG)
report("S4_oov_penalty_cv", held, f"| CV mean {m:.5f}±{s:.5f} chosen {ch[:3]}")

# S1 x S3 joint
gTB = {(T, bt): line_costs(T=T, beta=bt) for T in (0.5, 1.0, 1.5, 2.0, 3.0) for bt in (0.0, 0.1, 0.2, 0.3, 0.5)}
m, s, ch, held_TB = cv(gTB)
report("S1xS3_joint_cv", held_TB, f"| CV mean {m:.5f}±{s:.5f} chosen {ch[:3]}")


# S5 extended hypothesis space: greedy word-level median search
def exp_cost(h, ev):
    return sum(w * cost(c, h) for c, w in ev)


def median_search(start, ev, max_iter=6):
    h = start.split(); best = exp_cost(start, ev)
    for _ in range(max_iter):
        props = set()
        for c, _w in ev:
            cw = c.split()
            for op in Levenshtein.editops(h, cw):
                if op.tag == "replace":
                    nh = h[:op.src_pos] + [cw[op.dest_pos]] + h[op.src_pos + 1:]
                elif op.tag == "delete":
                    nh = h[:op.src_pos] + h[op.src_pos + 1:]
                else:
                    nh = h[:op.src_pos] + [cw[op.dest_pos]] + h[op.src_pos:]
                props.add(" ".join(nh))
        props.discard(" ".join(h))
        if not props:
            break
        cand = min(props, key=lambda p: exp_cost(p, ev))
        cst = exp_cost(cand, ev)
        if cst < best - 1e-12:
            h, best = cand.split(), cst
        else:
            break
    return " ".join(h)


def run_median(**kw):
    out, costs, changed = [], [], 0
    for i, L in enumerate(LINES):
        w = weights(L, **kw)
        _, k = pick_cost(L, w)
        ev = [(c, float(x)) for c, x in zip(L["cands"], w) if x > 1e-9]
        h = median_search(L["cands"][k], ev)
        changed += h != L["cands"][k]
        out.append(h); costs.append(cost(gt[i], h))
    return out, np.array(costs), changed


med_hyp, med_cost, n_changed = run_median()
report("S5_median_search_T1", med_cost, f"| lines changed {n_changed} | outputs not in the pool: "
       f"{sum(h not in L['cands'] for h, L in zip(med_hyp, LINES))}")
for T in (1.5, 2.0):
    _, mc, nc = run_median(T=T)
    report(f"S5_median_search_T{T}", mc, f"| lines changed {nc} (fixed T, in-sample)")

# ------------------------------------------------------------------ where does the oracle sit when MBR misses it?
miss = [(i, L) for i, L in enumerate(LINES) if L["true"].min() < base[i] - 1e-12]
rk = collections.Counter()
for i, L in miss:
    o = L["cands"][int(np.argmin(L["true"]))]
    r8 = next((r for r, (t, _) in enumerate(NB8[i]) if t == o), None)
    r7 = next((r for r, (t, _) in enumerate(NB7[i]) if t == o), None)
    rk[("8B#" + str(r8 + 1) if r8 is not None else "8B-") + "/" + ("7B#" + str(r7 + 1) if r7 is not None else "7B-")] += 1
gap = base - oracle
print(f"\nMBR misses the pool's best candidate on {len(miss)} of {N} lines; recoverable points {gap.mean():.5f}")
print("  gap share by class:", {c: round(float(gap[cls == c].sum() / gap.sum()), 3) for c in ("A", "B")},
      "| by group:", {g: round(float(gap[grp == g].sum() / gap.sum()), 3) for g in ("A1", "A2", "A3", "B")})
print("  rank of the oracle candidate (8B rank / 7B rank):", rk.most_common(10))
wc = collections.Counter()
for i, L in miss:
    o = L["cands"][int(np.argmin(L["true"]))]; p = L["cands"][int(np.argmin(weights(L) @ L["C"]))]
    ow, pw = o.split(), p.split()
    for op in Levenshtein.editops(pw, ow):
        if op.tag == "replace":
            x, y = pw[op.src_pos], ow[op.dest_pos]
            wc["casing" if x.lower() == y.lower() else "punct/markup" if "".join(filter(str.isalnum, x)) == "".join(filter(str.isalnum, y)) else "spelling"] += 1
        else:
            wc["word " + op.tag] += 1
print("  what differs between MBR pick and oracle pick (word ops):", dict(wc))
R["gap"] = {"pool_oracle": float(1 - oracle.mean()), "lines_missed": len(miss), "oracle_rank": dict(rk), "diff_types": dict(wc)}
json.dump(R, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
