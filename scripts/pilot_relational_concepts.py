#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relational concept pilot：静态概念 vs 关系型概念。

不训练 g，不调用 LLM。只使用已有的 train/validation predictions：

  static        : 21 个 A-E 概念
  indep         : 21 concepts + 3 stance proxies + 4 target probs
  rel_stance    : indep + concept × stance
  rel_target    : rel_stance + concept × target
  rel_full      : rel_target + concept × use × target

stance proxy（来自已有概念，不是新标注）：
  quote  = E01（报告/引用）
  oppose = E03（反对）
  mention = 1 - (1-quote)(1-oppose)
  use = 1 - mention

target probs：T01-T04（T05 排除）。

评估：
  teacher concepts -> independent CBM 的概念上限
  predicted concepts -> 实际 g(x) 管线

输出：
  results/relational_pilot/<timestamp>/
    metrics_by_run.csv
    metrics_aggregate.csv
    intervention.csv
    report.txt
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

CBM_DIR = Path(__file__).resolve().parent / (
    "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0"
)
OUTPUTS = CBM_DIR / "outputs" / "independent_true_seed20261122"
TRAIN_PRED = OUTPUTS / "train_predictions.csv"
VAL_PRED = OUTPUTS / "validation_predictions.csv"

A_E = [
    "A01", "A02", "A03",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04",
    "D01", "D02",
    "E01", "E02", "E03", "E04", "E05",
]
T4 = ["T01", "T02", "T03", "T04"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_predictions", type=Path, default=TRAIN_PRED)
    p.add_argument("--val_predictions", type=Path, default=VAL_PRED)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="默认 results/relational_pilot/<timestamp>",
    )
    p.add_argument("--seeds", type=str, default="729,1136,1196")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--concept_source",
        choices=["teacher", "predicted", "both"],
        default="both",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = (
        ["id", "toxic_true"]
        + [f"concept_target_{c}" for c in A_E]
        + [f"predicted_concept_{c}" for c in A_E]
    )
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺少列：{miss}")
    return df


def concept_matrix(df: pd.DataFrame, source: str) -> np.ndarray:
    prefix = "concept_target" if source == "teacher" else "predicted_concept"
    return df[[f"{prefix}_{c}" for c in A_E]].to_numpy(np.float32)


def target_matrix(df: pd.DataFrame, source: str) -> np.ndarray:
    prefix = "concept_target" if source == "teacher" else "predicted_concept"
    return df[[f"{prefix}_{c}" for c in T4]].to_numpy(np.float32)


def stance_proxy(df: pd.DataFrame, source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = "concept_target" if source == "teacher" else "predicted_concept"
    quote = df[f"{prefix}_E01"].to_numpy(np.float32)
    oppose = df[f"{prefix}_E03"].to_numpy(np.float32)
    mention = 1.0 - (1.0 - quote) * (1.0 - oppose)
    use = 1.0 - mention
    return use, quote, oppose, mention


def build_features(
    df: pd.DataFrame,
    source: str,
    variant: str,
):
    c = concept_matrix(df, source)
    t = target_matrix(df, source)
    use, quote, oppose, mention = stance_proxy(df, source)
    stance = np.stack([use, quote, oppose], axis=1)

    feats = [c]
    names = [f"c_{x}" for x in A_E]
    if variant == "static":
        return np.concatenate(feats, axis=1), names

    feats.append(stance)
    names += ["stance_use", "stance_quote", "stance_oppose"]
    feats.append(t)
    names += [f"target_{x}" for x in T4]

    if variant == "indep":
        return np.concatenate(feats, axis=1), names

    # concept x stance: 21 x 3
    cs = (c[:, :, None] * stance[:, None, :]).reshape(c.shape[0], -1)
    feats.append(cs)
    names += [f"{x}_x_{s}" for x in A_E for s in ("use", "quote", "oppose")]

    if variant == "rel_stance":
        return np.concatenate(feats, axis=1), names

    # concept x target: 21 x 4
    ct = (c[:, :, None] * t[:, None, :]).reshape(c.shape[0], -1)
    feats.append(ct)
    names += [f"{x}_x_{y}" for x in A_E for y in T4]

    if variant == "rel_target":
        return np.concatenate(feats, axis=1), names

    # concept x use x target: 21 x 4
    cut = (c[:, :, None] * use[:, None, None] * t[:, None, :]).reshape(c.shape[0], -1)
    feats.append(cut)
    names += [f"{x}_x_use_x_{y}" for x in A_E for y in T4]

    if variant == "rel_full":
        return np.concatenate(feats, axis=1), names

    raise ValueError(f"未知 variant：{variant}")


class LinearHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_prob": float(p.mean()),
    }


def train_one(
    Xtr, ytr, Xva, yva, kind, seed, args, device
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)

    d = Xtr.shape[1]
    model = (LinearHead(d) if kind == "linear" else MLPHead(d)).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, device=device)

    order = np.arange(len(Xtr))
    best = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            idx = torch.as_tensor(
                order[i:i + args.batch_size], dtype=torch.long, device=device
            )
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva_t)).cpu().numpy()
        f1 = f1_score(yva, (p >= 0.5).astype(int), average="macro", zero_division=0)
        if f1 > best + 1e-6:
            best = float(f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("无 best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xva_t)).cpu().numpy()
    return model, scaler, p, best_epoch


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent
        / "results"
        / "relational_pilot"
        / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_df(args.train_predictions)
    val = load_df(args.val_predictions)
    if len(train) != 7680 or len(val) != 1920:
        raise ValueError("train/val 行数不是 7680/1920")
    if set(train["id"].astype(str)) & set(val["id"].astype(str)):
        raise ValueError("train/val id 有重叠")

    sources = ["teacher", "predicted"] if args.concept_source == "both" else [args.concept_source]
    variants = ["static", "indep", "rel_stance", "rel_target", "rel_full"]

    rows = []
    for source in sources:
        Xtr_static, _ = build_features(train, source, "static")
        Xva_static, _ = build_features(val, source, "static")
        for variant in variants:
            Xtr, names = build_features(train, source, variant)
            Xva, _ = build_features(val, source, variant)
            for kind in ("linear", "mlp"):
                for seed in seeds:
                    model, scaler, p, best_epoch = train_one(
                        Xtr, train["toxic_true"].to_numpy(int),
                        Xva, val["toxic_true"].to_numpy(int),
                        kind, seed, args, device,
                    )
                    m = metrics(val["toxic_true"].to_numpy(int), p)
                    m.update(
                        {
                            "source": source,
                            "variant": variant,
                            "head": kind,
                            "seed": seed,
                            "dim": Xtr.shape[1],
                            "best_epoch": best_epoch,
                        }
                    )
                    rows.append(m)
            print(
                f"[done] {source}/{variant} dim={Xtr.shape[1]}",
                flush=True,
            )

    by_run = pd.DataFrame(rows)
    by_run.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")
    agg = (
        by_run.groupby(["source", "variant", "head", "dim"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
            n=("macro_f1", "size"),
        )
        .reset_index()
        .sort_values(["source", "head", "macro_f1_mean"], ascending=[True, True, False])
    )
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    # 简单 stance 干预：rel_full + linear + teacher，把 use 置 0、mention/oppose 置 1，
    # 重建特征并看预测变化。只作诊断，不是因果证明。
    intervention_rows = []
    try:
        source = "teacher"
        Xtr, names = build_features(train, source, "rel_full")
        Xva, _ = build_features(val, source, "rel_full")
        model, scaler, p0, _ = train_one(
            Xtr, train["toxic_true"].to_numpy(int),
            Xva, val["toxic_true"].to_numpy(int),
            "linear", seeds[0], args, device,
        )
        # 重建 val 的 rel_full 特征，手动改 stance
        val2 = val.copy()
        prefix = "concept_target"
        val2[f"{prefix}_E01"] = 0.0
        val2[f"{prefix}_E03"] = 1.0
        Xva_int, _ = build_features(val2, source, "rel_full")
        Xva_int = scaler.transform(Xva_int).astype(np.float32)
        with torch.no_grad():
            p1 = torch.sigmoid(model(torch.as_tensor(Xva_int, device=device))).cpu().numpy()
        y = val["toxic_true"].to_numpy(int)
        for name, p in (("original", p0), ("stance_mention_intervention", p1)):
            m = metrics(y, p)
            m["variant"] = name
            intervention_rows.append(m)
        flip = float(np.mean((p0 >= 0.5).astype(int) != (p1 >= 0.5).astype(int)))
        intervention_rows.append(
            {
                "variant": "flip_rate",
                "macro_f1": flip,
                "auc": float("nan"),
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "mean_prob": float(np.mean(p1 - p0)),
            }
        )
    except Exception as exc:  # 诊断失败不影响主结果
        intervention_rows.append(
            {
                "variant": f"intervention_error:{type(exc).__name__}",
                "macro_f1": float("nan"), "auc": float("nan"),
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "mean_prob": float("nan"),
            }
        )
    pd.DataFrame(intervention_rows).to_csv(
        out_dir / "intervention.csv", index=False, encoding="utf-8-sig"
    )

    lines = []
    lines.append("Relational Concept Pilot：静态 vs 关系型概念")
    lines.append("=" * 76)
    lines.append(f"时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"device：{device}；seeds={seeds}")
    lines.append(f"train={len(train)} val={len(val)}；不训练 g，不调用 LLM")
    lines.append("")
    for source in sources:
        lines.append(f"[{source} concepts]")
        lines.append("-" * 76)
        sub = agg[agg["source"] == source]
        for _, r in sub.iterrows():
            lines.append(
                f"{r['variant']:<12s} {r['head']:<6s} dim={int(r['dim']):4d} "
                f"MacroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
                f"AUC={r['auc_mean']:.4f} FP={r['fp_mean']:.1f} FN={r['fn_mean']:.1f}"
            )
        lines.append("")
    lines.append("[stance intervention: teacher/rel_full/linear]")
    lines.append("-" * 76)
    for r in intervention_rows:
        lines.append(
            f"{r['variant']:<32s} MacroF1={r['macro_f1']:.4f} "
            f"mean_prob={r['mean_prob']:.4f}"
        )
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
