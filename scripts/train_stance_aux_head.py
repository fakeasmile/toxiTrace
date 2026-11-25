#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在已有 INDEPENDENT-CBM 概念上训练 stance auxiliary head，并测试它对毒性预测的帮助。

不训练 g，不碰 test。输入：
  - data/processed/TOXICN/stance_labels/stance_labels.csv（generate_stance_labels.py 产出）
  - train_predictions.csv / validation_predictions.csv

流程：
  1. 用 21 个 g 概念训练 4 类 stance 分类器（use/mention/oppose/none）。
  2. 对全部 train/val 预测 stance 概率。
  3. 比较三种毒性头：
       static        : 21 concepts
       stance        : 21 concepts + 4 stance probs
       stance_inter  : 21 concepts + 4 stance probs + concept × stance
  4. 报告 stance 分类质量与毒性头结果。

输出：
  results/stance_aux/<timestamp>/
    stance_metrics.csv
    toxicity_by_run.csv
    toxicity_aggregate.csv
    report.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

CBM_DIR = Path(__file__).resolve().parent / (
    "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0"
)
OUTPUTS = CBM_DIR / "outputs" / "independent_true_seed20261122"
TRAIN_PRED = OUTPUTS / "train_predictions.csv"
VAL_PRED = OUTPUTS / "validation_predictions.csv"
DEFAULT_STANCE = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "TOXICN" / "stance_labels" / "stance_labels.csv"
)

A_E = [
    "A01", "A02", "A03",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04",
    "D01", "D02",
    "E01", "E02", "E03", "E04", "E05",
]
STANCES = ["use", "mention", "oppose", "none"]
STANCE_TO_ID = {s: i for i, s in enumerate(STANCES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stance_csv", type=Path, default=DEFAULT_STANCE)
    p.add_argument("--train_predictions", type=Path, default=TRAIN_PRED)
    p.add_argument("--val_predictions", type=Path, default=VAL_PRED)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="默认 results/stance_aux/<timestamp>",
    )
    p.add_argument("--seeds", type=str, default="729,1136,1196")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
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


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = ["id", "toxic_true"] + [f"predicted_concept_{c}" for c in A_E]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺少列：{miss}")
    return df


def concept_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"predicted_concept_{c}" for c in A_E]].to_numpy(np.float32)


def load_stance(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"id", "split", "stance", "valid"}
    if not need.issubset(df.columns):
        raise ValueError(f"stance_csv 缺少列：{need - set(df.columns)}")
    df = df[df["valid"] == 1].copy()
    unknown = set(df["stance"]) - set(STANCES)
    if unknown:
        raise ValueError(f"stance 有未知标签：{unknown}")
    df["stance_id"] = df["stance"].map(STANCE_TO_ID)
    return df


class StanceMLP(nn.Module):
    def __init__(self, d, n_class=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_class)
        )

    def forward(self, x):
        return self.net(x)


class ToxHead(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(d, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1)
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0))


def train_stance(
    Xtr, ytr, Xva, yva, seed, args, device
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    d = Xtr.shape[1]
    model = StanceMLP(d).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    classes = np.arange(4)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=ytr
    )
    weights = np.nan_to_num(weights, nan=1.0).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, device=device)
    )
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
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
            pred = model(Xva_t).argmax(dim=1).cpu().numpy()
        f1 = macro_f1(yva, pred)
        if f1 > best + 1e-6:
            best = f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("stance 无 best state")
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best_epoch


def predict_stance(model, scaler, X, device):
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.as_tensor(Xs, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def build_features(concepts, stance_probs, variant):
    if variant == "static":
        return concepts
    feats = [concepts, stance_probs]
    if variant == "stance":
        return np.concatenate(feats, axis=1)
    # concept x stance
    cs = (concepts[:, :, None] * stance_probs[:, None, :]).reshape(
        concepts.shape[0], -1
    )
    feats.append(cs)
    if variant == "stance_inter":
        return np.concatenate(feats, axis=1)
    raise ValueError(variant)


def train_tox(Xtr, ytr, Xva, yva, kind, seed, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    d = Xtr.shape[1]
    model = ToxHead(d, kind).to(device)
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
        f1 = macro_f1(yva, (p >= 0.5).astype(int))
        if f1 > best + 1e-6:
            best = f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("tox 无 best state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xva_t)).cpu().numpy()
    return p, best_epoch


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "macro_f1": macro_f1(y, pred),
        "auc": float(roc_auc_score(y, p)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_prob": float(p.mean()),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent
        / "results"
        / "stance_aux"
        / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_pred(args.train_predictions)
    val = load_pred(args.val_predictions)
    stance = load_stance(args.stance_csv)

    tr_lab = train.merge(stance[["id", "stance_id", "stance"]], on="id", how="inner")
    va_lab = val.merge(stance[["id", "stance_id", "stance"]], on="id", how="inner")
    if len(tr_lab) == 0 or len(va_lab) == 0:
        raise ValueError("stance 标签与 train/val 没有交集")
    if set(tr_lab["stance_id"].unique()) != set(range(4)):
        print("[warn] 训练集 stance 类别不全，macro-F1 会偏低", flush=True)

    Xtr_stance = concept_matrix(tr_lab)
    Xva_stance = concept_matrix(va_lab)
    Xtr_all = concept_matrix(train)
    Xva_all = concept_matrix(val)
    ytr = tr_lab["stance_id"].to_numpy(int)
    yva = va_lab["stance_id"].to_numpy(int)
    ytr_tox = train["toxic_true"].to_numpy(int)
    yva_tox = val["toxic_true"].to_numpy(int)

    stance_rows = []
    tox_rows = []
    for seed in seeds:
        smodel, sscaler, s_epoch = train_stance(
            Xtr_stance, ytr, Xva_stance, yva, seed, args, device
        )
        va_pred = predict_stance(smodel, sscaler, Xva_stance, device).argmax(1)
        s_metrics = {
            "seed": seed,
            "accuracy": float(accuracy_score(yva, va_pred)),
            "macro_f1": macro_f1(yva, va_pred),
            "best_epoch": s_epoch,
            "n_train": len(tr_lab),
            "n_val": len(va_lab),
        }
        stance_rows.append(s_metrics)

        # 对全部 train/val 预测 stance 概率
        Ptr = predict_stance(smodel, sscaler, Xtr_all, device)
        Pva = predict_stance(smodel, sscaler, Xva_all, device)

        for variant in ("static", "stance", "stance_inter"):
            Xtr_f = build_features(Xtr_all, Ptr, variant)
            Xva_f = build_features(Xva_all, Pva, variant)
            for kind in ("linear", "mlp"):
                p, best_epoch = train_tox(
                    Xtr_f, ytr_tox, Xva_f, yva_tox,
                    kind, seed, args, device,
                )
                m = metrics(yva_tox, p)
                m.update(
                    {
                        "seed": seed,
                        "variant": variant,
                        "head": kind,
                        "dim": Xtr_f.shape[1],
                        "best_epoch": best_epoch,
                        "stance_best_epoch": s_epoch,
                    }
                )
                tox_rows.append(m)
        print(f"[done] seed={seed}", flush=True)

    stance_df = pd.DataFrame(stance_rows)
    stance_df.to_csv(out_dir / "stance_metrics.csv", index=False, encoding="utf-8-sig")

    tox_df = pd.DataFrame(tox_rows)
    tox_df.to_csv(out_dir / "toxicity_by_run.csv", index=False, encoding="utf-8-sig")
    agg = (
        tox_df.groupby(["variant", "head", "dim"])
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
        .sort_values(["head", "macro_f1_mean"], ascending=[True, False])
    )
    agg.to_csv(out_dir / "toxicity_aggregate.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("Stance Auxiliary Head：概念上的 stance 监督")
    lines.append("=" * 76)
    lines.append(f"时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"device：{device}；seeds={seeds}")
    lines.append(f"stance labeled train={len(tr_lab)} val={len(va_lab)}")
    lines.append(
        f"stance 分布 train={tr_lab['stance'].value_counts().to_dict()} "
        f"val={va_lab['stance'].value_counts().to_dict()}"
    )
    lines.append("")
    lines.append("Stance 分类（g concepts -> 4 类）")
    lines.append("-" * 76)
    for _, r in stance_df.iterrows():
        lines.append(
            f"seed={int(r['seed'])} Acc={r['accuracy']:.4f} "
            f"MacroF1={r['macro_f1']:.4f} best_epoch={int(r['best_epoch'])}"
        )
    lines.append("")
    lines.append("毒性头（mean±std，val，threshold=0.5）")
    lines.append("-" * 76)
    for _, r in agg.iterrows():
        lines.append(
            f"{r['variant']:<12s} {r['head']:<6s} dim={int(r['dim']):4d} "
            f"MacroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
            f"AUC={r['auc_mean']:.4f} FP={r['fp_mean']:.1f} FN={r['fn_mean']:.1f}"
        )
    lines.append("")
    lines.append("判读")
    lines.append("-" * 76)
    lines.append("若 stance 分类本身很低，stance 概率就是噪声，毒性头不会涨。")
    lines.append("若 stance_inter 比 static 高，说明真实 stance 标签比 proxy 有价值。")
    lines.append("若仍停在 0.80 附近，下一步必须训练 encoder，而不是继续加 f 侧特征。")
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
