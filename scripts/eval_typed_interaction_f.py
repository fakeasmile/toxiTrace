#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 true INDEPENDENT-CBM 的 teacher concept 上测试 typed interaction f。

目的
----
independent CBM 的上限是 f(teacher c)。当前 f(c_val)≈0.7893，f(g(x_val))≈0.7874。
本脚本只改任务头 f，不改 g、不重训 g、不碰 test：

  baseline:      c(21 A-E) -> MLP / Linear
  typed21:       c + A×E/B×E/C×E/D×E/A×B/A×C/B×C/C×D 的块级交互
  typed25:       c(21 A-E + T01-T04) + A×E/B×T/C×T/D×E/A×T/B×E/C×E/D×T/E×T

f 仍然只用 teacher c 训练；评估两种输入：
  f(teacher c_val)  -> independent CBM 的概念上限
  f(g(x_val))       -> 实际部署管线

输出
----
results/typed_interaction_f/<timestamp>/
  metrics_by_run.csv
  metrics_aggregate.csv
  linear_weights.csv
  report.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score

CBM_DIR = Path(__file__).resolve().parent / (
    "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0"
)
OUTPUTS = CBM_DIR / "outputs" / "independent_true_seed20261122"
TRAIN_PRED = OUTPUTS / "train_predictions.csv"
VAL_PRED = OUTPUTS / "validation_predictions.csv"

CONCEPT_COLS = [
    "A01", "A02", "A03",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04",
    "D01", "D02",
    "E01", "E02", "E03", "E04", "E05",
    "T01", "T02", "T03", "T04", "T05",
]
CONCEPTS_21 = CONCEPT_COLS[:21]
CONCEPTS_25 = CONCEPT_COLS[:25]  # 去掉 T05
BLOCKS = {
    "A": ["A01", "A02", "A03"],
    "B": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
    "C": ["C01", "C02", "C03", "C04"],
    "D": ["D01", "D02"],
    "E": ["E01", "E02", "E03", "E04", "E05"],
    "T": ["T01", "T02", "T03", "T04"],
}
TYPED_PAIRS_21 = [
    ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E"),
    ("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"),
]
TYPED_PAIRS_25 = [
    ("A", "E"), ("B", "T"), ("C", "T"), ("D", "E"),
    ("A", "T"), ("B", "E"), ("C", "E"), ("D", "T"),
    ("E", "T"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_predictions", type=Path, default=TRAIN_PRED)
    p.add_argument("--val_predictions", type=Path, default=VAL_PRED)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="默认 results/typed_interaction_f/<timestamp>",
    )
    p.add_argument("--seeds", type=str, default="729,1136,1196,4468,5114")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_predictions(path: Path, concepts) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = ["id", "toxic_true"]
    for c in concepts:
        need += [f"concept_target_{c}", f"predicted_concept_{c}"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺少列：{miss}")
    teacher = df[[f"concept_target_{c}" for c in concepts]].to_numpy(np.float32)
    pred = df[[f"predicted_concept_{c}" for c in concepts]].to_numpy(np.float32)
    y = df["toxic_true"].to_numpy(int)
    return {
        "df": df,
        "ids": df["id"].astype(str).tolist(),
        "teacher": teacher,
        "pred": pred,
        "y": y,
    }


def block_slices(concepts):
    out = {}
    for block, members in BLOCKS.items():
        idx = [concepts.index(c) for c in members if c in concepts]
        if idx:
            out[block] = np.asarray(idx, dtype=int)
    return out


def build_features(x: np.ndarray, concepts, typed: bool):
    if not typed:
        return x, [f"{c}" for c in concepts]
    slices = block_slices(concepts)
    means = {}
    for block, idx in slices.items():
        means[block] = x[:, idx].mean(axis=1)
    pairs = TYPED_PAIRS_25 if len(concepts) == 25 else TYPED_PAIRS_21
    feats = [x]
    names = [f"{c}" for c in concepts]
    for block in slices:
        feats.append(means[block][:, None])
        names.append(f"mean_{block}")
    for b1, b2 in pairs:
        if b1 in means and b2 in means:
            feats.append((means[b1] * means[b2])[:, None])
            names.append(f"{b1}x{b2}")
    return np.concatenate(feats, axis=1), names


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

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
    model,
    Xtr,
    ytr,
    Xval_pipe,
    Xval_oracle,
    yval,
    seed,
    args,
    device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    Xp = torch.as_tensor(Xval_pipe, dtype=torch.float32, device=device)
    Xo = torch.as_tensor(Xval_oracle, dtype=torch.float32, device=device)

    order = np.arange(len(Xtr))
    best = -1.0
    best_epoch = 0
    best_state = None
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
            p_pipe = torch.sigmoid(model(Xp)).cpu().numpy()
        val_f1 = float(
            f1_score(yval, (p_pipe >= 0.5).astype(int), average="macro", zero_division=0)
        )
        if val_f1 > best + 1e-6:
            best = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("未产生 best_state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_pipe = torch.sigmoid(model(Xp)).cpu().numpy()
        p_oracle = torch.sigmoid(model(Xo)).cpu().numpy()
    return model, best_epoch, p_pipe, p_oracle


def main():
    args = parse_args()
    device = resolve_device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent
        / "results"
        / "typed_interaction_f"
        / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train21 = load_predictions(args.train_predictions, CONCEPTS_21)
    val21 = load_predictions(args.val_predictions, CONCEPTS_21)
    train25 = load_predictions(args.train_predictions, CONCEPTS_25)
    val25 = load_predictions(args.val_predictions, CONCEPTS_25)

    if len(train21["ids"]) != 7680 or len(val21["ids"]) != 1920:
        raise ValueError("train/val 行数不是 7680/1920")
    if set(train21["ids"]) & set(val21["ids"]):
        raise ValueError("train/val id 有重叠")
    if not np.array_equal(train21["y"], train25["y"]):
        raise ValueError("21/25 的 y 不一致")
    if not np.array_equal(val21["y"], val25["y"]):
        raise ValueError("21/25 val y 不一致")

    specs = [
        ("mlp21", 21, False, "mlp"),
        ("linear21", 21, False, "linear"),
        ("mlp_typed21", 21, True, "mlp"),
        ("linear_typed21", 21, True, "linear"),
        ("mlp25", 25, False, "mlp"),
        ("linear25", 25, False, "linear"),
        ("mlp_typed25", 25, True, "mlp"),
        ("linear_typed25", 25, True, "linear"),
    ]

    rows = []
    weight_rows = []
    for name, n_con, typed, kind in specs:
        tr = train21 if n_con == 21 else train25
        va = val21 if n_con == 21 else val25
        concepts = CONCEPTS_21 if n_con == 21 else CONCEPTS_25
        Xtr, feat_names = build_features(tr["teacher"], concepts, typed)
        Xp, _ = build_features(va["pred"], concepts, typed)
        Xo, _ = build_features(va["teacher"], concepts, typed)
        d = Xtr.shape[1]
        for seed in seeds:
            model = MLP(d) if kind == "mlp" else Linear(d)
            model, best_epoch, p_pipe, p_o = train_one(
                model, Xtr, tr["y"], Xp, Xo, va["y"], seed, args, device
            )
            for input_name, p in (
                ("teacher_c", p_o),
                ("pipeline_g_x", p_pipe),
            ):
                m = metrics(va["y"], p)
                m.update(
                    {
                        "variant": name,
                        "n_concepts": n_con,
                        "typed": int(typed),
                        "head": kind,
                        "seed": seed,
                        "input": input_name,
                        "best_epoch": best_epoch,
                        "dim": d,
                    }
                )
                rows.append(m)
            if kind == "linear":
                w = model.net.weight.detach().cpu().numpy().reshape(-1)
                for fn, wv in zip(feat_names, w):
                    weight_rows.append(
                        {
                            "variant": name,
                            "seed": seed,
                            "feature": fn,
                            "weight": float(wv),
                        }
                    )
        print(
            f"[done] {name}: dim={d} seeds={len(seeds)}", flush=True
        )

    by_run = pd.DataFrame(rows)
    by_run.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")
    agg = (
        by_run.groupby(["variant", "n_concepts", "typed", "head", "input", "dim"])
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
        .sort_values(["input", "macro_f1_mean"], ascending=[True, False])
    )
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(
            out_dir / "linear_weights.csv", index=False, encoding="utf-8-sig"
        )

    lines = []
    lines.append("Typed Interaction f：INDEPENDENT-CBM 任务头对比")
    lines.append("=" * 76)
    lines.append(f"时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"device：{device}；seeds={seeds}")
    lines.append(
        f"train={len(train21['ids'])}, val={len(val21['ids'])}；"
        "f 只用 teacher c 训练，test 未使用"
    )
    lines.append("")
    lines.append("指标（mean±std，threshold=0.5）")
    lines.append("-" * 76)
    for input_name in ("teacher_c", "pipeline_g_x"):
        lines.append(f"[{input_name}]")
        sub = agg[agg["input"] == input_name]
        for _, r in sub.iterrows():
            lines.append(
                f"{r['variant']:<16s} dim={int(r['dim']):3d} "
                f"MacroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
                f"AUC={r['auc_mean']:.4f} "
                f"FP={r['fp_mean']:.1f} FN={r['fn_mean']:.1f}"
            )
        lines.append("")
    lines.append("判读")
    lines.append("-" * 76)
    lines.append("teacher_c 是 independent CBM 的概念上限；pipeline_g_x 是实际部署。")
    lines.append("若 typed 只提升 teacher_c 不提升 pipeline，说明 g 对新增概念/交互的预测是瓶颈。")
    lines.append("若两者都不提升，说明当前概念集/交互定义不足以突破 f(c) 上限。")
    (out_dir / "report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
