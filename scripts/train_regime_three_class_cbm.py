#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-class regime readout CBM（自包含，复用 regime-aware 脚本的数据/指标工具）。

背景
----
上一轮 two-gate + 2 维 combiner 的结果显示：
  - regime-aware 的整体 Macro-F1 没有提升（0.7839 vs flat_gated 0.7877）；
  - 但 hate gate 本身 AUC=0.921，combiner 输出与 offensive gate 相关 0.995，
    把 hate gate 的信号丢掉了；
  - 直接 OR 两个 gate 可以把 hate FN 从 0.274 降到 0.116，代价是 non_toxic FP 升高。

本轮调整
--------
不再把 34 维概念压缩成 2 个 gate logit，而是直接在全部 34 维概念上训练
non_toxic / hate / offensive 三分类 readout：
  - 二分类 toxic 概率 = 1 - P(non_toxic) = P(hate) + P(offensive)；
  - 同时保留 P(hate)/P(offensive)，支持 regime-conditional 的 threshold / OR 决策；
  - 主模型使用 sqrt-inverse 类别权重（hate 约 1.7x），避免 6.8% 的 hate 被忽略；
  - 对照：不加权三分类，用于区分“三分类结构”与“类别加权”的贡献。

评估
----
  - 二分类：Acc / Macro-F1 / Pos-F1 / P / R / AUC；non_toxic FP、hate FN、offensive FN；
  - 三分类：各类 precision/recall/F1、混淆矩阵；
  - operating policy：固定 0.5、val 选 Macro-F1、val 选 hate recall>=90%、
    三分类 argmax、P(hate)/P(offensive) 的 OR 规则；
  - operating frontier：在固定 non_toxic FP 预算下，比较各模型的 hate FN / offensive FN / F1；
  - 所有阈值/策略只在 val 上选，test 只用于报告。

用法
----
python scripts/train_regime_three_class_cbm.py
python scripts/train_regime_three_class_cbm.py --seeds 729,1136,1196,4468,5114 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_regime_aware_cbm as base  # noqa: E402


PROJECT_ROOT = base.PROJECT_ROOT
OUT_ROOT = PROJECT_ROOT / "results" / "regime_three_class"
SEED_DEFAULT = base.SEED_DEFAULT
REGIME_NAMES = base.REGIME_NAMES
STRATA = base.STRATA
GRID_1D = [round(0.02 + 0.01 * i, 2) for i in range(97)]   # 0.02 .. 0.98
GRID_2D = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class Linear3(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Gated3(nn.Module):
    def __init__(self, d: int, hidden: int = 96, dropout: float = 0.5):
        super().__init__()
        self.gate = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        h = x * g
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


def train_three_class(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    model_type: str,
    class_weight: Optional[np.ndarray],
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    """三分类 readout；early stop 用 val 上的二分类 Macro-F1（1-P(non_toxic)）。"""
    base.set_seed(seed)
    d = x_tr.shape[1]
    model: nn.Module = Linear3(d) if model_type == "linear" else Gated3(d)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    weight = None
    if class_weight is not None:
        weight = torch.tensor(
            np.asarray(class_weight, dtype=np.float32), device=device
        )
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    xtr = _tensor(x_tr, device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    xva = _tensor(x_val, device)
    xte = _tensor(x_te, device)
    y_val_bin = (np.asarray(y_val) > 0).astype(int)

    loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob3 = torch.softmax(model(xva), dim=1).cpu().numpy()
        val_toxic = 1.0 - val_prob3[:, 0]
        val_f1 = base.macro_f1(
            y_val_bin, (val_toxic >= 0.5).astype(int)
        )
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob3 = torch.softmax(model(xva), dim=1).cpu().numpy()
        test_prob3 = torch.softmax(model(xte), dim=1).cpu().numpy()
    return {
        "model": model,
        "val_prob3": val_prob3,
        "test_prob3": test_prob3,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val,
    }


# ---------------------------------------------------------------------------
# 决策规则与指标
# ---------------------------------------------------------------------------

def decision_metrics(
    y_bin: np.ndarray, pred: np.ndarray, regime: np.ndarray
) -> Dict[str, float]:
    y_bin = np.asarray(y_bin).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred, labels=[0, 1]).ravel()
    p, r, f1_pos, _ = precision_recall_fscore_support(
        y_bin, pred, average="binary", pos_label=1, zero_division=0
    )
    hate_mask = regime == 1
    off_mask = regime == 2
    return {
        "macro_f1": base.macro_f1(y_bin, pred),
        "pos_f1": float(f1_pos),
        "precision": float(p),
        "recall": float(r),
        "fp_rate_non_toxic": base.safe_div(fp, fp + tn),
        "fn_rate_toxic": base.safe_div(fn, fn + tp),
        "hate_fn_rate": float(1.0 - pred[hate_mask].mean()) if hate_mask.any() else float("nan"),
        "offensive_fn_rate": float(1.0 - pred[off_mask].mean()) if off_mask.any() else float("nan"),
        "n_pos_pred": int(pred.sum()),
    }


def _fp_key(m: Dict[str, float]) -> float:
    fp = m["fp_rate_non_toxic"]
    return fp if math.isfinite(fp) else float("inf")


def select_1d(
    y_bin: np.ndarray,
    prob: np.ndarray,
    regime: np.ndarray,
    objective: str,
    hate_recall_target: float = 0.90,
) -> float:
    cands: List[Tuple[float, Dict[str, float]]] = []
    for t in GRID_1D:
        m = decision_metrics(y_bin, (prob >= t).astype(int), regime)
        cands.append((t, m))
    if objective == "macro_f1":
        return float(max(cands, key=lambda c: c[1]["macro_f1"])[0])
    feasible = [
        c for c in cands
        if math.isfinite(c[1]["hate_fn_rate"])
        and (1.0 - c[1]["hate_fn_rate"]) >= hate_recall_target
    ]
    if not feasible:
        feasible = cands
    best = min(
        feasible,
        key=lambda c: (_fp_key(c[1]), -c[1]["macro_f1"]),
    )
    return float(best[0])


def select_2d(
    prob3: np.ndarray,
    y_bin: np.ndarray,
    regime: np.ndarray,
    objective: str,
    hate_recall_target: float = 0.90,
) -> Tuple[float, float]:
    p_h = prob3[:, 1]
    p_o = prob3[:, 2]
    cands: List[Tuple[Tuple[float, float], Dict[str, float]]] = []
    for th in GRID_2D:
        for to in GRID_2D:
            pred = ((p_h >= th) | (p_o >= to)).astype(int)
            m = decision_metrics(y_bin, pred, regime)
            cands.append(((th, to), m))
    if objective == "macro_f1":
        return tuple(max(cands, key=lambda c: c[1]["macro_f1"])[0])  # type: ignore[return-value]
    feasible = [
        c for c in cands
        if math.isfinite(c[1]["hate_fn_rate"])
        and (1.0 - c[1]["hate_fn_rate"]) >= hate_recall_target
    ]
    if not feasible:
        feasible = cands
    best = min(
        feasible,
        key=lambda c: (_fp_key(c[1]), -c[1]["macro_f1"]),
    )
    return tuple(best[0])  # type: ignore[return-value]


def policy_record(
    model_name: str,
    seed: int,
    policy: str,
    th: float,
    to: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    test_auc: float,
) -> Dict[str, object]:
    return {
        "model": model_name,
        "seed": seed,
        "policy": policy,
        "threshold_hate": th,
        "threshold_offensive": to,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_fp_rate_non_toxic": val_metrics["fp_rate_non_toxic"],
        "val_hate_fn_rate": val_metrics["hate_fn_rate"],
        "val_offensive_fn_rate": val_metrics["offensive_fn_rate"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_pos_f1": test_metrics["pos_f1"],
        "test_auc": test_auc,
        "test_fp_rate_non_toxic": test_metrics["fp_rate_non_toxic"],
        "test_hate_fn_rate": test_metrics["hate_fn_rate"],
        "test_offensive_fn_rate": test_metrics["offensive_fn_rate"],
        "test_n_pos_pred": test_metrics["n_pos_pred"],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_DEFAULT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    split_seed = int(args.split_seed) if args.split_seed else seeds[0]
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}，seeds={seeds}，split_seed={split_seed}")

    train_df, concept_cols = base.load_split("train")
    test_df, test_concepts = base.load_split("test")
    if concept_cols != test_concepts:
        raise ValueError("train/test 概念列顺序不一致，请先对齐。")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录：{out_dir}")

    tr_idx, val_idx = base.stratified_split_by_regime(
        train_df, seed=split_seed, val_ratio=args.val_ratio
    )
    x_train_raw = train_df[concept_cols].to_numpy(dtype=float)
    x_test_raw = test_df[concept_cols].to_numpy(dtype=float)
    y3_train = train_df["regime"].to_numpy(dtype=int)
    y3_test = test_df["regime"].to_numpy(dtype=int)
    y_bin_train = train_df["toxic"].to_numpy(dtype=int)
    y_bin_test = test_df["toxic"].to_numpy(dtype=int)
    regime_test = test_df["regime"].to_numpy(dtype=int)
    regime_val = y3_train[val_idx]

    scaler = StandardScaler().fit(x_train_raw[tr_idx])
    x_tr = scaler.transform(x_train_raw[tr_idx])
    x_val = scaler.transform(x_train_raw[val_idx])
    x_te = scaler.transform(x_test_raw)

    counts = np.bincount(y3_train[tr_idx], minlength=3).astype(float)
    n_tr = float(len(tr_idx))
    weights_sqrt = np.sqrt(n_tr / np.maximum(counts, 1.0))
    weights_inv = n_tr / np.maximum(counts, 1.0)
    print(
        f"[info] train regime counts={ {REGIME_NAMES[i]: int(counts[i]) for i in range(3)} }；"
        f"sqrt weights={np.round(weights_sqrt, 3).tolist()}；inv weights={np.round(weights_inv, 3).tolist()}"
    )

    overview = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_concepts": int(len(concept_cols)),
        "concepts": concept_cols,
        "train_regime_counts": {
            REGIME_NAMES[k]: int(v)
            for k, v in train_df["regime"].value_counts().sort_index().items()
        },
        "test_regime_counts": {
            REGIME_NAMES[k]: int(v)
            for k, v in test_df["regime"].value_counts().sort_index().items()
        },
        "split_seed": int(split_seed),
        "val_ratio": float(args.val_ratio),
        "seeds": seeds,
        "class_weight_sqrt": {
            REGIME_NAMES[i]: float(weights_sqrt[i]) for i in range(3)
        },
        "class_weight_inv": {
            REGIME_NAMES[i]: float(weights_inv[i]) for i in range(3)
        },
    }
    with (out_dir / "overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    strata = base.strata_masks(test_df)

    run_rows: List[Dict[str, object]] = []
    strata_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    policy_rows: List[Dict[str, object]] = []
    frontier_rows: List[Dict[str, object]] = []
    seed_test_probs: Dict[str, Dict[int, np.ndarray]] = {}
    seed_test_prob3: Dict[str, Dict[int, np.ndarray]] = {}

    model_specs = [
        ("flat_linear", "binary", "linear", None),
        ("flat_gated", "binary", "gated", None),
        ("regime3_linear_balanced", "three", "linear", weights_sqrt),
        ("regime3_gated_balanced", "three", "gated", weights_sqrt),
        ("regime3_gated_unweighted", "three", "gated", None),
    ]

    for seed in tqdm(seeds, desc="seeds", leave=False):
        for model_name, kind, model_type, class_weight in model_specs:
            if kind == "binary":
                out = base.train_flat(
                    x_tr,
                    y_bin_train[tr_idx],
                    x_val,
                    y_bin_train[val_idx],
                    x_te,
                    model_type=model_type,
                    seed=seed,
                    epochs=args.epochs,
                    device=device,
                    batch_size=args.batch_size,
                    patience=args.patience,
                )
                val_prob = np.asarray(out["val_prob"])
                test_prob = np.asarray(out["test_prob"])
                val_prob3 = None
                test_prob3 = None
                val_toxic = val_prob
                test_toxic = test_prob
                best_epoch = int(out["best_epoch"])
                seed_test_probs.setdefault(model_name, {})[seed] = test_prob
            else:
                out = train_three_class(
                    x_tr,
                    y3_train[tr_idx],
                    x_val,
                    y3_train[val_idx],
                    x_te,
                    model_type=model_type,
                    class_weight=class_weight,
                    seed=seed,
                    epochs=args.epochs,
                    device=device,
                    batch_size=args.batch_size,
                    patience=args.patience,
                )
                val_prob3 = np.asarray(out["val_prob3"])
                test_prob3 = np.asarray(out["test_prob3"])
                val_toxic = 1.0 - val_prob3[:, 0]
                test_toxic = 1.0 - test_prob3[:, 0]
                best_epoch = int(out["best_epoch"])
                seed_test_probs.setdefault(model_name, {})[seed] = test_toxic
                seed_test_prob3.setdefault(model_name, {})[seed] = test_prob3

            test_auc = base.auc_or_nan(y_bin_test, test_toxic)

            # 固定 0.5 / val-opt 二分类指标
            t_opt = base.best_threshold(y_bin_train[val_idx], val_toxic)
            for policy, t in [("fixed_0.5", 0.5), ("val_opt", t_opt)]:
                m = base.binary_metrics(y_bin_test, test_toxic, t)
                run_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "threshold_policy": policy,
                        "threshold": float(t),
                        "acc": m["acc"],
                        "macro_f1": m["macro_f1"],
                        "pos_f1": m["pos_f1"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "auc": m["auc"],
                        "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                        "fn_rate_toxic": m["fn_rate_toxic"],
                        "hate_fn_rate": float(np.mean(test_toxic[regime_test == 1] < t)),
                        "offensive_fn_rate": float(np.mean(test_toxic[regime_test == 2] < t)),
                        "best_epoch": best_epoch,
                    }
                )
                for stratum_name in STRATA:
                    mask = strata[stratum_name]
                    if mask.sum() == 0:
                        continue
                    sm = base.stratum_metrics(y_bin_test, test_toxic, t, mask)
                    strata_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "threshold_policy": policy,
                            "stratum": stratum_name,
                            **sm,
                        }
                    )

            # 三分类 per-class / 混淆矩阵
            if kind == "three" and test_prob3 is not None:
                pred3 = test_prob3.argmax(axis=1)
                p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
                    y3_test, pred3, labels=[0, 1, 2], average=None, zero_division=0
                )
                for k in range(3):
                    per_class_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "regime": REGIME_NAMES[k],
                            "precision": float(p_c[k]),
                            "recall": float(r_c[k]),
                            "f1": float(f1_c[k]),
                            "support": int(sup_c[k]),
                        }
                    )
                cm = confusion_matrix(y3_test, pred3, labels=[0, 1, 2])
                for i in range(3):
                    cm_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "true_regime": REGIME_NAMES[i],
                            "pred_non_toxic": int(cm[i, 0]),
                            "pred_hate": int(cm[i, 1]),
                            "pred_offensive": int(cm[i, 2]),
                        }
                    )

            # operating policies
            val_bin = y_bin_train[val_idx]
            # 1) fixed 0.5
            val_m = decision_metrics(val_bin, (val_toxic >= 0.5).astype(int), regime_val)
            test_m = decision_metrics(y_bin_test, (test_toxic >= 0.5).astype(int), regime_test)
            policy_rows.append(
                policy_record(model_name, seed, "binary_fixed_0.5", 0.5, float("nan"),
                              val_m, test_m, test_auc)
            )
            # 2) val Macro-F1 1D（与 metrics_by_run 的 val_opt 保持一致）
            t_f1 = base.best_threshold(val_bin, val_toxic)
            val_m = decision_metrics(val_bin, (val_toxic >= t_f1).astype(int), regime_val)
            test_m = decision_metrics(y_bin_test, (test_toxic >= t_f1).astype(int), regime_test)
            policy_rows.append(
                policy_record(model_name, seed, "binary_val_macroF1", t_f1, float("nan"),
                              val_m, test_m, test_auc)
            )
            # 3) val hate recall >=90% 1D
            t_h = select_1d(val_bin, val_toxic, regime_val, "hate_recall", 0.90)
            val_m = decision_metrics(val_bin, (val_toxic >= t_h).astype(int), regime_val)
            test_m = decision_metrics(y_bin_test, (test_toxic >= t_h).astype(int), regime_test)
            policy_rows.append(
                policy_record(model_name, seed, "binary_val_hateRecall90", t_h, float("nan"),
                              val_m, test_m, test_auc)
            )

            if kind == "three" and val_prob3 is not None and test_prob3 is not None:
                # 4) argmax
                val_pred3 = val_prob3.argmax(axis=1)
                test_pred3 = test_prob3.argmax(axis=1)
                val_m = decision_metrics(val_bin, (val_pred3 != 0).astype(int), regime_val)
                test_m = decision_metrics(y_bin_test, (test_pred3 != 0).astype(int), regime_test)
                policy_rows.append(
                    policy_record(model_name, seed, "three_argmax", float("nan"), float("nan"),
                                  val_m, test_m, test_auc)
                )
                # 5) OR, val Macro-F1
                th, to = select_2d(val_prob3, val_bin, regime_val, "macro_f1")
                val_pred = ((val_prob3[:, 1] >= th) | (val_prob3[:, 2] >= to)).astype(int)
                test_pred = ((test_prob3[:, 1] >= th) | (test_prob3[:, 2] >= to)).astype(int)
                val_m = decision_metrics(val_bin, val_pred, regime_val)
                test_m = decision_metrics(y_bin_test, test_pred, regime_test)
                policy_rows.append(
                    policy_record(model_name, seed, "or_val_macroF1", th, to,
                                  val_m, test_m, test_auc)
                )
                # 6) OR, val hate recall >=90%
                th2, to2 = select_2d(val_prob3, val_bin, regime_val, "hate_recall", 0.90)
                val_pred = ((val_prob3[:, 1] >= th2) | (val_prob3[:, 2] >= to2)).astype(int)
                test_pred = ((test_prob3[:, 1] >= th2) | (test_prob3[:, 2] >= to2)).astype(int)
                val_m = decision_metrics(val_bin, val_pred, regime_val)
                test_m = decision_metrics(y_bin_test, test_pred, regime_test)
                policy_rows.append(
                    policy_record(model_name, seed, "or_val_hateRecall90", th2, to2,
                                  val_m, test_m, test_auc)
                )

            # operating frontier（test 诊断；不用于选模型）
            for t in GRID_1D:
                m = decision_metrics(y_bin_test, (test_toxic >= t).astype(int), regime_test)
                frontier_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "rule": "binary",
                        "threshold_hate": float(t),
                        "threshold_offensive": float("nan"),
                        **m,
                    }
                )
            if kind == "three" and test_prob3 is not None:
                for th in GRID_2D:
                    for to in GRID_2D:
                        pred = ((test_prob3[:, 1] >= th) | (test_prob3[:, 2] >= to)).astype(int)
                        m = decision_metrics(y_bin_test, pred, regime_test)
                        frontier_rows.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "rule": "or",
                                "threshold_hate": float(th),
                                "threshold_offensive": float(to),
                                **m,
                            }
                        )

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(run_rows)
    metrics_df.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")

    agg_cols = [
        "acc", "macro_f1", "pos_f1", "precision", "recall", "auc",
        "fp_rate_non_toxic", "fn_rate_toxic", "hate_fn_rate", "offensive_fn_rate",
        "threshold",
    ]
    agg = (
        metrics_df.groupby(["model", "threshold_policy"])[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(out_dir / "regime_strata.csv", index=False, encoding="utf-8-sig")

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=False, encoding="utf-8-sig")

    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(out_dir / "operating_policy.csv", index=False, encoding="utf-8-sig")

    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(out_dir / "operating_points.csv", index=False, encoding="utf-8-sig")

    # test 预测（跨 seed 平均）
    pred_df = test_df[
        ["id", "content", "toxic", "expression", "toxic_type", "target_any"]
        + [f"target_{j}" for j in range(base.TARGET_DIMS)]
    ].copy()
    for model_name, _, _, _ in model_specs:
        arr = np.stack([seed_test_probs[model_name][s] for s in seeds], axis=0)
        pred_df[f"{model_name}_prob_toxic_mean"] = arr.mean(axis=0)
        pred_df[f"{model_name}_prob_toxic_std"] = arr.std(
            axis=0, ddof=1 if len(seeds) > 1 else 0
        )
        pred_df[f"{model_name}_pred_05"] = (arr.mean(axis=0) >= 0.5).astype(int)
    for model_name in ["regime3_linear_balanced", "regime3_gated_balanced", "regime3_gated_unweighted"]:
        if model_name not in seed_test_prob3:
            continue
        arr3 = np.stack([seed_test_prob3[model_name][s] for s in seeds], axis=0)
        mean3 = arr3.mean(axis=0)
        pred_df[f"{model_name}_p_non_toxic"] = mean3[:, 0]
        pred_df[f"{model_name}_p_hate"] = mean3[:, 1]
        pred_df[f"{model_name}_p_offensive"] = mean3[:, 2]
        pred_df[f"{model_name}_argmax"] = mean3.argmax(axis=1)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def ms(model: str, policy: str, col: str) -> Tuple[float, float]:
        vals = metrics_df[
            (metrics_df["model"] == model) & (metrics_df["threshold_policy"] == policy)
        ][col].dropna()
        if len(vals) == 0:
            return float("nan"), float("nan")
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return float(vals.mean()), std

    def ms_text(model: str, policy: str, col: str) -> str:
        m, s = ms(model, policy, col)
        if math.isnan(m):
            return "na"
        return f"{base.fmt(m)}±{base.fmt(s)}"

    def pol(model: str, policy: str, col: str) -> Tuple[float, float]:
        vals = policy_df[
            (policy_df["model"] == model) & (policy_df["policy"] == policy)
        ][col].dropna()
        if len(vals) == 0:
            return float("nan"), float("nan")
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        return float(vals.mean()), std

    def pol_text(model: str, policy: str, col: str) -> str:
        m, s = pol(model, policy, col)
        if math.isnan(m):
            return "na"
        return f"{base.fmt(m)}±{base.fmt(s)}"

    lines: List[str] = []
    lines.append("3-class regime readout CBM 训练与评估报告")
    lines.append("=" * 70)
    lines.append(f"时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"device：{device}")
    lines.append(f"seeds：{seeds}")
    lines.append(
        f"split：按 regime 分层 {1 - args.val_ratio:.0%}/{args.val_ratio:.0%}，"
        f"split_seed={split_seed}；test 不参与训练/阈值选择"
    )
    lines.append(
        f"概念：{len(concept_cols)} 维（v8 26 + normal 8）；"
        "三分类标签 = non_toxic / hate / offensive；toxic = hate or offensive"
    )
    lines.append(
        f"类别权重（sqrt-inverse）："
        f"non_toxic={weights_sqrt[0]:.3f}，hate={weights_sqrt[1]:.3f}，"
        f"offensive={weights_sqrt[2]:.3f}"
    )
    lines.append("")

    lines.append("一、数据与标注 regime")
    lines.append("-" * 70)
    lines.append(f"train rows={len(train_df)}，test rows={len(test_df)}")
    for name in ["non_toxic", "hate", "offensive"]:
        lines.append(
            f"  {name:<10} train={overview['train_regime_counts'].get(name, 0):<5} "
            f"test={overview['test_regime_counts'].get(name, 0):<5}"
        )
    lines.append(
        "  说明：non_toxic 与 hate 都落在 expression=0；"
        "三分类 readout 直接用 toxic_type 监督，避免二分类把两者混在一起。"
    )
    lines.append("")

    lines.append("二、二分类整体指标（threshold=0.5；seed 平均±std）")
    lines.append("-" * 70)
    lines.append(
        f"{'model':<27} {'Acc':<17} {'MacroF1':<17} {'PosF1':<17} {'AUC':<17}"
        f"{'FP_rate':<17} {'FN_hate':<17} {'FN_off'}"
    )
    for model_name, _, _, _ in model_specs:
        lines.append(
            f"{model_name:<27} "
            f"{ms_text(model_name, 'fixed_0.5', 'acc'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'macro_f1'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'pos_f1'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'auc'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'fp_rate_non_toxic'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'hate_fn_rate'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'offensive_fn_rate')}"
        )
    lines.append("")

    lines.append("三、val-opt 二分类指标")
    lines.append("-" * 70)
    lines.append(
        f"{'model':<27} {'MacroF1':<17} {'PosF1':<17} {'AUC':<17} {'threshold'}"
    )
    for model_name, _, _, _ in model_specs:
        lines.append(
            f"{model_name:<27} "
            f"{ms_text(model_name, 'val_opt', 'macro_f1'):<17} "
            f"{ms_text(model_name, 'val_opt', 'pos_f1'):<17} "
            f"{ms_text(model_name, 'val_opt', 'auc'):<17} "
            f"{ms_text(model_name, 'val_opt', 'threshold')}"
        )
    lines.append("")

    lines.append("四、三分类 per-class 指标（argmax；seed 平均±std）")
    lines.append("-" * 70)
    for model_name in ["regime3_linear_balanced", "regime3_gated_balanced", "regime3_gated_unweighted"]:
        lines.append(f"[{model_name}]")
        sub = per_class_df[per_class_df["model"] == model_name]
        if len(sub) == 0:
            lines.append("  （无）")
            continue
        for regime_name in ["non_toxic", "hate", "offensive"]:
            s = sub[sub["regime"] == regime_name]
            lines.append(
                f"  {regime_name:<10} support={int(s['support'].iloc[0]):<5} "
                f"Precision={base.mean_std_text(s['precision'])} "
                f"Recall={base.mean_std_text(s['recall'])} "
                f"F1={base.mean_std_text(s['f1'])}"
            )
        csub = cm_df[cm_df["model"] == model_name]
        if len(csub):
            mean_cm = (
                csub.groupby("true_regime")[
                    ["pred_non_toxic", "pred_hate", "pred_offensive"]
                ]
                .mean()
                .reindex(["non_toxic", "hate", "offensive"])
                .to_numpy()
            )
            lines.append(
                f"  mean confusion（true rows: non_toxic/hate/offensive）= "
                f"[[{mean_cm[0, 0]:.1f}, {mean_cm[0, 1]:.1f}, {mean_cm[0, 2]:.1f}], "
                f"[{mean_cm[1, 0]:.1f}, {mean_cm[1, 1]:.1f}, {mean_cm[1, 2]:.1f}], "
                f"[{mean_cm[2, 0]:.1f}, {mean_cm[2, 1]:.1f}, {mean_cm[2, 2]:.1f}]]"
            )
        lines.append("")

    lines.append("五、operating policy（阈值只在 val 选；test 指标为 seed 平均±std）")
    lines.append("-" * 70)
    lines.append(
        f"{'model':<27} {'policy':<26} {'th_h':<7} {'th_o':<7} "
        f"{'MacroF1':<17} {'FP_rate':<17} {'FN_hate':<17} {'FN_off'}"
    )
    for model_name, _, _, _ in model_specs:
        policies = [
            "binary_fixed_0.5", "binary_val_macroF1", "binary_val_hateRecall90",
        ]
        if model_name.startswith("regime3_"):
            policies += ["three_argmax", "or_val_macroF1", "or_val_hateRecall90"]
        for policy in policies:
            sub = policy_df[
                (policy_df["model"] == model_name) & (policy_df["policy"] == policy)
            ]
            if len(sub) == 0:
                continue
            th = sub["threshold_hate"].mean()
            to = sub["threshold_offensive"].mean()
            lines.append(
                f"{model_name:<27} {policy:<26} "
                f"{base.fmt(th, 2):<7} {base.fmt(to, 2):<7} "
                f"{pol_text(model_name, policy, 'test_macro_f1'):<17} "
                f"{pol_text(model_name, policy, 'test_fp_rate_non_toxic'):<17} "
                f"{pol_text(model_name, policy, 'test_hate_fn_rate'):<17} "
                f"{pol_text(model_name, policy, 'test_offensive_fn_rate')}"
            )
    lines.append("")

    lines.append("六、operating frontier（test 诊断；固定 non_toxic FP 预算下最小 hate FN）")
    lines.append("-" * 70)
    lines.append(
        f"{'model':<27} {'FP_budget':<10} {'rule':<7} {'th_h':<7} {'th_o':<7} "
        f"{'FP_rate':<10} {'FN_hate':<10} {'FN_off':<10} {'MacroF1'}"
    )
    for model_name in ["flat_gated", "regime3_linear_balanced", "regime3_gated_balanced", "regime3_gated_unweighted"]:
        sub_model = frontier_df[frontier_df["model"] == model_name]
        for budget in [0.10, 0.15, 0.20, 0.25]:
            sub = sub_model[sub_model["fp_rate_non_toxic"] <= budget]
            if len(sub) == 0:
                continue
            # 每个 seed 取 hate FN 最小点，再跨 seed 平均
            seed_rows = []
            for seed, g in sub.groupby("seed"):
                b = g.sort_values(["hate_fn_rate", "macro_f1"], ascending=[True, False]).iloc[0]
                seed_rows.append(b)
            bdf = pd.DataFrame(seed_rows)
            lines.append(
                f"{model_name:<27} {budget:<10.2f} {str(bdf['rule'].iloc[0]):<7} "
                f"{base.fmt(bdf['threshold_hate'].mean(), 2):<7} "
                f"{base.fmt(bdf['threshold_offensive'].mean(), 2):<7} "
                f"{base.fmt(bdf['fp_rate_non_toxic'].mean()):<10} "
                f"{base.fmt(bdf['hate_fn_rate'].mean()):<10} "
                f"{base.fmt(bdf['offensive_fn_rate'].mean()):<10} "
                f"{base.fmt(bdf['macro_f1'].mean())}"
            )
    lines.append(
        "  说明：这是 test 上的诊断 frontier，不是模型选择；"
        "用于判断在相同 non_toxic FP 预算下，三分类能否比 flat 少漏 hate。"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # 结论
    # ------------------------------------------------------------------
    def paired(model_a: str, model_b: str, policy: str, col: str) -> Tuple[float, float]:
        return base.paired_delta(metrics_df, model_a, model_b, policy, col)

    dm_lin, ds_lin = paired("regime3_gated_balanced", "flat_linear", "fixed_0.5", "macro_f1")
    dm_gat, ds_gat = paired("regime3_gated_balanced", "flat_gated", "fixed_0.5", "macro_f1")
    dm_lin_opt, ds_lin_opt = paired("regime3_gated_balanced", "flat_linear", "val_opt", "macro_f1")
    dm_gat_opt, ds_gat_opt = paired("regime3_gated_balanced", "flat_gated", "val_opt", "macro_f1")

    lines.append("七、结论与判读")
    lines.append("-" * 70)
    lines.append(
        f"- 整体 Macro-F1（0.5）：regime3_gated_balanced="
        f"{ms_text('regime3_gated_balanced', 'fixed_0.5', 'macro_f1')}，"
        f"flat_gated={ms_text('flat_gated', 'fixed_0.5', 'macro_f1')}，"
        f"flat_linear={ms_text('flat_linear', 'fixed_0.5', 'macro_f1')}。"
    )
    lines.append(
        f"  vs flat_linear：Δ={base.fmt_signed(dm_lin)}±{base.fmt(ds_lin)} → "
        f"{base.verdict_improve(dm_lin, ds_lin)}；"
        f"vs flat_gated：Δ={base.fmt_signed(dm_gat)}±{base.fmt(ds_gat)} → "
        f"{base.verdict_improve(dm_gat, ds_gat)}。"
    )
    lines.append(
        f"  val-opt 下：vs flat_linear Δ={base.fmt_signed(dm_lin_opt)}±{base.fmt(ds_lin_opt)}；"
        f"vs flat_gated Δ={base.fmt_signed(dm_gat_opt)}±{base.fmt(ds_gat_opt)}。"
    )
    lines.append("")
    lines.append(
        f"- 在 val 选出的 hate recall>=90% 策略下（test）："
        f"flat_gated FP={pol_text('flat_gated', 'binary_val_hateRecall90', 'test_fp_rate_non_toxic')}，"
        f"FN_hate={pol_text('flat_gated', 'binary_val_hateRecall90', 'test_hate_fn_rate')}，"
        f"FN_off={pol_text('flat_gated', 'binary_val_hateRecall90', 'test_offensive_fn_rate')}，"
        f"MacroF1={pol_text('flat_gated', 'binary_val_hateRecall90', 'test_macro_f1')}；"
    )
    lines.append(
        f"  regime3_gated_balanced OR："
        f"FP={pol_text('regime3_gated_balanced', 'or_val_hateRecall90', 'test_fp_rate_non_toxic')}，"
        f"FN_hate={pol_text('regime3_gated_balanced', 'or_val_hateRecall90', 'test_hate_fn_rate')}，"
        f"FN_off={pol_text('regime3_gated_balanced', 'or_val_hateRecall90', 'test_offensive_fn_rate')}，"
        f"MacroF1={pol_text('regime3_gated_balanced', 'or_val_hateRecall90', 'test_macro_f1')}。"
    )
    lines.append("")
    lines.append("  诚实的操作性结论：")
    lines.append(
        "  - 本实验检验的是“三分类 regime readout + 类别加权”能否把 hate 信号真正用于决策，"
        "而不是继续压缩成二分类 combiner。"
    )
    lines.append(
        "  - 若整体 Macro-F1 没有提升，但相同 hate recall 目标下 non_toxic FP 更低，"
        "说明 regime 信息的价值在 operating point，而不是全局 F1。"
    )
    lines.append(
        "  - 若整体 F1 与 hate recall/FP 都没有改善，说明当前 34 维概念对 hate vs non_toxic 的"
        "可分性已经接近上限，下一步应转向概念测量（PLM/LLM），而不是继续换 readout。"
    )
    lines.append(
        f"  本次结果判读：MacroF1 vs flat_gated={base.verdict_improve(dm_gat, ds_gat)}；"
        f"vs flat_linear={base.verdict_improve(dm_lin, ds_lin)}。"
    )
    lines.append("")
    lines.append("八、限制")
    lines.append("-" * 70)
    lines.append("- 三分类标签由 toxic_type 派生；hate 只有 6.8%，指标方差大。")
    lines.append("- 所有 seed 共用同一个 train/val split；seed 间波动只反映初始化/采样。")
    lines.append("- 类别权重 sqrt-inverse 是一种折中；inv 权重更强，未作为主模型。")
    lines.append("- 三分类 argmax 与二分类 1-P(non_toxic) 是不同决策规则，报告中分别列出。")
    lines.append("- T05 仍作为特征输入（与 flat baseline 对齐），但其 target 映射不成立。")
    lines.append("- operating frontier 是 test 诊断，不用于模型选择；策略阈值只在 val 选。")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] 结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
