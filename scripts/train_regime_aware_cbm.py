#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regime-aware CBM：两阶段门控训练与评估（自包含，不调用 LLM）。

背景（由 TOXICN 标注结构分析得到）
----------------------------------
ToxiCN 的 test/train 标注存在三个互斥 regime：
  - non_toxic：toxic=0, toxic_type=0, expression=0, target_any=0
  - hate     ：toxic=1, toxic_type=1, expression=0, target_any=0
  - offensive：toxic=1, toxic_type=2, expression>=1, target_any=1
其中 expression=0 混有 non_toxic 与 hate，是扁平 CBM 的主要错误层。

本脚本检验一个可操作推论
------------------------
把二分类拆成两个 gate（hate gate / offensive gate），再用一个两特征
combiner 组合，是否比扁平 readout 更少地把 non_toxic 误报为 toxic、
更少漏掉 hate，并在整体 Macro-F1 上不下降。

训练协议
--------
1. 数据：v8 26 维 + normal 8 维，共 34 维概念矩阵；原始标注按 id 后缀
   与 content 双重校验对齐。
2. 划分：仅对 train 做按 regime 分层的 80/20 train/val；test 完全不参与
   训练和阈值选择。所有 seed 共用同一个 split，seed 只影响初始化与采样。
3. 基线：扁平 linear / gated readout（34 维），PyTorch 全连接层，
   与既有诊断脚本保持同一训练设置。
4. Regime-aware：
   Stage 1  hate gate：non_toxic vs hate，特征排除 T01–T05；
            offensive gate：non_toxic vs offensive，特征排除 T05
           （T01–T04 与 raw target 的映射已通过 AUC 验证）。
            两个 gate 均为单层线性 + BCEWithLogitsLoss(pos_weight=neg/pos)。
   Stage 2  冻结两个 gate，取它们的 logit 作为 2 维输入，训练一个线性
            combiner（2→2），在全部 train 样本上用二分类 toxic 标签监督。
5. Target 辅助头：只在 offensive 样本上训练 target_0..target_3（排除
   target_4，因为 T05 与 raw target 映射不一致），仅作诊断，不影响二分类。
6. 指标：Acc / Macro-F1（主）/ Pos-F1 / P / R / AUC；non_toxic FP、
   hate FN、offensive FN；gate 自身 FP/FN/AUC；regime 分层；target recall。
7. 阈值：主表用 0.5；另用 val 上 Macro-F1 最优阈值（val-opt）应用到 test，
   仅用于诊断校准，不做 test 调参。

重要说明
--------
- 这是概念向量上的敏感性分析，不是有 ground-truth 概念的因果干预。
- 结果如实报告：若 regime-aware 没有提升整体 Macro-F1，或没有降低
  non_toxic FP / hate FN，报告会明确写出，不粉饰。
- 所有随机性显式设种子；串行执行；结果写入 results/regime_aware_cbm/。

用法
----
python scripts/train_regime_aware_cbm.py
python scripts/train_regime_aware_cbm.py --seeds 729,1136,1196,4468,5114 --epochs 120
python scripts/train_regime_aware_cbm.py --device cpu
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 PyTorch。请先安装 torch。") from e

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 scikit-learn。请先安装 scikit-learn。") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "TOXICN"
V8_DIR = DATA_DIR / "processed" / "TOXICN" / "v8_glm-4-9b-chat"
NORMAL_DIR = DATA_DIR / "processed" / "TOXICN" / "normal_glm-4-9b-chat"
OUT_ROOT = PROJECT_ROOT / "results" / "regime_aware_cbm"

V8_PREFIX = "items_v8_glm-4-9b-chat"
NORMAL_PREFIX = "items_normal_glm-4-9b-chat"

META_COLS = ["id", "toxic", "content"]
TARGET_DIMS = 5
T_ALL = [f"T{i:02d}" for i in range(1, 6)]
T_VALID = [f"T{i:02d}" for i in range(1, 5)]  # T05 映射不成立，仅用于说明
T_EXCLUDED = "T05"

# 概念区块（运行时只保留矩阵中实际存在的列）
BLOCK_DEFS: Dict[str, List[str]] = {
    "T": T_ALL,
    "A": [f"A{i:02d}" for i in range(1, 5)],
    "B": [f"B{i:02d}" for i in range(1, 8)],
    "C": [f"C{i:02d}" for i in range(1, 5)],
    "D": [f"D{i:02d}" for i in range(1, 5)],
    "E": [f"E{i:02d}" for i in range(1, 6)],
    "N": [f"N{i:02d}" for i in range(1, 9)],
}

REGIME_NAMES = {0: "non_toxic", 1: "hate", 2: "offensive"}
SEED_DEFAULT = [729, 1136, 1196, 4468, 5114]

STRATA = [
    "all",
    "non_toxic",
    "hate",
    "offensive",
    "expression_0",
    "expression_1_3",
    "non_toxic_expression_0",
    "hate_expression_0",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_text(s: object) -> str:
    return re.sub(r"\s+", "", str(s))


def parse_target(value: object) -> List[int]:
    """把 'target' 字段解析成 5 维 0/1 列表。"""
    if value is None:
        return [0] * TARGET_DIMS
    if isinstance(value, (list, tuple)):
        arr = list(value)
    else:
        text = str(value).strip()
        if not text:
            return [0] * TARGET_DIMS
        try:
            arr = ast.literal_eval(text)
        except Exception:
            try:
                arr = json.loads(text)
            except Exception:
                return [0] * TARGET_DIMS
    arr = [int(x) for x in list(arr)[:TARGET_DIMS]]
    if len(arr) < TARGET_DIMS:
        arr += [0] * (TARGET_DIMS - len(arr))
    return [1 if x else 0 for x in arr]


def id_suffix(sample_id: object) -> Optional[int]:
    m = re.search(r"_(\d+)$", str(sample_id))
    if not m:
        return None
    return int(m.group(1))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def fmt(v: object, nd: int = 4) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "na"
    if math.isnan(f):
        return "nan"
    return f"{f:.{nd}f}"


def fmt_signed(v: object, nd: int = 4) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "na"
    if math.isnan(f):
        return "nan"
    return f"{f:+.{nd}f}"


def mean_std_text(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return "na"
    if len(vals) == 1:
        return fmt(vals.iloc[0])
    return f"{fmt(vals.mean())}±{fmt(vals.std(ddof=1))}"


def auc_or_nan(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 数据加载与对齐
# ---------------------------------------------------------------------------

def load_concept_matrix(split: str) -> Tuple[pd.DataFrame, List[str]]:
    v8_path = V8_DIR / f"{V8_PREFIX}_{split}_matrix.csv"
    normal_path = NORMAL_DIR / f"{NORMAL_PREFIX}_{split}_matrix.csv"
    for p in (v8_path, normal_path):
        if not p.is_file():
            raise FileNotFoundError(f"找不到概念矩阵：{p}")

    v8 = pd.read_csv(v8_path)
    normal = pd.read_csv(normal_path)

    def _clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
        for c in ("id", "toxic", "content"):
            if c not in df.columns:
                raise ValueError(f"{name} 缺少列 {c}；实际列：{df.columns.tolist()}")
        df = df.copy()
        df["id"] = df["id"].astype(str)
        df["toxic"] = df["toxic"].astype(int)
        return df

    v8 = _clean(v8, "v8")
    normal = _clean(normal, "normal")

    v8_concepts = [c for c in v8.columns if c not in META_COLS]
    normal_concepts = [c for c in normal.columns if c not in META_COLS]

    overlap = set(v8_concepts) & set(normal_concepts)
    if overlap:
        raise ValueError(f"v8 与 normal 概念列重名：{sorted(overlap)}")

    if v8["id"].duplicated().any():
        raise ValueError("v8 概念矩阵存在重复 id。")
    if normal["id"].duplicated().any():
        raise ValueError("normal 概念矩阵存在重复 id。")

    merged = v8.merge(
        normal[["id"] + normal_concepts],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(v8) or len(merged) != len(normal):
        print(
            f"[warn] merge 后行数 {len(merged)}；v8={len(v8)}，normal={len(normal)}。"
            "将只保留交集 id。"
        )

    concept_cols = v8_concepts + normal_concepts
    missing = [c for c in concept_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"merge 后缺少概念列：{missing}")

    merged["_suffix"] = merged["id"].map(id_suffix)
    return merged, concept_cols


def load_raw(split: str) -> List[dict]:
    path = RAW_DIR / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到原始数据：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层不是 list")
    return data


def attach_metadata(df: pd.DataFrame, raw: List[dict], split: str) -> pd.DataFrame:
    """优先按 id 后缀索引对齐；失败时按 content 唯一匹配；再失败则报错。"""
    df = df.copy()
    n = len(raw)
    by_index_ok = True
    for _, row in df.iterrows():
        idx = int(row["_suffix"]) if pd.notna(row["_suffix"]) else -1
        if idx < 0 or idx >= n:
            by_index_ok = False
            break
        if norm_text(row["content"]) != norm_text(raw[idx].get("content", "")):
            by_index_ok = False
            break

    if by_index_ok:
        meta = []
        for _, row in df.iterrows():
            r = raw[int(row["_suffix"])]
            meta.append(
                {
                    "toxic_raw": int(str(r.get("toxic", 0))),
                    "expression": int(str(r.get("expression", 0))),
                    "toxic_type": int(str(r.get("toxic_type", 0))),
                    **{f"target_{j}": v for j, v in enumerate(parse_target(r.get("target")))},
                }
            )
        meta_df = pd.DataFrame(meta, index=df.index)
        df = pd.concat([df, meta_df], axis=1)
    else:
        content_map: Dict[str, int] = {}
        ambiguous = set()
        for i, r in enumerate(raw):
            key = norm_text(r.get("content", ""))
            if key in content_map:
                ambiguous.add(key)
            else:
                content_map[key] = i
        for key in ambiguous:
            content_map.pop(key, None)

        meta_rows = []
        unmatched = 0
        for _, row in df.iterrows():
            key = norm_text(row["content"])
            if key not in content_map:
                meta_rows.append(
                    {
                        "toxic_raw": np.nan,
                        "expression": np.nan,
                        "toxic_type": np.nan,
                        **{f"target_{j}": np.nan for j in range(TARGET_DIMS)},
                    }
                )
                unmatched += 1
                continue
            r = raw[content_map[key]]
            meta_rows.append(
                {
                    "toxic_raw": int(str(r.get("toxic", 0))),
                    "expression": int(str(r.get("expression", 0))),
                    "toxic_type": int(str(r.get("toxic_type", 0))),
                    **{f"target_{j}": v for j, v in enumerate(parse_target(r.get("target")))},
                }
            )
        if unmatched:
            raise ValueError(
                f"{split}: {unmatched} 条样本无法按 content 唯一匹配原始标注；"
                "请检查 id/content 是否被修改。"
            )
        meta_df = pd.DataFrame(meta_rows, index=df.index)
        df = pd.concat([df, meta_df], axis=1)
        print(f"[info] {split}: 使用 content 唯一匹配对齐原始标注。")

    if df["toxic_raw"].isna().any():
        raise ValueError(f"{split}: 存在未匹配的毒性标签。")
    mismatch = (df["toxic"].astype(int) != df["toxic_raw"].astype(int))
    if mismatch.any():
        bad = df.loc[mismatch, ["id", "toxic", "toxic_raw"]].head(5)
        raise ValueError(f"{split}: 概念矩阵与原始数据的 toxic 标签不一致：\n{bad}")

    df["target_any"] = (
        df[[f"target_{j}" for j in range(TARGET_DIMS)]].sum(axis=1) > 0
    ).astype(int)
    return df


def add_regime(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = df["toxic_type"].map({0: 0, 1: 1, 2: 2})
    if df["regime"].isna().any():
        bad = df[df["regime"].isna()]["toxic_type"].value_counts().to_dict()
        raise ValueError(f"{split}: 出现未知 toxic_type：{bad}")
    df["regime"] = df["regime"].astype(int)

    # 结构校验：只报告，不静默改写标签
    bad_toxic = int(
        (
            ((df["regime"] == 0) & (df["toxic"] != 0))
            | ((df["regime"] > 0) & (df["toxic"] != 1))
        ).sum()
    )
    bad_target = int(((df["target_any"] == 1) & (df["regime"] != 2)).sum())
    bad_expr = int(((df["expression"] >= 1) & (df["regime"] != 2)).sum())
    bad_expr0 = int(((df["expression"] == 0) & (df["regime"] == 2)).sum())
    if bad_toxic:
        raise ValueError(f"{split}: toxic 与 toxic_type 存在 {bad_toxic} 条不一致。")
    if bad_target or bad_expr or bad_expr0:
        print(
            f"[warn] {split}: 结构边界异常 target_any=1&regime!=offensive: {bad_target}, "
            f"expression>=1&regime!=offensive: {bad_expr}, "
            f"expression=0&regime=offensive: {bad_expr0}"
        )

    counts = df["regime"].value_counts().sort_index().to_dict()
    print(
        f"[info] {split}: rows={len(df)}，"
        f"toxic_rate={df['toxic'].mean():.4f}，regime={ {REGIME_NAMES[k]: int(v) for k, v in counts.items()} }"
    )
    return df


def load_split(split: str) -> Tuple[pd.DataFrame, List[str]]:
    df, concept_cols = load_concept_matrix(split)
    raw = load_raw(split)
    df = attach_metadata(df, raw, split)
    df = add_regime(df, split)
    return df, concept_cols


def stratified_split_by_regime(
    df: pd.DataFrame, seed: int, val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_parts, val_parts = [], []
    for cls in [0, 1, 2]:
        idx = np.where(df["regime"].to_numpy() == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        n_val = min(max(n_val, 1), max(len(idx) - 1, 0))
        val_parts.append(idx[:n_val])
        tr_parts.append(idx[n_val:])
    tr_idx = np.concatenate(tr_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return tr_idx, val_idx


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class LinearReadout(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GatedReadout(nn.Module):
    def __init__(self, d: int, hidden: int = 96, dropout: float = 0.5):
        super().__init__()
        self.gate = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        h = x * g
        h = self.drop(torch.relu(self.fc1(h)))
        return self.fc2(h)


class BinaryGate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class Combiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TargetHead(nn.Module):
    def __init__(self, d: int, n_targets: int = 4):
        super().__init__()
        self.fc = nn.Linear(d, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(
        f1_score(
            y,
            pred,
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )
    )


def binary_metrics(
    y: np.ndarray, prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    p, r, f1_pos, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "toxic_rate": float(np.mean(y)),
        "mean_prob": float(np.mean(prob)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": macro_f1(y, pred),
        "pos_f1": float(f1_pos),
        "precision": float(p),
        "recall": float(r),
        "auc": auc_or_nan(y, prob),
        "fp_rate_non_toxic": safe_div(fp, fp + tn),
        "fn_rate_toxic": safe_div(fn, fn + tp),
        "n_non_toxic": int(tn + fp),
        "n_toxic": int(fn + tp),
    }


def stratum_metrics(
    y: np.ndarray, prob: np.ndarray, threshold: float, mask: np.ndarray
) -> Dict[str, float]:
    yy = y[mask]
    pp = prob[mask]
    pred = (pp >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    p, r, f1_pos, _ = precision_recall_fscore_support(
        yy, pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "n": int(len(yy)),
        "toxic_rate": float(np.mean(yy)) if len(yy) else float("nan"),
        "mean_prob": float(np.mean(pp)) if len(pp) else float("nan"),
        "macro_f1": macro_f1(yy, pred) if len(yy) else float("nan"),
        "pos_f1": float(f1_pos),
        "precision": float(p),
        "recall": float(r),
        "auc": auc_or_nan(yy, pp),
        "fp_rate_non_toxic": safe_div(fp, fp + tn),
        "fn_rate_toxic": safe_div(fn, fn + tp),
    }


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """在验证集上按 Macro-F1 选阈值；同分时取更接近 0.5 的阈值。不接触 test。"""
    best_t, best_f1 = 0.5, -1.0
    for i in range(91):
        t = round(0.05 + 0.01 * i, 2)
        pred = (prob >= t).astype(int)
        f1 = macro_f1(y, pred)
        better = f1 > best_f1 + 1e-9
        tie_better = abs(f1 - best_f1) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5)
        if better or tie_better:
            best_f1 = f1
            best_t = float(t)
    return best_t


def strata_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    regime = df["regime"].to_numpy()
    expr = df["expression"].to_numpy()
    return {
        "all": np.ones(len(df), dtype=bool),
        "non_toxic": regime == 0,
        "hate": regime == 1,
        "offensive": regime == 2,
        "expression_0": expr == 0,
        "expression_1_3": expr >= 1,
        "non_toxic_expression_0": (regime == 0) & (expr == 0),
        "hate_expression_0": (regime == 1) & (expr == 0),
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


def train_flat(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    model_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    set_seed(seed)
    d = x_tr.shape[1]
    model: nn.Module = LinearReadout(d) if model_type == "linear" else GatedReadout(d)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    xtr = _tensor(x_tr, device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    xva = _tensor(x_val, device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    xte = _tensor(x_te, device)

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
            val_prob = torch.softmax(model(xva), dim=1)[:, 1].cpu().numpy()
        val_f1 = macro_f1(y_val, (val_prob >= 0.5).astype(int))
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        val_prob = torch.softmax(model(xva), dim=1)[:, 1].cpu().numpy()
        test_prob = torch.softmax(model(xte), dim=1)[:, 1].cpu().numpy()
    return {
        "model": model,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "best_epoch": best_epoch,
    }


def train_gate(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    set_seed(seed)
    d = x_tr.shape[1]
    gate = BinaryGate(d).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_pos = float(np.sum(y_tr))
    n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor(
        [max(n_neg / max(n_pos, 1.0), 1.0)], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xtr = _tensor(x_tr, device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    xva = _tensor(x_val, device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_state = None
    best_score = -1e9
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        gate.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(gate(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        gate.eval()
        with torch.no_grad():
            val_logits = gate(xva)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = float(loss_fn(val_logits, yva).item())
        val_auc = auc_or_nan(y_val.astype(int), val_prob)
        score = val_auc if math.isfinite(val_auc) else -val_loss
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    return {
        "model": gate,
        "best_epoch": best_epoch,
        "pos_weight": float(pos_weight.item()),
    }


def gate_logits(
    gate: nn.Module, x: np.ndarray, device: torch.device
) -> np.ndarray:
    gate.eval()
    with torch.no_grad():
        return gate(_tensor(x, device)).cpu().numpy()


def train_combiner(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_te: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    set_seed(seed)
    model = Combiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    ztr = _tensor(z_tr, device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    zva = _tensor(z_val, device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    zte = _tensor(z_te, device)

    loader = DataLoader(
        TensorDataset(ztr, ytr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for zb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(zb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.softmax(model(zva), dim=1)[:, 1].cpu().numpy()
        val_f1 = macro_f1(y_val, (val_prob >= 0.5).astype(int))
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        val_prob = torch.softmax(model(zva), dim=1)[:, 1].cpu().numpy()
        test_prob = torch.softmax(model(zte), dim=1)[:, 1].cpu().numpy()
    return {
        "model": model,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val,
    }


def train_target_head(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    set_seed(seed)
    d = x_tr.shape[1]
    n_targets = y_tr.shape[1]
    head = TargetHead(d, n_targets).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos = y_tr.sum(axis=0).astype(float)
    neg = len(y_tr) - pos
    pos_weight = torch.tensor(
        np.clip(neg / np.maximum(pos, 1.0), 1.0, 50.0),
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xtr = _tensor(x_tr, device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    xva = _tensor(x_val, device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    xte = _tensor(x_te, device)

    loader = DataLoader(
        TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_state = None
    best_score = -1e9
    best_epoch = 0
    bad = 0

    for epoch in range(1, epochs + 1):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(xva)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = float(loss_fn(val_logits, yva).item())
        aucs = []
        for j in range(n_targets):
            a = auc_or_nan(y_val[:, j].astype(int), val_prob[:, j])
            if math.isfinite(a):
                aucs.append(a)
        score = float(np.mean(aucs)) if aucs else -val_loss
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(head(xva)).cpu().numpy()
        test_prob = torch.sigmoid(head(xte)).cpu().numpy()
    return {
        "model": head,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Regime-aware 预测与敏感性
# ---------------------------------------------------------------------------

def predict_regime(
    bundle: Dict[str, object], x_raw: np.ndarray, device: torch.device
) -> np.ndarray:
    hate_gate: nn.Module = bundle["hate_gate"]  # type: ignore[assignment]
    off_gate: nn.Module = bundle["off_gate"]  # type: ignore[assignment]
    combiner: nn.Module = bundle["combiner"]  # type: ignore[assignment]
    hate_scaler: StandardScaler = bundle["hate_scaler"]  # type: ignore[assignment]
    off_scaler: StandardScaler = bundle["off_scaler"]  # type: ignore[assignment]
    hate_idx: List[int] = bundle["hate_idx"]  # type: ignore[assignment]
    off_idx: List[int] = bundle["off_idx"]  # type: ignore[assignment]

    z_h = gate_logits(hate_gate, hate_scaler.transform(x_raw[:, hate_idx]), device)
    z_o = gate_logits(off_gate, off_scaler.transform(x_raw[:, off_idx]), device)
    z = torch.tensor(np.stack([z_h, z_o], axis=1), dtype=torch.float32, device=device)
    combiner.eval()
    with torch.no_grad():
        return torch.softmax(combiner(z), dim=1)[:, 1].cpu().numpy()


def _random_control(
    bundle: Dict[str, object],
    x_raw: np.ndarray,
    mask: np.ndarray,
    base_prob: np.ndarray,
    candidate_idx: Sequence[int],
    k: int,
    value: float,
    device: torch.device,
    n_random: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if k <= 0 or len(candidate_idx) < k:
        return {
            "mean_delta_p": float("nan"),
            "mean_abs_delta_p": float("nan"),
            "flip_rate": float("nan"),
            "new_mean_prob": float("nan"),
        }
    rows = np.where(mask)[0]
    deltas, abs_deltas, flips, new_means = [], [], [], []
    for _ in range(n_random):
        idxs = rng.choice(list(candidate_idx), size=k, replace=False).tolist()
        x = x_raw.copy()
        x[np.ix_(rows, idxs)] = value
        p = predict_regime(bundle, x, device)
        d = p[mask] - base_prob[mask]
        deltas.append(float(np.mean(d)))
        abs_deltas.append(float(np.mean(np.abs(d))))
        flips.append(float(np.mean((p[mask] >= 0.5) != (base_prob[mask] >= 0.5))))
        new_means.append(float(np.mean(p[mask])))
    return {
        "mean_delta_p": float(np.mean(deltas)),
        "mean_abs_delta_p": float(np.mean(abs_deltas)),
        "flip_rate": float(np.mean(flips)),
        "new_mean_prob": float(np.mean(new_means)),
    }


def run_sensitivity(
    bundle: Dict[str, object],
    test_df: pd.DataFrame,
    concept_cols: List[str],
    device: torch.device,
    n_random: int = 20,
) -> pd.DataFrame:
    x_raw = test_df[concept_cols].to_numpy(dtype=float)
    base_prob = predict_regime(bundle, x_raw, device)
    regime = test_df["regime"].to_numpy()
    rng = np.random.default_rng(0)

    block_map = {
        name: [c for c in cols if c in concept_cols]
        for name, cols in BLOCK_DEFS.items()
    }

    t_intervene = [c for c in block_map["T"] if c != T_EXCLUDED]
    analyses = [
        ("T_off_offensive", regime == 2, t_intervene, 0.0),
        ("T_on_offensive", regime == 2, t_intervene, 1.0),
        ("A_off_hate", regime == 1, block_map["A"], 0.0),
        ("A_on_hate", regime == 1, block_map["A"], 1.0),
        ("N_off_hate", regime == 1, block_map["N"], 0.0),
        ("N_on_hate", regime == 1, block_map["N"], 1.0),
    ]

    rows = []
    for name, mask, cols, value in analyses:
        if mask.sum() == 0:
            continue
        idxs = [concept_cols.index(c) for c in cols]
        if not idxs:
            continue
        rows_mask = np.where(mask)[0]
        x = x_raw.copy()
        x[np.ix_(rows_mask, idxs)] = value
        p = predict_regime(bundle, x, device)
        d = p[mask] - base_prob[mask]
        rows.append(
            {
                "analysis": name,
                "n": int(mask.sum()),
                "base_mean_prob": float(np.mean(base_prob[mask])),
                "new_mean_prob": float(np.mean(p[mask])),
                "mean_delta_p": float(np.mean(d)),
                "mean_abs_delta_p": float(np.mean(np.abs(d))),
                "flip_rate": float(np.mean((p[mask] >= 0.5) != (base_prob[mask] >= 0.5))),
            }
        )

        block_set = set(cols)
        candidate_idx = [
            i for i, c in enumerate(concept_cols) if c not in block_set
        ]
        rand = _random_control(
            bundle, x_raw, mask, base_prob, candidate_idx, len(idxs), value,
            device, n_random, rng,
        )
        rows.append(
            {
                "analysis": f"random_{name}",
                "n": int(mask.sum()),
                "base_mean_prob": float(np.mean(base_prob[mask])),
                **rand,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def paired_delta(
    metrics_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    policy: str,
    col: str,
) -> Tuple[float, float]:
    a = (
        metrics_df[
            (metrics_df["model"] == model_a)
            & (metrics_df["threshold_policy"] == policy)
        ]
        .set_index("seed")[col]
        .astype(float)
    )
    b = (
        metrics_df[
            (metrics_df["model"] == model_b)
            & (metrics_df["threshold_policy"] == policy)
        ]
        .set_index("seed")[col]
        .astype(float)
    )
    d = (a - b).dropna()
    if len(d) == 0:
        return float("nan"), float("nan")
    std = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return float(d.mean()), std


def verdict_improve(dm: float, ds: float) -> str:
    if not math.isfinite(dm):
        return "无法比较"
    if dm > 0 and dm > max(ds, 0.002):
        return "稳定提升"
    if dm > 0:
        return "方向为正，但幅度小或不超过 seed 波动"
    if dm > -max(ds, 0.002):
        return "无稳定差异"
    return "下降"


def verdict_lower(dm: float, ds: float, label: str) -> str:
    if not math.isfinite(dm):
        return f"{label} 无法比较"
    if dm < 0 and abs(dm) > max(ds, 0.002):
        return f"{label} 稳定降低"
    if dm < 0:
        return f"{label} 略降但不稳定"
    if dm < max(ds, 0.002):
        return f"{label} 无稳定差异"
    return f"{label} 上升"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_DEFAULT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--gate_epochs", type=int, default=150)
    parser.add_argument("--target_epochs", type=int, default=120)
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

    train_df, concept_cols = load_split("train")
    test_df, test_concepts = load_split("test")
    if concept_cols != test_concepts:
        raise ValueError("train/test 概念列顺序不一致，请先对齐。")

    # 概念区块只保留实际存在的列
    block_cols = {
        name: [c for c in cols if c in concept_cols]
        for name, cols in BLOCK_DEFS.items()
    }
    missing_t = [c for c in T_ALL if c not in concept_cols]
    if missing_t:
        raise ValueError(f"概念矩阵缺少 T 列：{missing_t}")
    if T_EXCLUDED not in concept_cols:
        raise ValueError(f"概念矩阵缺少 {T_EXCLUDED}；本脚本按既定协议需要它作为排除项。")

    hate_features = [c for c in concept_cols if c not in set(T_ALL)]
    offensive_features = [c for c in concept_cols if c != T_EXCLUDED]
    hate_idx = [concept_cols.index(c) for c in hate_features]
    offensive_idx = [concept_cols.index(c) for c in offensive_features]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录：{out_dir}")

    # ------------------------------------------------------------------
    # 数据概览与 split
    # ------------------------------------------------------------------
    tr_idx, val_idx = stratified_split_by_regime(
        train_df, seed=split_seed, val_ratio=args.val_ratio
    )
    x_train_raw = train_df[concept_cols].to_numpy(dtype=float)
    x_test_raw = test_df[concept_cols].to_numpy(dtype=float)
    y_train = train_df["toxic"].to_numpy(dtype=int)
    y_test = test_df["toxic"].to_numpy(dtype=int)

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
        "train_expression_counts": {
            str(k): int(v)
            for k, v in train_df["expression"].value_counts().sort_index().items()
        },
        "test_expression_counts": {
            str(k): int(v)
            for k, v in test_df["expression"].value_counts().sort_index().items()
        },
        "train_target_any_rate": float(train_df["target_any"].mean()),
        "test_target_any_rate": float(test_df["target_any"].mean()),
        "split_seed": int(split_seed),
        "val_ratio": float(args.val_ratio),
        "seeds": seeds,
        "hate_features": hate_features,
        "offensive_features": offensive_features,
    }
    with (out_dir / "overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    strata = strata_masks(test_df)
    test_regime = test_df["regime"].to_numpy()

    # ------------------------------------------------------------------
    # 标准化器：只在 train 上 fit
    # ------------------------------------------------------------------
    flat_scaler = StandardScaler().fit(x_train_raw[tr_idx])
    x_tr_flat = flat_scaler.transform(x_train_raw[tr_idx])
    x_val_flat = flat_scaler.transform(x_train_raw[val_idx])
    x_te_flat = flat_scaler.transform(x_test_raw)

    gate_h_tr_mask = np.isin(train_df["regime"].to_numpy(), [0, 1])
    gate_o_tr_mask = np.isin(train_df["regime"].to_numpy(), [0, 2])
    gate_h_val_mask = np.isin(train_df["regime"].to_numpy()[val_idx], [0, 1])
    gate_o_val_mask = np.isin(train_df["regime"].to_numpy()[val_idx], [0, 2])

    tr_h_idx = tr_idx[gate_h_tr_mask[tr_idx]]
    val_h_idx = val_idx[gate_h_val_mask]
    tr_o_idx = tr_idx[gate_o_tr_mask[tr_idx]]
    val_o_idx = val_idx[gate_o_val_mask]

    hate_scaler = StandardScaler().fit(x_train_raw[tr_h_idx][:, hate_idx])
    off_scaler = StandardScaler().fit(x_train_raw[tr_o_idx][:, offensive_idx])

    y_h_tr = (train_df["regime"].to_numpy()[tr_h_idx] == 1).astype(int)
    y_h_val = (train_df["regime"].to_numpy()[val_h_idx] == 1).astype(int)
    y_o_tr = (train_df["regime"].to_numpy()[tr_o_idx] == 2).astype(int)
    y_o_val = (train_df["regime"].to_numpy()[val_o_idx] == 2).astype(int)

    # target 辅助头：只在 offensive 训练样本上 fit
    tr_off_mask = train_df["regime"].to_numpy() == 2
    val_off_mask = train_df["regime"].to_numpy()[val_idx] == 2
    tr_off_idx = tr_idx[tr_off_mask[tr_idx]]
    val_off_idx = val_idx[val_off_mask]

    target_scaler = StandardScaler().fit(x_train_raw[tr_off_idx])
    y_t_tr = train_df.loc[tr_off_idx, [f"target_{j}" for j in range(4)]].to_numpy(dtype=float)
    y_t_val = train_df.loc[val_off_idx, [f"target_{j}" for j in range(4)]].to_numpy(dtype=float)

    test_off_mask = test_df["regime"].to_numpy() == 2
    y_t_test = test_df.loc[test_off_mask, [f"target_{j}" for j in range(4)]].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 逐 seed 训练
    # ------------------------------------------------------------------
    run_rows: List[Dict[str, object]] = []
    strata_rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []
    seed_test_probs: Dict[str, Dict[int, np.ndarray]] = {
        "flat_linear": {},
        "flat_gated": {},
        "regime_aware": {},
    }
    seed_gate_probs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    first_bundle: Optional[Dict[str, object]] = None

    for seed in tqdm(seeds, desc="seeds", leave=False):
        # ---- baselines ----
        flat_out: Dict[str, Dict[str, object]] = {}
        for model_type in ["linear", "gated"]:
            out = train_flat(
                x_tr_flat,
                y_train[tr_idx],
                x_val_flat,
                y_train[val_idx],
                x_te_flat,
                model_type=model_type,
                seed=seed,
                epochs=args.epochs,
                device=device,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            name = f"flat_{model_type}"
            flat_out[name] = out
            seed_test_probs[name][seed] = out["test_prob"]  # type: ignore[index]

        # ---- Stage 1: gates ----
        gate_h = train_gate(
            hate_scaler.transform(x_train_raw[tr_h_idx][:, hate_idx]),
            y_h_tr,
            hate_scaler.transform(x_train_raw[val_h_idx][:, hate_idx]),
            y_h_val,
            seed=seed,
            epochs=args.gate_epochs,
            device=device,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        gate_o = train_gate(
            off_scaler.transform(x_train_raw[tr_o_idx][:, offensive_idx]),
            y_o_tr,
            off_scaler.transform(x_train_raw[val_o_idx][:, offensive_idx]),
            y_o_val,
            seed=seed,
            epochs=args.gate_epochs,
            device=device,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        def _gate_logits_all(gate: Dict[str, object], scaler: StandardScaler, idx: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            model = gate["model"]
            z_tr = gate_logits(model, scaler.transform(x_train_raw[tr_idx][:, idx]), device)  # type: ignore[arg-type]
            z_val = gate_logits(model, scaler.transform(x_train_raw[val_idx][:, idx]), device)  # type: ignore[arg-type]
            z_te = gate_logits(model, scaler.transform(x_test_raw[:, idx]), device)  # type: ignore[arg-type]
            return z_tr, z_val, z_te

        zh_tr, zh_val, zh_te = _gate_logits_all(gate_h, hate_scaler, hate_idx)
        zo_tr, zo_val, zo_te = _gate_logits_all(gate_o, off_scaler, offensive_idx)

        # ---- Stage 2: combiner ----
        z_tr = np.stack([zh_tr, zo_tr], axis=1)
        z_val = np.stack([zh_val, zo_val], axis=1)
        z_te = np.stack([zh_te, zo_te], axis=1)
        comb = train_combiner(
            z_tr,
            y_train[tr_idx],
            z_val,
            y_train[val_idx],
            z_te,
            seed=seed,
            epochs=args.epochs,
            device=device,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        seed_test_probs["regime_aware"][seed] = comb["test_prob"]  # type: ignore[index]

        hate_prob_te = 1.0 / (1.0 + np.exp(-np.clip(zh_te, -60.0, 60.0)))
        off_prob_te = 1.0 / (1.0 + np.exp(-np.clip(zo_te, -60.0, 60.0)))
        seed_gate_probs[seed] = (hate_prob_te, off_prob_te)

        if first_bundle is None:
            first_bundle = {
                "hate_gate": gate_h["model"],
                "hate_scaler": hate_scaler,
                "hate_idx": hate_idx,
                "off_gate": gate_o["model"],
                "off_scaler": off_scaler,
                "off_idx": offensive_idx,
                "combiner": comb["model"],
            }

        # ---- target 辅助头 ----
        tgt = train_target_head(
            target_scaler.transform(x_train_raw[tr_off_idx]),
            y_t_tr,
            target_scaler.transform(x_train_raw[val_off_idx]),
            y_t_val,
            target_scaler.transform(x_test_raw[test_off_mask]),
            seed=seed,
            epochs=args.target_epochs,
            device=device,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        tgt_test_prob = np.asarray(tgt["test_prob"])

        for j in range(4):
            yj = y_t_test[:, j].astype(int)
            pj = tgt_test_prob[:, j]
            predj = (pj >= 0.5).astype(int)
            p_, r_, f1_, _ = precision_recall_fscore_support(
                yj, predj, average="binary", pos_label=1, zero_division=0
            )
            target_rows.append(
                {
                    "seed": seed,
                    "target_dim": f"target_{j}",
                    "n_pos": int(yj.sum()),
                    "n": int(len(yj)),
                    "threshold": 0.5,
                    "recall": float(r_),
                    "precision": float(p_),
                    "f1": float(f1_),
                    "auc": auc_or_nan(yj, pj),
                    "mean_prob": float(np.mean(pj)),
                }
            )

        # ---- gate 自身错误 ----
        non_toxic_mask = test_regime == 0
        hate_mask = test_regime == 1
        off_mask = test_regime == 2
        hate_gate_fp = float(np.mean(hate_prob_te[non_toxic_mask] >= 0.5))
        hate_gate_fn = float(np.mean(hate_prob_te[hate_mask] < 0.5))
        off_gate_fp = float(np.mean(off_prob_te[non_toxic_mask] >= 0.5))
        off_gate_fn = float(np.mean(off_prob_te[off_mask] < 0.5))
        y_h_te = (test_regime == 1).astype(int)
        h_te_sel = np.isin(test_regime, [0, 1])
        y_o_te = (test_regime == 2).astype(int)
        o_te_sel = np.isin(test_regime, [0, 2])
        hate_gate_auc = auc_or_nan(y_h_te[h_te_sel], hate_prob_te[h_te_sel])
        off_gate_auc = auc_or_nan(y_o_te[o_te_sel], off_prob_te[o_te_sel])

        # ---- 指标：两个阈值策略 ----
        model_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "flat_linear": (flat_out["flat_linear"]["val_prob"], flat_out["flat_linear"]["test_prob"]),  # type: ignore[index]
            "flat_gated": (flat_out["flat_gated"]["val_prob"], flat_out["flat_gated"]["test_prob"]),  # type: ignore[index]
            "regime_aware": (comb["val_prob"], comb["test_prob"]),  # type: ignore[index]
        }
        best_epochs = {
            "flat_linear": flat_out["flat_linear"]["best_epoch"],
            "flat_gated": flat_out["flat_gated"]["best_epoch"],
            "regime_aware": comb["best_epoch"],
        }

        for model_name, (val_prob, test_prob) in model_probs.items():
            val_prob = np.asarray(val_prob)
            test_prob = np.asarray(test_prob)
            t_opt = best_threshold(y_train[val_idx], val_prob)
            policies = {"fixed_0.5": 0.5, "val_opt": t_opt}

            for policy, t in policies.items():
                m = binary_metrics(y_test, test_prob, t)
                y_val = y_train[val_idx]
                val_pred = (val_prob >= t).astype(int)
                row = {
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
                    "hate_fn_rate": float(np.mean(test_prob[hate_mask] < t)),
                    "offensive_fn_rate": float(np.mean(test_prob[off_mask] < t)),
                    "val_macro_f1_05": macro_f1(y_val, (val_prob >= 0.5).astype(int)),
                    "val_macro_f1_selected": macro_f1(y_val, val_pred),
                    "best_epoch": best_epochs[model_name],
                    "hate_gate_fp_non_toxic": hate_gate_fp if model_name == "regime_aware" else np.nan,
                    "hate_gate_fn_hate": hate_gate_fn if model_name == "regime_aware" else np.nan,
                    "hate_gate_auc": hate_gate_auc if model_name == "regime_aware" else np.nan,
                    "offensive_gate_fp_non_toxic": off_gate_fp if model_name == "regime_aware" else np.nan,
                    "offensive_gate_fn_offensive": off_gate_fn if model_name == "regime_aware" else np.nan,
                    "offensive_gate_auc": off_gate_auc if model_name == "regime_aware" else np.nan,
                    "hate_gate_best_epoch": gate_h["best_epoch"],
                    "offensive_gate_best_epoch": gate_o["best_epoch"],
                    "combiner_best_epoch": comb["best_epoch"],
                    "target_head_best_epoch": tgt["best_epoch"],
                }
                run_rows.append(row)

                for stratum_name in STRATA:
                    mask = strata[stratum_name]
                    if mask.sum() == 0:
                        continue
                    sm = stratum_metrics(y_test, test_prob, t, mask)
                    strata_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "threshold_policy": policy,
                            "stratum": stratum_name,
                            **sm,
                        }
                    )

        # ---- gate / combiner 系数 ----
        for comp_name, gate_obj, feats in [
            ("hate_gate", gate_h["model"], hate_features),
            ("offensive_gate", gate_o["model"], offensive_features),
        ]:
            w = gate_obj.fc.weight.detach().cpu().numpy().ravel()  # type: ignore[attr-defined]
            b = float(gate_obj.fc.bias.detach().cpu().numpy().ravel()[0])  # type: ignore[attr-defined]
            for feat, wi in zip(feats, w):
                coef_rows.append(
                    {"seed": seed, "component": comp_name, "term": feat, "weight": float(wi)}
                )
            coef_rows.append(
                {"seed": seed, "component": comp_name, "term": "_bias", "weight": b}
            )
        cw = comb["model"].fc.weight.detach().cpu().numpy()  # type: ignore[union-attr]
        cb = comb["model"].fc.bias.detach().cpu().numpy()  # type: ignore[union-attr]
        coef_rows.append(
            {"seed": seed, "component": "combiner", "term": "w_hate", "weight": float(cw[1, 0])}
        )
        coef_rows.append(
            {"seed": seed, "component": "combiner", "term": "w_offensive", "weight": float(cw[1, 1])}
        )
        coef_rows.append(
            {"seed": seed, "component": "combiner", "term": "_bias", "weight": float(cb[1])}
        )

    # ------------------------------------------------------------------
    # 汇总表
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(run_rows)
    metrics_df.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")

    agg_cols = [
        "acc", "macro_f1", "pos_f1", "precision", "recall", "auc",
        "fp_rate_non_toxic", "fn_rate_toxic", "hate_fn_rate", "offensive_fn_rate",
        "hate_gate_fp_non_toxic", "hate_gate_fn_hate", "hate_gate_auc",
        "offensive_gate_fp_non_toxic", "offensive_gate_fn_offensive", "offensive_gate_auc",
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

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(out_dir / "target_recall.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / "gate_coefficients.csv", index=False, encoding="utf-8-sig")

    # 敏感性（第一个 seed 的 regime-aware 模型）
    if first_bundle is not None:
        sens_df = run_sensitivity(first_bundle, test_df, concept_cols, device, n_random=20)
        sens_df.to_csv(out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    else:
        sens_df = pd.DataFrame()

    # test 预测：跨 seed 平均概率
    pred_df = test_df[
        ["id", "content", "toxic", "expression", "toxic_type", "target_any"]
        + [f"target_{j}" for j in range(TARGET_DIMS)]
    ].copy()
    for model_name in ["flat_linear", "flat_gated", "regime_aware"]:
        arr = np.stack([seed_test_probs[model_name][s] for s in seeds], axis=0)
        pred_df[f"{model_name}_prob_mean"] = arr.mean(axis=0)
        pred_df[f"{model_name}_prob_std"] = arr.std(axis=0, ddof=1 if len(seeds) > 1 else 0)
        pred_df[f"{model_name}_pred_05"] = (arr.mean(axis=0) >= 0.5).astype(int)
    if seed_gate_probs:
        h_arr = np.stack([seed_gate_probs[s][0] for s in seeds], axis=0)
        o_arr = np.stack([seed_gate_probs[s][1] for s in seeds], axis=0)
        pred_df["hate_gate_prob_mean"] = h_arr.mean(axis=0)
        pred_df["offensive_gate_prob_mean"] = o_arr.mean(axis=0)
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
        return f"{fmt(m)}±{fmt(s)}"

    lines: List[str] = []
    lines.append("Regime-aware CBM 训练与评估报告")
    lines.append("=" * 68)
    lines.append(f"时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"device：{device}")
    lines.append(f"seeds：{seeds}")
    lines.append(
        f"split：按 regime 分层 {1 - args.val_ratio:.0%}/{args.val_ratio:.0%}，"
        f"split_seed={split_seed}；test 不参与训练/阈值选择"
    )
    lines.append(
        f"概念：v8 26 + normal 8 = {len(concept_cols)} 维；"
        f"hate gate 用 {len(hate_features)} 维（排除 T01–T05），"
        f"offensive gate 用 {len(offensive_features)} 维（排除 T05）"
    )
    lines.append(
        "训练：flat/gate/target 用 AdamW lr=1e-3；2 维 combiner 用 lr=5e-3；"
        "batch=256，patience=20；全部 StandardScaler 只在 train 上 fit。"
    )
    lines.append("")

    lines.append("一、数据与标注 regime")
    lines.append("-" * 68)
    lines.append(f"train rows={len(train_df)}，test rows={len(test_df)}")
    for name in ["non_toxic", "hate", "offensive"]:
        lines.append(
            f"  {name:<10} train={overview['train_regime_counts'].get(name, 0):<5} "
            f"test={overview['test_regime_counts'].get(name, 0):<5} "
            f"（test 占比={overview['test_regime_counts'].get(name, 0) / len(test_df):.4f}）"
        )
    lines.append("结构校验：")
    lines.append(
        f"  target_any=1 & regime!=offensive："
        f"{int(((test_df['target_any'] == 1) & (test_df['regime'] != 2)).sum())}（test）"
    )
    lines.append(
        f"  expression>=1 & regime!=offensive："
        f"{int(((test_df['expression'] >= 1) & (test_df['regime'] != 2)).sum())}（test）"
    )
    lines.append(
        f"  expression=0 & regime in {{non_toxic,hate}}："
        f"{int(((test_df['expression'] == 0) & (test_df['regime'].isin([0, 1]))).sum())}（test）"
    )
    lines.append(
        f"  expression=0 & toxic=1："
        f"{int(((test_df['expression'] == 0) & (test_df['toxic'] == 1)).sum())}（test，hate block）"
    )
    lines.append(
        f"  expression=0 & toxic=0："
        f"{int(((test_df['expression'] == 0) & (test_df['toxic'] == 0)).sum())}（test，non_toxic）"
    )
    lines.append(
        "  → non_toxic 与 hate 都落在 expression=0，是扁平模型的主要混淆层。"
    )
    lines.append("")

    lines.append("二、整体指标（threshold=0.5；seed 平均±std）")
    lines.append("-" * 68)
    lines.append(
        f"{'model':<14} {'Acc':<17} {'MacroF1':<17} {'PosF1':<17} {'AUC':<17}"
        f"{'FP_rate(non_toxic)':<19} {'FN_rate(hate)':<17} {'FN_rate(offensive)'}"
    )
    for model_name in ["flat_linear", "flat_gated", "regime_aware"]:
        lines.append(
            f"{model_name:<14} "
            f"{ms_text(model_name, 'fixed_0.5', 'acc'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'macro_f1'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'pos_f1'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'auc'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'fp_rate_non_toxic'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'hate_fn_rate'):<17} "
            f"{ms_text(model_name, 'fixed_0.5', 'offensive_fn_rate')}"
        )
    lines.append("")

    lines.append("三、验证集最优阈值（val-opt）下的指标")
    lines.append("-" * 68)
    lines.append(
        f"{'model':<14} {'MacroF1':<17} {'PosF1':<17} {'AUC':<17} {'threshold'}"
    )
    for model_name in ["flat_linear", "flat_gated", "regime_aware"]:
        lines.append(
            f"{model_name:<14} "
            f"{ms_text(model_name, 'val_opt', 'macro_f1'):<17} "
            f"{ms_text(model_name, 'val_opt', 'pos_f1'):<17} "
            f"{ms_text(model_name, 'val_opt', 'auc'):<17} "
            f"{ms_text(model_name, 'val_opt', 'threshold')}"
        )
    lines.append("")

    lines.append("四、与 flat baseline 的配对差值（同 seed、同 split；负的 FP/FN 差值更好）")
    lines.append("-" * 68)
    for base_name in ["flat_linear", "flat_gated"]:
        dm, ds = paired_delta(metrics_df, "regime_aware", base_name, "fixed_0.5", "macro_f1")
        da, das = paired_delta(metrics_df, "regime_aware", base_name, "fixed_0.5", "auc")
        dfp, dfps = paired_delta(metrics_df, "regime_aware", base_name, "fixed_0.5", "fp_rate_non_toxic")
        dfn, dfns = paired_delta(metrics_df, "regime_aware", base_name, "fixed_0.5", "hate_fn_rate")
        don, dons = paired_delta(metrics_df, "regime_aware", base_name, "fixed_0.5", "offensive_fn_rate")
        lines.append(
            f"regime_aware - {base_name}（0.5）："
            f"ΔMacroF1={fmt_signed(dm)}±{fmt(ds)}，ΔAUC={fmt_signed(da)}±{fmt(das)}，"
            f"ΔFP(non_toxic)={fmt_signed(dfp)}±{fmt(dfps)}，"
            f"ΔFN(hate)={fmt_signed(dfn)}±{fmt(dfns)}，"
            f"ΔFN(offensive)={fmt_signed(don)}±{fmt(dons)}"
        )
        dm_opt, ds_opt = paired_delta(metrics_df, "regime_aware", base_name, "val_opt", "macro_f1")
        lines.append(
            f"  val-opt 下：ΔMacroF1={fmt_signed(dm_opt)}±{fmt(ds_opt)}"
        )
    lines.append("")

    lines.append("五、regime 分层（threshold=0.5；seed 平均）")
    lines.append("-" * 68)
    lines.append(
        f"{'stratum':<26} {'model':<14} {'n':<6} {'toxic':<7} "
        f"{'MacroF1':<10} {'PosF1':<10} {'FP_rate':<10} {'FN_rate'}"
    )
    strata_05 = strata_df[strata_df["threshold_policy"] == "fixed_0.5"]
    for stratum_name in STRATA:
        for model_name in ["flat_linear", "flat_gated", "regime_aware"]:
            sub = strata_05[
                (strata_05["stratum"] == stratum_name)
                & (strata_05["model"] == model_name)
            ]
            if len(sub) == 0:
                continue
            lines.append(
                f"{stratum_name:<26} {model_name:<14} {int(sub['n'].iloc[0]):<6} "
                f"{fmt(sub['toxic_rate'].mean()):<7} "
                f"{fmt(sub['macro_f1'].mean()):<10} "
                f"{fmt(sub['pos_f1'].mean()):<10} "
                f"{fmt(sub['fp_rate_non_toxic'].mean()):<10} "
                f"{fmt(sub['fn_rate_toxic'].mean())}"
            )
    lines.append("")

    lines.append("六、gate 行为与系数（threshold=0.5；seed 平均）")
    lines.append("-" * 68)
    lines.append(
        f"hate gate      ：FP(non_toxic)={ms_text('regime_aware', 'fixed_0.5', 'hate_gate_fp_non_toxic')}，"
        f"FN(hate)={ms_text('regime_aware', 'fixed_0.5', 'hate_gate_fn_hate')}，"
        f"AUC(non_toxic vs hate)={ms_text('regime_aware', 'fixed_0.5', 'hate_gate_auc')}"
    )
    lines.append(
        f"offensive gate ：FP(non_toxic)={ms_text('regime_aware', 'fixed_0.5', 'offensive_gate_fp_non_toxic')}，"
        f"FN(offensive)={ms_text('regime_aware', 'fixed_0.5', 'offensive_gate_fn_offensive')}，"
        f"AUC(non_toxic vs offensive)={ms_text('regime_aware', 'fixed_0.5', 'offensive_gate_auc')}"
    )
    coef_mean = coef_df.groupby(["component", "term"], as_index=False)["weight"].mean()
    for comp in ["hate_gate", "offensive_gate"]:
        sub = coef_mean[coef_mean["component"] == comp].copy()
        sub["abs"] = sub["weight"].abs()
        sub = sub.sort_values("abs", ascending=False).head(5)
        top = "，".join(f"{r['term']}={r['weight']:+.3f}" for _, r in sub.iterrows())
        lines.append(f"  {comp} top5 |w|（标准化特征）：{top}")
    comb_sub = coef_mean[coef_mean["component"] == "combiner"].set_index("term")["weight"]
    lines.append(
        "  combiner（对 toxic 类）："
        f"w_hate={fmt_signed(comb_sub.get('w_hate', float('nan')))}，"
        f"w_offensive={fmt_signed(comb_sub.get('w_offensive', float('nan')))}，"
        f"bias={fmt_signed(comb_sub.get('_bias', float('nan')))}"
    )
    lines.append("")

    lines.append("七、target 辅助头（仅 offensive 样本；threshold=0.5；seed 平均）")
    lines.append("-" * 68)
    for j in range(4):
        sub = target_df[target_df["target_dim"] == f"target_{j}"]
        name_map = {0: "LGBTQ", 1: "region", 2: "gender", 3: "race"}
        lines.append(
            f"target_{j} ({name_map[j]:<6})：n_pos={int(sub['n_pos'].iloc[0]) if len(sub) else 0:<5} "
            f"Recall={mean_std_text(sub['recall'])} "
            f"Precision={mean_std_text(sub['precision'])} "
            f"F1={mean_std_text(sub['f1'])} "
            f"AUC={mean_std_text(sub['auc'])}"
        )
    recall_means = [
        target_df[target_df["target_dim"] == f"target_{j}"]["recall"].mean()
        for j in range(4)
    ]
    lines.append(
        f"4 维平均 recall={fmt(float(np.mean(recall_means)))}；"
        "该辅助头不影响二分类预测。"
    )
    lines.append(
        "target_4(other) 已排除：T05 与 raw target 的最佳匹配 AUC≈0.56，"
        "不满足概念有效性要求。"
    )
    lines.append("")

    lines.append("八、概念向量敏感性（seed=第一个 seed；非因果干预）")
    lines.append("-" * 68)
    if len(sens_df):
        lines.append(
            f"{'analysis':<34} {'n':<6} {'base_p':<9} {'new_p':<9} {'mean_dp':<10} {'|dp|':<9} {'flip'}"
        )
        for _, r in sens_df.iterrows():
            lines.append(
                f"{r['analysis']:<34} {int(r['n']):<6} "
                f"{fmt(r['base_mean_prob']):<9} {fmt(r['new_mean_prob']):<9} "
                f"{fmt(r['mean_delta_p']):<10} {fmt(r['mean_abs_delta_p']):<9} "
                f"{fmt(r['flip_rate'])}"
            )
        lines.append(
            "  说明：这是把概念向量置 0/1 的敏感性分析，不是有 ground-truth 概念的因果干预；"
            "random_* 行是同样数量的随机非该区块概念干预。"
        )
    else:
        lines.append("（未生成）")
    lines.append("")

    # ------------------------------------------------------------------
    # 结论
    # ------------------------------------------------------------------
    dm_lin, ds_lin = paired_delta(metrics_df, "regime_aware", "flat_linear", "fixed_0.5", "macro_f1")
    dm_gat, ds_gat = paired_delta(metrics_df, "regime_aware", "flat_gated", "fixed_0.5", "macro_f1")
    fp_delta, fp_std = paired_delta(metrics_df, "regime_aware", "flat_linear", "fixed_0.5", "fp_rate_non_toxic")
    hfn_delta, hfn_std = paired_delta(metrics_df, "regime_aware", "flat_linear", "fixed_0.5", "hate_fn_rate")
    ofn_delta, ofn_std = paired_delta(metrics_df, "regime_aware", "flat_linear", "fixed_0.5", "offensive_fn_rate")
    fp_delta_g, fp_std_g = paired_delta(metrics_df, "regime_aware", "flat_gated", "fixed_0.5", "fp_rate_non_toxic")
    hfn_delta_g, hfn_std_g = paired_delta(metrics_df, "regime_aware", "flat_gated", "fixed_0.5", "hate_fn_rate")
    ofn_delta_g, ofn_std_g = paired_delta(metrics_df, "regime_aware", "flat_gated", "fixed_0.5", "offensive_fn_rate")

    ra_m, ra_s = ms("regime_aware", "fixed_0.5", "macro_f1")
    lin_m, lin_s = ms("flat_linear", "fixed_0.5", "macro_f1")
    gat_m, gat_s = ms("flat_gated", "fixed_0.5", "macro_f1")

    lines.append("九、结论与判读")
    lines.append("-" * 68)
    lines.append(
        f"- 整体 Macro-F1：regime_aware={fmt(ra_m)}±{fmt(ra_s)}，"
        f"flat_linear={fmt(lin_m)}±{fmt(lin_s)}，flat_gated={fmt(gat_m)}±{fmt(gat_s)}。"
    )
    lines.append(
        f"  vs flat_linear：Δ={fmt_signed(dm_lin)}±{fmt(ds_lin)} → {verdict_improve(dm_lin, ds_lin)}；"
        f"vs flat_gated：Δ={fmt_signed(dm_gat)}±{fmt(ds_gat)} → {verdict_improve(dm_gat, ds_gat)}。"
    )
    lines.append(
        f"- non_toxic 误报率：vs flat_linear Δ={fmt_signed(fp_delta)}±{fmt(fp_std)} → "
        f"{verdict_lower(fp_delta, fp_std, 'FP')}；"
        f"vs flat_gated Δ={fmt_signed(fp_delta_g)}±{fmt(fp_std_g)} → "
        f"{verdict_lower(fp_delta_g, fp_std_g, 'FP')}。"
    )
    lines.append(
        f"- hate 漏报率：vs flat_linear Δ={fmt_signed(hfn_delta)}±{fmt(hfn_std)} → "
        f"{verdict_lower(hfn_delta, hfn_std, 'FN(hate)')}；"
        f"vs flat_gated Δ={fmt_signed(hfn_delta_g)}±{fmt(hfn_std_g)} → "
        f"{verdict_lower(hfn_delta_g, hfn_std_g, 'FN(hate)')}。"
    )
    lines.append(
        f"- offensive 漏报率：vs flat_linear Δ={fmt_signed(ofn_delta)}±{fmt(ofn_std)} → "
        f"{verdict_lower(ofn_delta, ofn_std, 'FN(offensive)')}；"
        f"vs flat_gated Δ={fmt_signed(ofn_delta_g)}±{fmt(ofn_std_g)} → "
        f"{verdict_lower(ofn_delta_g, ofn_std_g, 'FN(offensive)')}。"
    )
    lines.append("")
    lines.append("  诚实的操作性结论：")
    lines.append(
        "  - 本实验检验的是“两阶段 regime gate + combiner”这一具体实现，"
        "而不是直接检验 regime 结构本身是否存在。"
    )
    lines.append(
        "  - 若上面的 ΔMacroF1 为正且超过 seed 波动，同时 non_toxic FP 或 hate FN 至少一项稳定下降，"
        "则说明把 hate 与 offensive 分开建模能转化为二分类收益，支持继续沿 regime-aware 方向做。"
    )
    lines.append(
        "  - 若 ΔMacroF1 与 0 的差距小于 seed 标准差，或 MacroF1 下降，"
        "则说明在该数据/该实现下 regime 信息没有转化为整体收益；"
        "此时应优先检查 gate 的标签定义、特征与训练目标，而不是直接否定 regime 结构。"
    )
    lines.append(
        "  - gate 层的 hate/offensive 分支本身很难（既有诊断中区分二者的最佳单概念 AUC≈0.64），"
        "因此 combiner 的收益上限受 gate 质量限制；本报告同时给出 gate 自身的 FP/FN/AUC，"
        "便于判断瓶颈在 gate 还是在 combiner。"
    )
    lines.append(
        f"  本次结果判读：MacroF1 vs flat_linear={verdict_improve(dm_lin, ds_lin)}；"
        f"vs flat_gated={verdict_improve(dm_gat, ds_gat)}；"
        f"non_toxic FP vs flat_linear={verdict_lower(fp_delta, fp_std, 'FP')}；"
        f"hate FN vs flat_linear={verdict_lower(hfn_delta, hfn_std, 'FN(hate)')}。"
    )
    lines.append("")
    lines.append("十、限制")
    lines.append("-" * 68)
    lines.append("- regime 由 toxic_type 派生；hate 样本量小（test n=164），hate 指标方差大。")
    lines.append("- 所有 seed 共用同一个 train/val split，seed 间波动只反映初始化/采样，不反映 split 波动。")
    lines.append("- Stage 2 combiner 用 gate 在 train 上的 logit 训练；gate 在 train 上是 in-sample 输出，")
    lines.append("  与 val/test 的 logit 分布可能有差异，这是两阶段训练的已知风险。")
    lines.append("- val-opt 阈值只在 val 上选，再应用到 test；主表仍以 0.5 为准。")
    lines.append("- target 辅助头只在 offensive 样本上训练，不能外推到 non_toxic/hate；")
    lines.append("  target_4 因 T05 映射不成立而排除。")
    lines.append("- 敏感性分析是概念向量干预，不是因果干预；random_* 是对照，不是显著性检验。")
    lines.append("- 本脚本不使用 LLM/PLM 在线推理；概念矩阵为既有 v8 + normal 管线产物。")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] 结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
