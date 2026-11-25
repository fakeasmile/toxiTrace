#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-conditioned CBM 诊断脚本（自包含，不调用 LLM）。

目的
----
在现有 TOXICN v8（26 维）+ normal（8 维）概念矩阵上，诊断三件事：
  1. target 分层错误：target mention（target=1, toxic=0）的误报、
     target 攻击（target=1, toxic=1）的漏报，是否显著高于其他样本。
  2. 概念纠缠/坍缩：T 概念是否与 A/B/C/D/E 概念纠缠；是否接近常数；
     去掉 T 概念后模型掉多少（ΔMacro-F1/ΔAUC）。
  3. 概念向量敏感性：把 T 概念置 0/1 后预测变化，是否与随机概念干预不同；
     block ablation 轨迹是否显示 target 先移除会改变预测。

重要说明
--------
- 本脚本只读现有矩阵和原始标注，不调用任何 LLM，不生成新概念向量。
- 这里的“干预”是概念向量上的敏感性分析（concept-vector sensitivity），
  不是有 ground-truth 概念的因果干预。报告里会明确标注，避免过度解释。
- 下游分类器用 PyTorch 全连接层实现（linear / gated），不使用 sklearn LR。
- 所有随机性显式设种子；串行执行；结果写入 results/target_diagnosis/。

用法
----
python scripts/diagnose_target_conditioned_cbm.py
python scripts/diagnose_target_conditioned_cbm.py --seeds 729,1136,1196,4468,5114 --epochs 120
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
OUT_ROOT = PROJECT_ROOT / "results" / "target_diagnosis"

V8_PREFIX = "items_v8_glm-4-9b-chat"
NORMAL_PREFIX = "items_normal_glm-4-9b-chat"

META_COLS = ["id", "toxic", "content"]
TARGET_DIMS = 5
T_CONCEPTS = [f"T{i:02d}" for i in range(1, TARGET_DIMS + 1)]

# 概念区块（只包含矩阵中实际存在的列）
BLOCKS: Dict[str, List[str]] = {
    "T": T_CONCEPTS,
    "A": ["A01", "A02", "A03"],
    "B": [f"B{i:02d}" for i in range(1, 8)],
    "C": [f"C{i:02d}" for i in range(1, 5)],
    "D": ["D01", "D02"],
    "E": [f"E{i:02d}" for i in range(1, 6)],
    "N": [f"N{i:02d}" for i in range(1, 9)],
}

SEED_DEFAULT = [729, 1136, 1196, 4468, 5114]


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


# ---------------------------------------------------------------------------
# 数据加载与对齐
# ---------------------------------------------------------------------------

def load_concept_matrix(split: str) -> pd.DataFrame:
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
    for i, row in df.reset_index(drop=True).iterrows():
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
        # 按 content 唯一匹配
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

    df["target_any"] = (df[[f"target_{j}" for j in range(TARGET_DIMS)]].sum(axis=1) > 0).astype(int)
    return df


def load_split(split: str) -> Tuple[pd.DataFrame, List[str]]:
    df, concept_cols = load_concept_matrix(split)
    raw = load_raw(split)
    df = attach_metadata(df, raw, split)
    print(
        f"[info] {split}: rows={len(df)}，concepts={len(concept_cols)}，"
        f"toxic_rate={df['toxic'].mean():.4f}，target_any_rate={df['target_any'].mean():.4f}"
    )
    return df, concept_cols


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


def make_model(model_type: str, d: int) -> nn.Module:
    if model_type == "linear":
        return LinearReadout(d)
    if model_type == "gated":
        return GatedReadout(d)
    raise ValueError(f"未知模型类型：{model_type}")


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def compute_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    p, r, f1_pos, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_macro),
        "pos_f1": float(f1_pos),
        "pos_precision": float(p),
        "pos_recall": float(r),
        "auc": auc,
        "toxic_rate": float(np.mean(y)),
        "mean_prob": float(np.mean(prob)),
    }


def train_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
    patience: int = 20,
) -> Dict[str, object]:
    set_seed(seed)
    model = make_model(model_type, x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.long, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.long, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)

    ds = TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val = -1.0
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
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
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_prob = torch.softmax(model(xte), dim=1)[:, 1].cpu().numpy()
        val_prob = torch.softmax(model(xva), dim=1)[:, 1].cpu().numpy()

    return {
        "model": model,
        "test_prob": test_prob,
        "val_f1": best_val,
        "metrics": compute_metrics(y_test, test_prob),
        "val_metrics": compute_metrics(y_val, val_prob),
    }


# ---------------------------------------------------------------------------
# 分层指标
# ---------------------------------------------------------------------------

def strata_metrics(
    df: pd.DataFrame,
    prob: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    y = df["toxic"].to_numpy()
    rows = []
    for key, idx in df.groupby(group_col, dropna=False).groups.items():
        idx = np.asarray(list(idx))
        yy = y[idx]
        pp = prob[idx]
        if len(yy) == 0:
            continue
        pred = (pp >= 0.5).astype(int)
        p, r, f1_pos, _ = precision_recall_fscore_support(
            yy, pred, average="binary", pos_label=1, zero_division=0
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            yy, pred, average="macro", zero_division=0
        )
        try:
            auc = float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "group": str(key),
                "n": int(len(yy)),
                "toxic_rate": float(np.mean(yy)),
                "mean_prob": float(np.mean(pp)),
                "acc": float(accuracy_score(yy, pred)),
                "macro_f1": float(f1_macro),
                "pos_f1": float(f1_pos),
                "pos_precision": float(p),
                "pos_recall": float(r),
                "auc": auc,
                "fp_rate_non_toxic": safe_div(fp, fp + tn),
                "fn_rate_toxic": safe_div(fn, fn + tp),
                "n_non_toxic": int(tn + fp),
                "n_toxic": int(fn + tp),
            }
        )
    out = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    return out


def target_specific_strata(df: pd.DataFrame, prob: pd.Series) -> pd.DataFrame:
    """target mention / target attack / non-target toxic 等组合。"""
    rows = []
    y = df["toxic"].to_numpy()
    p = prob.to_numpy()
    for j in range(TARGET_DIMS):
        t = df[f"target_{j}"].to_numpy()
        # target mention: target=1, toxic=0；target attack: target=1, toxic=1
        for name, mask in [
            (f"T{j+1}_mention_non_toxic", (t == 1) & (y == 0)),
            (f"T{j+1}_attack_toxic", (t == 1) & (y == 1)),
            (f"T{j+1}_other", (t == 0)),
        ]:
            if mask.sum() == 0:
                continue
            yy = y[mask]
            pp = p[mask]
            pred = (pp >= 0.5).astype(int)
            p_, r_, f1_, _ = precision_recall_fscore_support(
                yy, pred, average="binary", pos_label=1, zero_division=0
            )
            tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "stratum": name,
                    "n": int(mask.sum()),
                    "toxic_rate": float(np.mean(yy)),
                    "mean_prob": float(np.mean(pp)),
                    "pos_f1": float(f1_),
                    "pos_precision": float(p_),
                    "pos_recall": float(r_),
                    "fp_rate_non_toxic": safe_div(fp, fp + tn),
                    "fn_rate_toxic": safe_div(fn, fn + tp),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 概念纠缠 / 坍缩
# ---------------------------------------------------------------------------

def effective_rank(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return float("nan")
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def concept_entanglement(
    df: pd.DataFrame,
    concept_cols: List[str],
) -> Dict[str, object]:
    x = df[concept_cols].to_numpy(dtype=float)
    y = df["toxic"].to_numpy(dtype=int)

    corr = pd.DataFrame(np.corrcoef(x, rowvar=False), index=concept_cols, columns=concept_cols)
    corr_pairs = []
    for i in range(len(concept_cols)):
        for j in range(i + 1, len(concept_cols)):
            r = corr.iloc[i, j]
            if np.isfinite(r):
                corr_pairs.append((concept_cols[i], concept_cols[j], float(r)))
    corr_pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    # 每概念对 toxicity 的 AUC
    auc_rows = []
    for c in concept_cols:
        vals = df[c].to_numpy(dtype=float)
        try:
            auc = float(roc_auc_score(y, vals)) if len(set(y)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        auc_rows.append({"concept": c, "auc_toxicity": auc})
    auc_df = pd.DataFrame(auc_rows)

    # T 概念对每个 target 维度的 AUC
    t_rows = []
    for t in T_CONCEPTS:
        if t not in df.columns:
            continue
        vals = df[t].to_numpy(dtype=float)
        for j in range(TARGET_DIMS):
            target = df[f"target_{j}"].to_numpy(dtype=int)
            try:
                auc = float(roc_auc_score(target, vals)) if len(set(target)) > 1 else float("nan")
            except ValueError:
                auc = float("nan")
            t_rows.append({"concept": t, "target_dim": f"target_{j}", "auc_target": auc})
    t_auc_df = pd.DataFrame(t_rows)

    # PCA 解释方差
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    var = s ** 2
    var = var / var.sum() if var.sum() > 0 else var
    pca_cum = np.cumsum(var[: min(10, len(var))]).tolist()

    return {
        "corr": corr,
        "corr_pairs": corr_pairs,
        "auc_df": auc_df,
        "t_auc_df": t_auc_df,
        "effective_rank": effective_rank(x),
        "pca_cum": pca_cum,
    }


def collapse_stats(df: pd.DataFrame, concept_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in concept_cols:
        v = df[c].to_numpy(dtype=float)
        hist, _ = np.histogram(v, bins=np.linspace(0, 1, 11))
        p = hist / max(hist.sum(), 1)
        ent = float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(p)))
        frac_hi = float(np.mean(v > 0.9))
        frac_lo = float(np.mean(v < 0.1))
        flag = bool(np.std(v) < 0.02 or frac_hi > 0.9 or frac_lo > 0.9)
        rows.append(
            {
                "concept": c,
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "frac_gt_0.9": frac_hi,
                "frac_lt_0.1": frac_lo,
                "entropy_norm": ent,
                "near_constant": flag,
            }
        )
    return pd.DataFrame(rows).sort_values("std").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 敏感性分析（不是 ground-truth 干预）
# ---------------------------------------------------------------------------

def predict_prob(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        return torch.softmax(model(xt), dim=1)[:, 1].cpu().numpy()


def sensitivity_analysis(
    model: nn.Module,
    df: pd.DataFrame,
    concept_cols: List[str],
    scaler: StandardScaler,
    device: torch.device,
    n_random: int = 20,
) -> Dict[str, object]:
    """概念向量敏感性：T 概念置 0/1 vs 随机非 T 概念。"""
    raw = df[concept_cols].to_numpy(dtype=float)
    base_scaled = scaler.transform(raw)
    base_prob = predict_prob(model, base_scaled, device)

    t_idx = [concept_cols.index(c) for c in T_CONCEPTS if c in concept_cols]
    non_t_idx = [i for i in range(len(concept_cols)) if i not in t_idx]
    target_any = df["target_any"].to_numpy(dtype=int)

    def _intervene(set_idxs: Sequence[int], value: float, mask: np.ndarray) -> np.ndarray:
        x = raw.copy()
        for i in set_idxs:
            x[mask, i] = value
        return predict_prob(model, scaler.transform(x), device)

    rows = []
    examples = []

    # T off on target-positive samples
    mask_pos = target_any == 1
    if mask_pos.sum() > 0:
        p_new = _intervene(t_idx, 0.0, mask_pos)
        d = p_new[mask_pos] - base_prob[mask_pos]
        flip = ((p_new[mask_pos] >= 0.5) != (base_prob[mask_pos] >= 0.5))
        rows.append(
            {
                "analysis": "T_off_target_positive",
                "n": int(mask_pos.sum()),
                "mean_delta_p": float(np.mean(d)),
                "mean_abs_delta_p": float(np.mean(np.abs(d))),
                "flip_rate": float(np.mean(flip)),
            }
        )
        # 随机非 T 对照
        rng = np.random.default_rng(0)
        rand_d, rand_abs, rand_flip = [], [], []
        for _ in range(n_random):
            idxs = rng.choice(non_t_idx, size=len(t_idx), replace=False).tolist()
            p_rand = _intervene(idxs, 0.0, mask_pos)
            dd = p_rand[mask_pos] - base_prob[mask_pos]
            rand_d.append(np.mean(dd))
            rand_abs.append(np.mean(np.abs(dd)))
            rand_flip.append(np.mean((p_rand[mask_pos] >= 0.5) != (base_prob[mask_pos] >= 0.5)))
        rows.append(
            {
                "analysis": "random_off_target_positive",
                "n": int(mask_pos.sum()),
                "mean_delta_p": float(np.mean(rand_d)),
                "mean_abs_delta_p": float(np.mean(rand_abs)),
                "flip_rate": float(np.mean(rand_flip)),
            }
        )
        # 例子
        idxs = np.where(mask_pos)[0]
        top = idxs[np.argsort(-np.abs(d))[:10]]
        for i in top:
            examples.append(
                {
                    "analysis": "T_off_target_positive",
                    "id": df.iloc[i]["id"],
                    "content": str(df.iloc[i]["content"])[:160],
                    "toxic": int(df.iloc[i]["toxic"]),
                    "target_any": int(target_any[i]),
                    "base_prob": float(base_prob[i]),
                    "new_prob": float(p_new[i]),
                    "delta_p": float(d[list(idxs).index(i)]),
                }
            )

    # T on on target-negative samples
    mask_neg = target_any == 0
    if mask_neg.sum() > 0:
        p_new = _intervene(t_idx, 1.0, mask_neg)
        d = p_new[mask_neg] - base_prob[mask_neg]
        flip = ((p_new[mask_neg] >= 0.5) != (base_prob[mask_neg] >= 0.5))
        rows.append(
            {
                "analysis": "T_on_target_negative",
                "n": int(mask_neg.sum()),
                "mean_delta_p": float(np.mean(d)),
                "mean_abs_delta_p": float(np.mean(np.abs(d))),
                "flip_rate": float(np.mean(flip)),
            }
        )
        rng = np.random.default_rng(1)
        rand_d, rand_abs, rand_flip = [], [], []
        for _ in range(n_random):
            idxs = rng.choice(non_t_idx, size=len(t_idx), replace=False).tolist()
            p_rand = _intervene(idxs, 1.0, mask_neg)
            dd = p_rand[mask_neg] - base_prob[mask_neg]
            rand_d.append(np.mean(dd))
            rand_abs.append(np.mean(np.abs(dd)))
            rand_flip.append(np.mean((p_rand[mask_neg] >= 0.5) != (base_prob[mask_neg] >= 0.5)))
        rows.append(
            {
                "analysis": "random_on_target_negative",
                "n": int(mask_neg.sum()),
                "mean_delta_p": float(np.mean(rand_d)),
                "mean_abs_delta_p": float(np.mean(rand_abs)),
                "flip_rate": float(np.mean(rand_flip)),
            }
        )

    # block ablation 轨迹：对 target-positive 有毒样本，按 T→A→B→C→D→E→N 置 0
    mask = (target_any == 1) & (df["toxic"].to_numpy(dtype=int) == 1)
    traj_rows = []
    if mask.sum() > 0:
        x = raw.copy()
        current_prob = base_prob.copy()
        traj_rows.append({"step": 0, "block": "none", "n": int(mask.sum()), "mean_prob": float(np.mean(current_prob[mask]))})
        for step, block in enumerate(["T", "A", "B", "C", "D", "E", "N"], start=1):
            cols = [c for c in BLOCKS.get(block, []) if c in concept_cols]
            idxs = [concept_cols.index(c) for c in cols]
            for i in idxs:
                x[mask, i] = 0.0
            current_prob = predict_prob(model, scaler.transform(x), device)
            traj_rows.append(
                {
                    "step": step,
                    "block": block,
                    "n": int(mask.sum()),
                    "mean_prob": float(np.mean(current_prob[mask])),
                }
            )

    return {
        "summary": pd.DataFrame(rows),
        "examples": pd.DataFrame(examples),
        "block_trajectory": pd.DataFrame(traj_rows),
        "base_prob": base_prob,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, seed: int, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(df["toxic"].to_numpy() == 0)[0]
    idx1 = np.where(df["toxic"].to_numpy() == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n_val0 = int(round(len(idx0) * val_ratio))
    n_val1 = int(round(len(idx1) * val_ratio))
    val_idx = np.concatenate([idx0[:n_val0], idx1[:n_val1]])
    tr_idx = np.concatenate([idx0[n_val0:], idx1[n_val1:]])
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    return tr_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_DEFAULT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--skip_gated", action="store_true", help="只跑 linear，加快诊断")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}，seeds={seeds}，epochs={args.epochs}")

    train_df, concept_cols = load_split("train")
    test_df, test_concepts = load_split("test")
    if concept_cols != test_concepts:
        raise ValueError("train/test 概念列顺序不一致，请先对齐。")

    # 概念列必须覆盖所有 block 的列
    missing_block_cols = [c for c in sum(BLOCKS.values(), []) if c not in concept_cols]
    if missing_block_cols:
        print(f"[warn] 以下 block 列不在矩阵中，将被跳过：{missing_block_cols}")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录：{out_dir}")

    # 保存数据概览
    overview = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_concepts": int(len(concept_cols)),
        "concepts": concept_cols,
        "train_toxic_rate": float(train_df["toxic"].mean()),
        "test_toxic_rate": float(test_df["toxic"].mean()),
        "train_target_any_rate": float(train_df["target_any"].mean()),
        "test_target_any_rate": float(test_df["target_any"].mean()),
        "train_expression_counts": {str(k): int(v) for k, v in train_df["expression"].value_counts().sort_index().items()},
        "test_expression_counts": {str(k): int(v) for k, v in test_df["expression"].value_counts().sort_index().items()},
        "train_toxic_type_counts": {str(k): int(v) for k, v in train_df["toxic_type"].value_counts().sort_index().items()},
        "test_toxic_type_counts": {str(k): int(v) for k, v in test_df["toxic_type"].value_counts().sort_index().items()},
        "target_dim_positive_counts_train": {
            f"target_{j}": int(train_df[f"target_{j}"].sum()) for j in range(TARGET_DIMS)
        },
        "target_dim_positive_counts_test": {
            f"target_{j}": int(test_df[f"target_{j}"].sum()) for j in range(TARGET_DIMS)
        },
    }
    with (out_dir / "overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    # 训练/验证划分只依赖训练集
    tr_idx, val_idx = stratified_split(train_df, seed=seeds[0])
    x_all = train_df[concept_cols].to_numpy(dtype=float)
    y_all = train_df["toxic"].to_numpy(dtype=int)
    x_test = test_df[concept_cols].to_numpy(dtype=float)
    y_test = test_df["toxic"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(x_all[tr_idx])
    x_tr = scaler.transform(x_all[tr_idx])
    x_val = scaler.transform(x_all[val_idx])
    x_te = scaler.transform(x_test)

    # 先做概念纠缠/坍缩（与模型无关）
    ent = concept_entanglement(train_df, concept_cols)
    ent["corr"].to_csv(out_dir / "concept_corr.csv", encoding="utf-8-sig")
    pd.DataFrame(ent["corr_pairs"], columns=["concept_a", "concept_b", "pearson_r"]).to_csv(
        out_dir / "concept_corr_pairs.csv", index=False, encoding="utf-8-sig"
    )
    ent["auc_df"].to_csv(out_dir / "concept_auc_toxicity.csv", index=False, encoding="utf-8-sig")
    ent["t_auc_df"].to_csv(out_dir / "concept_auc_target.csv", index=False, encoding="utf-8-sig")
    collapse = collapse_stats(train_df, concept_cols)
    collapse.to_csv(out_dir / "concept_collapse.csv", index=False, encoding="utf-8-sig")

    # 模型列表：linear/gated × 全概念/去 T
    model_types = ["linear"] if args.skip_gated else ["linear", "gated"]
    feature_sets = {"full": concept_cols, "no_T": [c for c in concept_cols if c not in T_CONCEPTS]}

    all_rows = []
    seed_model_cache = {}
    runs = [(mt, fs, seed) for mt in model_types for fs in feature_sets for seed in seeds]
    pbar = tqdm(runs, desc="runs", leave=False)
    for model_type, fs_name, seed in pbar:
        cols = feature_sets[fs_name]
        idx_cols = [concept_cols.index(c) for c in cols]
        xtr = x_tr[:, idx_cols]
        xva = x_val[:, idx_cols]
        xte = x_te[:, idx_cols]
        out = train_readout(
            xtr, y_all[tr_idx], xva, y_all[val_idx], xte, y_test,
            model_type=model_type, seed=seed, epochs=args.epochs,
            device=device, batch_size=args.batch_size,
        )
        m = out["metrics"]
        row = {
            "model": model_type,
            "features": fs_name,
            "seed": seed,
            "val_f1": float(out["val_f1"]),
            **{f"test_{k}": v for k, v in m.items()},
        }
        all_rows.append(row)
        pbar.set_postfix({"macro_f1": f"{m['macro_f1']:.4f}", "auc": f"{m['auc']:.4f}"})
        if fs_name == "full" and seed == seeds[0]:
            seed_model_cache[(model_type, fs_name)] = {
                "model": out["model"],
                "cols": cols,
                "test_prob": out["test_prob"],
            }

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")

    # 聚合
    agg = (
        metrics_df.groupby(["model", "features"])[
            ["test_acc", "test_macro_f1", "test_pos_f1", "test_auc", "test_pos_precision", "test_pos_recall"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    # 用第一个 seed 的 full 模型做分层/敏感性
    strata_out: Dict[str, pd.DataFrame] = {}
    if ("linear", "full") in seed_model_cache:
        cache = seed_model_cache[("linear", "full")]
        prob = cache["test_prob"]
        test_df = test_df.copy()
        test_df["pred_prob"] = prob
        test_df["pred"] = (prob >= 0.5).astype(int)

        strata_out["target"] = strata_metrics(test_df, prob, "target_any")
        for j in range(TARGET_DIMS):
            strata_out[f"target_{j}"] = strata_metrics(test_df, prob, f"target_{j}")
        strata_out["expression"] = strata_metrics(test_df, prob, "expression")
        strata_out["toxic_type"] = strata_metrics(test_df, prob, "toxic_type")
        strata_out["target_specific"] = target_specific_strata(test_df, pd.Series(prob))

        for name, sdf in strata_out.items():
            sdf.to_csv(out_dir / f"strata_{name}.csv", index=False, encoding="utf-8-sig")

        sens = sensitivity_analysis(
            cache["model"], test_df, concept_cols, scaler, device, n_random=20
        )
        sens["summary"].to_csv(out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
        sens["examples"].to_csv(out_dir / "sensitivity_examples.csv", index=False, encoding="utf-8-sig")
        sens["block_trajectory"].to_csv(out_dir / "block_ablation_trajectory.csv", index=False, encoding="utf-8-sig")
        test_df[["id", "content", "toxic", "target_any", "expression", "toxic_type", "pred_prob", "pred"]].to_csv(
            out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
        )

    # 报告
    lines = []
    lines.append("Target-conditioned CBM 诊断报告")
    lines.append("=" * 60)
    lines.append(f"时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"device：{device}")
    lines.append(f"seeds：{seeds}")
    lines.append("")
    lines.append("数据概览")
    lines.append("-" * 60)
    lines.append(f"train rows={overview['train_rows']}，test rows={overview['test_rows']}，concepts={overview['n_concepts']}")
    lines.append(f"train toxic rate={overview['train_toxic_rate']:.4f}，test toxic rate={overview['test_toxic_rate']:.4f}")
    lines.append(f"train target_any rate={overview['train_target_any_rate']:.4f}，test target_any rate={overview['test_target_any_rate']:.4f}")
    lines.append(f"test target positive counts：{overview['target_dim_positive_counts_test']}")
    lines.append(f"test expression counts：{overview['test_expression_counts']}")
    lines.append(f"test toxic_type counts：{overview['test_toxic_type_counts']}")
    lines.append("")
    lines.append("整体指标（mean ± std，test）")
    lines.append("-" * 60)
    for _, r in agg.iterrows():
        lines.append(
            f"{r['model']:<7} {r['features']:<5} "
            f"Acc={r['test_acc_mean']:.4f}±{r['test_acc_std']:.4f} "
            f"MacroF1={r['test_macro_f1_mean']:.4f}±{r['test_macro_f1_std']:.4f} "
            f"PosF1={r['test_pos_f1_mean']:.4f}±{r['test_pos_f1_std']:.4f} "
            f"AUC={r['test_auc_mean']:.4f}±{r['test_auc_std']:.4f}"
        )
    lines.append("")
    lines.append("T 概念消融（no_T - full，负值表示去掉 T 后变差）")
    lines.append("-" * 60)
    for mt in model_types:
        full_rows = agg[(agg["model"] == mt) & (agg["features"] == "full")]
        no_t_rows = agg[(agg["model"] == mt) & (agg["features"] == "no_T")]
        if len(full_rows) and len(no_t_rows):
            f = full_rows.iloc[0]
            n = no_t_rows.iloc[0]
            lines.append(
                f"{mt:<7} ΔMacroF1={n['test_macro_f1_mean'] - f['test_macro_f1_mean']:+.4f} "
                f"ΔAUC={n['test_auc_mean'] - f['test_auc_mean']:+.4f} "
                f"ΔPosF1={n['test_pos_f1_mean'] - f['test_pos_f1_mean']:+.4f}"
            )
    lines.append("")
    lines.append("概念纠缠/坍缩（train）")
    lines.append("-" * 60)
    lines.append(f"effective rank={ent['effective_rank']:.2f} / {len(concept_cols)}")
    lines.append(f"PCA cumulative variance (first 5)={['%.3f' % v for v in ent['pca_cum'][:5]]}")
    lines.append("Top 10 correlated concept pairs：")
    for a, b, r in ent["corr_pairs"][:10]:
        lines.append(f"  {a} ~ {b}: r={r:.3f}")
    flagged = collapse[collapse["near_constant"]]
    lines.append(f"近常数概念数：{len(flagged)}")
    for _, r in flagged.head(20).iterrows():
        lines.append(
            f"  {r['concept']}: mean={r['mean']:.3f} std={r['std']:.4f} "
            f"frac>0.9={r['frac_gt_0.9']:.2f} frac<0.1={r['frac_lt_0.1']:.2f}"
        )
    lines.append("")
    if "target" in strata_out:
        lines.append("Target 分层（seed=第一个 seed，linear/full）")
        lines.append("-" * 60)
        s = strata_out["target"]
        for _, r in s.iterrows():
            lines.append(
                f"{r['group']:<14} n={int(r['n']):<5} toxic_rate={r['toxic_rate']:.3f} "
                f"MacroF1={r['macro_f1']:.4f} PosF1={r['pos_f1']:.4f} "
                f"FP_rate={r['fp_rate_non_toxic']:.4f} FN_rate={r['fn_rate_toxic']:.4f}"
            )
        lines.append("")
        lines.append("Target-specific mention/attack（seed=第一个 seed）")
        lines.append("-" * 60)
        ts = strata_out["target_specific"]
        for _, r in ts.iterrows():
            lines.append(
                f"{r['stratum']:<28} n={int(r['n']):<5} toxic_rate={r['toxic_rate']:.3f} "
                f"mean_prob={r['mean_prob']:.4f} PosF1={r['pos_f1']:.4f} "
                f"FP_rate={r['fp_rate_non_toxic']:.4f} FN_rate={r['fn_rate_toxic']:.4f}"
            )
        lines.append("")
        lines.append("Expression / toxic_type 分层（seed=第一个 seed）")
        lines.append("-" * 60)
        for name in ("expression", "toxic_type"):
            if name in strata_out:
                lines.append(f"[{name}]")
                for _, r in strata_out[name].iterrows():
                    lines.append(
                        f"  {r['group']:<4} n={int(r['n']):<5} toxic_rate={r['toxic_rate']:.3f} "
                        f"MacroF1={r['macro_f1']:.4f} PosF1={r['pos_f1']:.4f} "
                        f"FP_rate={r['fp_rate_non_toxic']:.4f} FN_rate={r['fn_rate_toxic']:.4f}"
                    )
        lines.append("")
        lines.append("概念向量敏感性（seed=第一个 seed；非 ground-truth 干预）")
        lines.append("-" * 60)
        if "sensitivity_summary" in [p.stem for p in out_dir.glob("*.csv")]:
            sens_df = pd.read_csv(out_dir / "sensitivity_summary.csv")
            for _, r in sens_df.iterrows():
                lines.append(
                    f"{r['analysis']:<30} n={int(r['n']):<5} "
                    f"mean_dp={r['mean_delta_p']:+.4f} |dp|={r['mean_abs_delta_p']:.4f} "
                    f"flip_rate={r['flip_rate']:.4f}"
                )
        if (out_dir / "block_ablation_trajectory.csv").is_file():
            lines.append("")
            lines.append("Block ablation trajectory（target-positive toxic；按 T→A→B→C→D→E→N 置 0）")
            traj = pd.read_csv(out_dir / "block_ablation_trajectory.csv")
            for _, r in traj.iterrows():
                lines.append(f"  step={int(r['step'])} block={r['block']:<5} mean_prob={r['mean_prob']:.4f}")

    lines.append("")
    lines.append("说明")
    lines.append("-" * 60)
    lines.append("- 这里的 T_off/T_on 是把全部 T01–T05 概念置 0/1 的概念向量敏感性分析，")
    lines.append("  不是有 ground-truth 概念的因果干预，也不是单目标概念的精确干预。")
    lines.append("- target 维度顺序未做语义假设；target_0..target_4 按原始 one-hot 顺序编号。")
    lines.append("- 分层指标来自第一个 seed 的 linear/full 模型；整体指标是跨 seed 的 mean±std。")
    lines.append("- 如果 T 概念与 A/B/C/D/E 高度相关、T 消融掉点明显、T_off/T_on 与随机干预差异小，")
    lines.append("  说明当前扁平 CBM 的 target 通路弱或纠缠；反之则说明 target-conditioned 假设有实证起点。")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] 结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
