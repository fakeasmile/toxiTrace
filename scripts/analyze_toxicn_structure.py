#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToxiCN 标注结构分析（只读，不训练模型，不调用 LLM）。

目的
----
在写 soft conditional CBM 之前，把 ToxiCN 的标注层级、条件分布、边界噪声、
target 维度映射和条件分层指标固定下来。

分析内容
--------
1. 标注层级：toxic -> toxic_type / expression -> target（target 只在有毒样本上为正）。
2. 条件分布：P(expression|toxic)、P(toxic_type|toxic)、P(target|toxic)、target 共现。
3. 边界噪声：expression=0 但 toxic=1、toxic_type=0 但 toxic=1、target_any=1 但 toxic=0、
   target_any=0 但 toxic=1、expression>=1 但 toxic=0；输出数量与例子。
4. 条件指标：读取现有 test_predictions.csv，按 gate/type/target 分支输出指标；
   另做 oracle 阈值扫描（仅诊断，不做模型选择）。
5. target 映射：T 概念对 raw target 维度的 AUC，输出最佳匹配与例子。
6. 概念条件统计：每个概念在 toxic/non-toxic、expression=0 有毒 vs 无毒、
   hate vs offensive、各 target 分支上的均值差与 AUC。

用法
----
python scripts/analyze_toxicn_structure.py
python scripts/analyze_toxicn_structure.py --pred_csv results/target_diagnosis/20261124-211452/test_predictions.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "TOXICN"
CONCEPT_DIR = PROJECT_ROOT / "data" / "concepts"
V8_DIR = PROJECT_ROOT / "data" / "processed" / "TOXICN" / "v8_glm-4-9b-chat"
NORMAL_DIR = PROJECT_ROOT / "data" / "processed" / "TOXICN" / "normal_glm-4-9b-chat"
OUT_ROOT = PROJECT_ROOT / "results" / "annotation_structure"
DIAG_ROOT = PROJECT_ROOT / "results" / "target_diagnosis"

V8_PREFIX = "items_v8_glm-4-9b-chat"
NORMAL_PREFIX = "items_normal_glm-4-9b-chat"

TARGET_DIMS = 5
# 由 target 例子 + T 概念 AUC 推断；报告里明确标注“推断，需按数据集文档确认”。
TARGET_SEMANTICS_INFERRED = {
    0: "LGBTQ",
    1: "region",
    2: "gender",
    3: "race",
    4: "other",
}
T_CONCEPT_SEMANTICS = {
    "T01": "gender",
    "T02": "race",
    "T03": "region",
    "T04": "LGBTQ",
    "T05": "other",
}
EXPECTED_T_TO_TARGET = {
    "T01": 2,
    "T02": 3,
    "T03": 1,
    "T04": 0,
    "T05": 4,
}

META_COLS = ["id", "toxic", "content"]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def norm_text(s: object) -> str:
    return re.sub(r"\s+", "", str(s))


def parse_target(value: object) -> List[int]:
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


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def fmt(x: object, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if math.isnan(v) or math.isinf(v):
        return "nan"
    return f"{v:.{nd}f}"


def truncate(s: object, n: int = 140) -> str:
    text = str(s).replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


# ---------------------------------------------------------------------------
# 原始数据
# ---------------------------------------------------------------------------

def load_raw(split: str) -> pd.DataFrame:
    path = RAW_DIR / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到原始数据：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for i, r in enumerate(data):
        targets = parse_target(r.get("target"))
        rows.append(
            {
                "id": f"TOXICN_{split}_{i:04d}",
                "content": r.get("content", ""),
                "toxic": int(str(r.get("toxic", 0))),
                "expression": int(str(r.get("expression", 0))),
                "toxic_type": int(str(r.get("toxic_type", 0))),
                **{f"target_{j}": v for j, v in enumerate(targets)},
                "target_any": int(any(targets)),
                "target_count": int(sum(targets)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 标注结构表
# ---------------------------------------------------------------------------

def value_counts_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts().sort_index()
    out = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
    out["rate"] = out["count"] / len(df)
    return out


def crosstab_table(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    ct = pd.crosstab(df[a], df[b], dropna=False)
    return ct.reset_index()


def target_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"target_{j}" for j in range(TARGET_DIMS)]
    mat = np.zeros((TARGET_DIMS, TARGET_DIMS), dtype=int)
    arr = df[cols].to_numpy(dtype=int)
    for i in range(TARGET_DIMS):
        for j in range(TARGET_DIMS):
            mat[i, j] = int(np.sum((arr[:, i] == 1) & (arr[:, j] == 1)))
    labels = [f"target_{j}({TARGET_SEMANTICS_INFERRED[j]})" for j in range(TARGET_DIMS)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def target_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["target_count"].value_counts().sort_index()
    out = pd.DataFrame({"target_count": vc.index.astype(int), "count": vc.values})
    out["rate"] = out["count"] / len(df)
    return out


def boundary_cases(df: pd.DataFrame) -> List[Dict[str, object]]:
    cases = [
        ("expression=0 & toxic=1", (df["expression"] == 0) & (df["toxic"] == 1)),
        ("expression>=1 & toxic=0", (df["expression"] >= 1) & (df["toxic"] == 0)),
        ("toxic_type=0 & toxic=1", (df["toxic_type"] == 0) & (df["toxic"] == 1)),
        ("toxic_type>=1 & toxic=0", (df["toxic_type"] >= 1) & (df["toxic"] == 0)),
        ("target_any=1 & toxic=0", (df["target_any"] == 1) & (df["toxic"] == 0)),
        ("target_any=0 & toxic=1", (df["target_any"] == 0) & (df["toxic"] == 1)),
        ("expression=0 & toxic_type>=1", (df["expression"] == 0) & (df["toxic_type"] >= 1)),
        ("expression>=1 & toxic_type=0", (df["expression"] >= 1) & (df["toxic_type"] == 0)),
    ]
    out = []
    for name, mask in cases:
        sub = df.loc[mask]
        out.append(
            {
                "case": name,
                "n": int(len(sub)),
                "rate": float(len(sub) / len(df)),
                "ids": sub["id"].head(10).tolist(),
                "examples": sub["content"].head(3).map(truncate).tolist(),
            }
        )
    return out


def target_examples(df: pd.DataFrame, n: int = 4) -> List[Dict[str, object]]:
    out = []
    for j in range(TARGET_DIMS):
        sub = df.loc[(df[f"target_{j}"] == 1) & (df["toxic"] == 1)]
        out.append(
            {
                "target_dim": f"target_{j}",
                "inferred_semantics": TARGET_SEMANTICS_INFERRED[j],
                "n_toxic": int(len(sub)),
                "examples": sub["content"].head(n).map(truncate).tolist(),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 预测结果与条件指标
# ---------------------------------------------------------------------------

def find_default_pred_csv() -> Optional[Path]:
    if not DIAG_ROOT.is_dir():
        return None
    candidates = sorted(DIAG_ROOT.glob("*/test_predictions.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def id_suffix(sample_id: object) -> Optional[int]:
    m = re.search(r"_(\d+)$", str(sample_id))
    return int(m.group(1)) if m else None


def attach_target_dims(pred: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """pred 里没有 target 五维；按 id 后缀对齐 raw。"""
    pred = pred.copy()
    pred["_suffix"] = pred["id"].map(id_suffix)
    raw = raw.reset_index(drop=True)
    meta_rows = []
    for _, row in pred.iterrows():
        idx = row["_suffix"]
        if pd.isna(idx) or int(idx) < 0 or int(idx) >= len(raw):
            meta_rows.append({f"target_{j}": np.nan for j in range(TARGET_DIMS)})
            continue
        r = raw.iloc[int(idx)]
        if norm_text(row.get("content", "")) != norm_text(r.get("content", "")):
            meta_rows.append({f"target_{j}": np.nan for j in range(TARGET_DIMS)})
            continue
        meta_rows.append({f"target_{j}": int(r[f"target_{j}"]) for j in range(TARGET_DIMS)})
    meta = pd.DataFrame(meta_rows, index=pred.index)
    pred = pd.concat([pred, meta], axis=1)
    if pred[[f"target_{j}" for j in range(TARGET_DIMS)]].isna().any().any():
        raise ValueError("预测文件与原始 test 的 id/content 无法对齐，请检查 --pred_csv。")
    pred["target_any_raw"] = (
        pred[[f"target_{j}" for j in range(TARGET_DIMS)]].sum(axis=1) > 0
    ).astype(int)
    return pred


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    p, r, f1_pos, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "toxic_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_macro),
        "pos_f1": float(f1_pos),
        "pos_precision": float(p),
        "pos_recall": float(r),
        "auc": auc,
        "brier": float(brier_score_loss(y, prob)) if len(set(y)) > 1 else float("nan"),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tp": int(tp),
        "fp_rate_non_toxic": safe_div(fp, fp + tn),
        "fn_rate_toxic": safe_div(fn, fn + tp),
        "mean_prob": float(np.mean(prob)),
    }


def conditional_metrics(pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    y = pred["toxic"].to_numpy(dtype=int)
    p = pred["pred_prob"].to_numpy(dtype=float)
    rows = []

    def add(name: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            return
        m = binary_metrics(y[mask], p[mask], threshold)
        rows.append({"stratum": name, **m})

    add("all", np.ones(len(pred), dtype=bool))
    for v in sorted(pred["expression"].unique()):
        add(f"expression={v}", (pred["expression"] == v).to_numpy())
    for v in sorted(pred["toxic_type"].unique()):
        add(f"toxic_type={v}", (pred["toxic_type"] == v).to_numpy())
    add("target_any=0", (pred["target_any_raw"] == 0).to_numpy())
    add("target_any=1", (pred["target_any_raw"] == 1).to_numpy())
    add(
        "expression=0 & toxic=1",
        ((pred["expression"] == 0) & (pred["toxic"] == 1)).to_numpy(),
    )
    add(
        "expression=0 & toxic=0",
        ((pred["expression"] == 0) & (pred["toxic"] == 0)).to_numpy(),
    )
    add(
        "target_any=0 & toxic=1",
        ((pred["target_any_raw"] == 0) & (pred["toxic"] == 1)).to_numpy(),
    )
    add(
        "target_any=1 & toxic=1",
        ((pred["target_any_raw"] == 1) & (pred["toxic"] == 1)).to_numpy(),
    )
    for j in range(TARGET_DIMS):
        add(
            f"target_{j}({TARGET_SEMANTICS_INFERRED[j]})=1 & toxic=1",
            ((pred[f"target_{j}"] == 1) & (pred["toxic"] == 1)).to_numpy(),
        )
    return pd.DataFrame(rows)


def oracle_threshold_sweep(pred: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """仅诊断：在 test 上扫阈值，看 AUC-F1 gap 有多少来自阈值。
    这不是模型选择；正式实验必须在验证集上选阈值。
    """
    y = pred["toxic"].to_numpy(dtype=int)
    p = pred["pred_prob"].to_numpy(dtype=float)
    rows = []
    for t in np.round(np.arange(0.20, 0.801, 0.01), 2):
        m = binary_metrics(y, p, float(t))
        rows.append({"threshold": float(t), **m})
    df = pd.DataFrame(rows)
    best = df.loc[df["macro_f1"].idxmax()].to_dict()
    return df, best


def error_breakdown(pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    y = pred["toxic"].to_numpy(dtype=int)
    p = pred["pred_prob"].to_numpy(dtype=float)
    pred_label = (p >= threshold).astype(int)
    pred = pred.copy()
    pred["_pred"] = pred_label
    rows = []
    for col in ("expression", "toxic_type", "target_any_raw"):
        for v, sub in pred.groupby(col):
            fp = int(((sub["_pred"] == 1) & (sub["toxic"] == 0)).sum())
            fn = int(((sub["_pred"] == 0) & (sub["toxic"] == 1)).sum())
            rows.append(
                {
                    "group_col": col,
                    "group": str(v),
                    "n": int(len(sub)),
                    "fp": fp,
                    "fn": fn,
                    "fp_rate_non_toxic": safe_div(fp, int((sub["toxic"] == 0).sum())),
                    "fn_rate_toxic": safe_div(fn, int((sub["toxic"] == 1).sum())),
                }
            )
    return pd.DataFrame(rows)


def uncertainty_examples(pred: pd.DataFrame, n: int = 20, band: float = 0.1) -> pd.DataFrame:
    p = pred["pred_prob"].to_numpy(dtype=float)
    mask = np.abs(p - 0.5) <= band
    sub = pred.loc[mask].copy()
    sub["uncertainty"] = np.abs(sub["pred_prob"] - 0.5)
    sub = sub.sort_values("uncertainty").head(n)
    return sub[["id", "content", "toxic", "expression", "toxic_type", "pred_prob", "uncertainty"]]


# ---------------------------------------------------------------------------
# 概念矩阵与条件统计
# ---------------------------------------------------------------------------

def load_concepts(split: str) -> pd.DataFrame:
    v8_path = V8_DIR / f"{V8_PREFIX}_{split}_matrix.csv"
    normal_path = NORMAL_DIR / f"{NORMAL_PREFIX}_{split}_matrix.csv"
    for p in (v8_path, normal_path):
        if not p.is_file():
            raise FileNotFoundError(f"找不到概念矩阵：{p}")
    v8 = pd.read_csv(v8_path)
    normal = pd.read_csv(normal_path)
    for df, name in ((v8, "v8"), (normal, "normal")):
        for c in META_COLS:
            if c not in df.columns:
                raise ValueError(f"{name} 缺少列 {c}")
    v8["id"] = v8["id"].astype(str)
    v8["toxic"] = v8["toxic"].astype(int)
    normal["id"] = normal["id"].astype(str)
    normal["toxic"] = normal["toxic"].astype(int)
    v8_concepts = [c for c in v8.columns if c not in META_COLS]
    normal_concepts = [c for c in normal.columns if c not in META_COLS]
    merged = v8.merge(normal[["id"] + normal_concepts], on="id", how="inner", validate="one_to_one")
    merged["_suffix"] = merged["id"].map(id_suffix)
    return merged, v8_concepts + normal_concepts


def attach_raw_meta(concept_df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    df = concept_df.copy()
    raw = raw.reset_index(drop=True)
    meta_rows = []
    for _, row in df.iterrows():
        idx = row["_suffix"]
        if pd.isna(idx) or int(idx) < 0 or int(idx) >= len(raw):
            meta_rows.append({})
            continue
        r = raw.iloc[int(idx)]
        if norm_text(row["content"]) != norm_text(r["content"]):
            raise ValueError(f"概念矩阵 {row['id']} 与原始 {r['id']} 内容不一致")
        meta_rows.append(
            {
                "toxic_raw": int(r["toxic"]),
                "expression": int(r["expression"]),
                "toxic_type": int(r["toxic_type"]),
                **{f"target_{j}": int(r[f"target_{j}"]) for j in range(TARGET_DIMS)},
                "target_any": int(r["target_any"]),
            }
        )
    meta = pd.DataFrame(meta_rows, index=df.index)
    df = pd.concat([df, meta], axis=1)
    if (df["toxic"].astype(int) != df["toxic_raw"].astype(int)).any():
        raise ValueError("概念矩阵与原始 toxic 标签不一致")
    return df


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, x)) if len(set(y)) > 1 else float("nan")
    except ValueError:
        return float("nan")


def concept_conditional_stats(df: pd.DataFrame, concept_cols: List[str]) -> pd.DataFrame:
    rows = []
    y_toxic = df["toxic"].to_numpy(dtype=int)

    def add(concept: str, stratum: str, mask_pos: np.ndarray, mask_neg: np.ndarray) -> None:
        if mask_pos.sum() < 5 or mask_neg.sum() < 5:
            return
        v = df[concept].to_numpy(dtype=float)
        pos = v[mask_pos]
        neg = v[mask_neg]
        y = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
        x = np.concatenate([pos, neg])
        rows.append(
            {
                "concept": concept,
                "stratum": stratum,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "mean_pos": float(np.mean(pos)),
                "mean_neg": float(np.mean(neg)),
                "mean_diff": float(np.mean(pos) - np.mean(neg)),
                "auc": _auc(y, x),
            }
        )

    m_toxic = y_toxic == 1
    m_non = y_toxic == 0
    for c in concept_cols:
        add(c, "toxic_vs_non", m_toxic, m_non)

    m_exp0 = (df["expression"] == 0).to_numpy()
    add_mask_pos = m_exp0 & m_toxic
    add_mask_neg = m_exp0 & m_non
    for c in concept_cols:
        add(c, "expression0_toxic_vs_non", add_mask_pos, add_mask_neg)

    m_hate = ((df["toxic_type"] == 1) & (df["toxic"] == 1)).to_numpy()
    m_off = ((df["toxic_type"] == 2) & (df["toxic"] == 1)).to_numpy()
    for c in concept_cols:
        add(c, "hate_vs_offensive", m_hate, m_off)

    for j in range(TARGET_DIMS):
        m_pos = ((df[f"target_{j}"] == 1) & (df["toxic"] == 1)).to_numpy()
        m_neg = ((df[f"target_{j}"] == 0) & (df["toxic"] == 1)).to_numpy()
        for c in concept_cols:
            add(c, f"target_{j}({TARGET_SEMANTICS_INFERRED[j]})_vs_other_toxic", m_pos, m_neg)

    return pd.DataFrame(rows)


def target_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """T 概念对 raw target 维度的 AUC（只在有毒样本上）。"""
    rows = []
    m_toxic = df["toxic"] == 1
    for t in ["T01", "T02", "T03", "T04", "T05"]:
        if t not in df.columns:
            continue
        vals = df.loc[m_toxic, t].to_numpy(dtype=float)
        for j in range(TARGET_DIMS):
            y = df.loc[m_toxic, f"target_{j}"].to_numpy(dtype=int)
            rows.append(
                {
                    "concept": t,
                    "concept_semantics": T_CONCEPT_SEMANTICS[t],
                    "target_dim": f"target_{j}",
                    "target_semantics_inferred": TARGET_SEMANTICS_INFERRED[j],
                    "n_pos": int(y.sum()),
                    "auc": _auc(y, vals),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["is_best_for_concept"] = False
        for t, sub in out.groupby("concept"):
            idx = sub["auc"].idxmax()
            out.loc[idx, "is_best_for_concept"] = True
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录：{out_dir}")

    train = load_raw("train")
    test = load_raw("test")
    print(f"[info] train rows={len(train)}，test rows={len(test)}")

    # 1) 标注结构
    tables = {
        "train_toxic_dist": value_counts_table(train, "toxic"),
        "test_toxic_dist": value_counts_table(test, "toxic"),
        "train_expression_dist": value_counts_table(train, "expression"),
        "test_expression_dist": value_counts_table(test, "expression"),
        "train_toxic_type_dist": value_counts_table(train, "toxic_type"),
        "test_toxic_type_dist": value_counts_table(test, "toxic_type"),
        "train_target_any_dist": value_counts_table(train, "target_any"),
        "test_target_any_dist": value_counts_table(test, "target_any"),
        "test_toxic_x_expression": crosstab_table(test, "toxic", "expression"),
        "test_toxic_x_toxic_type": crosstab_table(test, "toxic", "toxic_type"),
        "test_expression_x_toxic_type": crosstab_table(test, "expression", "toxic_type"),
        "test_target_any_x_toxic": crosstab_table(test, "target_any", "toxic"),
        "test_target_count_dist": target_count_distribution(test),
        "test_target_cooccurrence": target_cooccurrence(test),
    }
    for name, df in tables.items():
        if name == "test_target_cooccurrence":
            df.to_csv(out_dir / f"{name}.csv", index=True, encoding="utf-8-sig")
        else:
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    boundaries = boundary_cases(test)
    pd.DataFrame(boundaries).to_csv(out_dir / "boundary_cases.csv", index=False, encoding="utf-8-sig")

    texamples = target_examples(test, n=args.examples)
    pd.DataFrame(texamples).to_csv(out_dir / "target_examples.csv", index=False, encoding="utf-8-sig")

    # 2) 条件指标（可选）
    pred_path = Path(args.pred_csv) if args.pred_csv else find_default_pred_csv()
    cond_df = None
    oracle_df = None
    oracle_best = None
    err_df = None
    unc_df = None
    if pred_path is not None and pred_path.is_file():
        print(f"[info] 读取预测：{pred_path}")
        pred = pd.read_csv(pred_path)
        pred["id"] = pred["id"].astype(str)
        pred["toxic"] = pred["toxic"].astype(int)
        pred = attach_target_dims(pred, test)
        cond_df = conditional_metrics(pred, threshold=0.5)
        cond_df.to_csv(out_dir / "conditional_metrics_thr0.5.csv", index=False, encoding="utf-8-sig")
        oracle_df, oracle_best = oracle_threshold_sweep(pred)
        oracle_df.to_csv(out_dir / "oracle_threshold_sweep.csv", index=False, encoding="utf-8-sig")
        err_df = error_breakdown(pred, threshold=0.5)
        err_df.to_csv(out_dir / "error_breakdown.csv", index=False, encoding="utf-8-sig")
        unc_df = uncertainty_examples(pred, n=30, band=0.1)
        unc_df.to_csv(out_dir / "uncertain_examples.csv", index=False, encoding="utf-8-sig")
    else:
        print("[warn] 未找到 test_predictions.csv；跳过条件指标、阈值和错误分解。")

    # 3) 概念条件统计
    concept_df, concept_cols = load_concepts("test")
    concept_df = attach_raw_meta(concept_df, test)
    cs = concept_conditional_stats(concept_df, concept_cols)
    cs.to_csv(out_dir / "concept_conditional_stats.csv", index=False, encoding="utf-8-sig")
    tm = target_mapping(concept_df)
    tm.to_csv(out_dir / "target_mapping_auc.csv", index=False, encoding="utf-8-sig")

    # 4) 报告
    lines = []
    lines.append("ToxiCN 标注结构分析报告")
    lines.append("=" * 60)
    lines.append(f"时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("一、标注层级（推断）")
    lines.append("-" * 60)
    lines.append("toxic (binary)")
    lines.append("├─ 0: non-toxic；expression 多为 0，但 expression=0 中仍混有有毒样本")
    lines.append("└─ 1: toxic")
    lines.append("   ├─ toxic_type: 1 hate / 2 offensive")
    lines.append("   ├─ expression: 1/2/3")
    lines.append("   └─ target: 5 维，仅在有毒性样本上为正；可为多目标")
    lines.append("")
    lines.append("二、基础分布（test）")
    lines.append("-" * 60)
    for name in (
        "test_toxic_dist",
        "test_expression_dist",
        "test_toxic_type_dist",
        "test_target_any_dist",
        "test_target_count_dist",
    ):
        lines.append(f"[{name}]")
        for _, r in tables[name].iterrows():
            lines.append("  " + "  ".join(f"{c}={r[c]}" for c in tables[name].columns))
    lines.append("")
    lines.append("[test_toxic_x_expression]")
    lines.append(tables["test_toxic_x_expression"].to_string(index=False))
    lines.append("")
    lines.append("[test_toxic_x_toxic_type]")
    lines.append(tables["test_toxic_x_toxic_type"].to_string(index=False))
    lines.append("")
    lines.append("[test_target_any_x_toxic]")
    lines.append(tables["test_target_any_x_toxic"].to_string(index=False))
    lines.append("")
    lines.append("[test_target_cooccurrence]")
    lines.append(tables["test_target_cooccurrence"].to_string())
    lines.append("")
    lines.append("三、边界噪声")
    lines.append("-" * 60)
    for r in boundaries:
        lines.append(f"{r['case']}: n={r['n']}（{r['rate']*100:.2f}%）")
        for ex in r["examples"]:
            lines.append(f"  - {ex}")
    lines.append("")
    lines.append("四、target 维度语义与例子（推断）")
    lines.append("-" * 60)
    for r in texamples:
        lines.append(f"{r['target_dim']} -> {r['inferred_semantics']}（n_toxic={r['n_toxic']}）")
        for ex in r["examples"]:
            lines.append(f"  - {ex}")
    lines.append("")
    lines.append("五、target 映射（T 概念 -> raw target）")
    lines.append("-" * 60)
    if len(tm):
        for t, sub in tm.groupby("concept"):
            best = sub.loc[sub["auc"].idxmax()]
            expected = EXPECTED_T_TO_TARGET.get(t)
            ok = int(best["target_dim"].split("_")[1]) == expected
            lines.append(
                f"{t}({T_CONCEPT_SEMANTICS[t]}) -> {best['target_dim']}"
                f"({best['target_semantics_inferred']}) AUC={fmt(best['auc'])}"
                f"；与推断映射{'一致' if ok else '不一致'}"
            )
    lines.append("")
    if cond_df is not None:
        lines.append("六、条件指标（threshold=0.5）")
        lines.append("-" * 60)
        for _, r in cond_df.iterrows():
            lines.append(
                f"{r['stratum']:<34} n={int(r['n']):<5} toxic_rate={fmt(r['toxic_rate'],3)} "
                f"MacroF1={fmt(r['macro_f1'])} PosF1={fmt(r['pos_f1'])} "
                f"P={fmt(r['pos_precision'])} R={fmt(r['pos_recall'])} "
                f"FP_rate={fmt(r['fp_rate_non_toxic'])} FN_rate={fmt(r['fn_rate_toxic'])}"
            )
        lines.append("")
        lines.append("七、oracle 阈值扫描（仅诊断，不可用于模型选择）")
        lines.append("-" * 60)
        lines.append(
            f"best threshold={fmt(oracle_best['threshold'],2)} "
            f"MacroF1={fmt(oracle_best['macro_f1'])} "
            f"PosF1={fmt(oracle_best['pos_f1'])} "
            f"P={fmt(oracle_best['pos_precision'])} R={fmt(oracle_best['pos_recall'])} "
            f"FP_rate={fmt(oracle_best['fp_rate_non_toxic'])} FN_rate={fmt(oracle_best['fn_rate_toxic'])}"
        )
        lines.append("")
        lines.append("八、错误分解（threshold=0.5）")
        lines.append("-" * 60)
        lines.append(err_df.to_string(index=False))
        lines.append("")
        lines.append("九、高不确定样本（|p-0.5|<=0.1，前 20 条）")
        lines.append("-" * 60)
        for _, r in unc_df.head(20).iterrows():
            lines.append(
                f"{r['id']} toxic={r['toxic']} expr={r['expression']} type={r['toxic_type']} "
                f"p={fmt(r['pred_prob'],3)} | {truncate(r['content'], 120)}"
            )
        lines.append("")
    else:
        lines.append("六、条件指标：未找到预测文件，跳过。")
        lines.append("")

    lines.append("十、概念条件统计（每组 top 5，按 |mean_diff| 排序）")
    lines.append("-" * 60)
    for stratum, sub in cs.groupby("stratum"):
        top = sub.reindex(sub["mean_diff"].abs().sort_values(ascending=False).index).head(5)
        lines.append(f"[{stratum}]")
        for _, r in top.iterrows():
            lines.append(
                f"  {r['concept']}: mean_pos={fmt(r['mean_pos'],3)} "
                f"mean_neg={fmt(r['mean_neg'],3)} diff={fmt(r['mean_diff'],3)} AUC={fmt(r['auc'])}"
            )
    lines.append("")
    lines.append("十一、对 soft conditional CBM 的直接含义")
    lines.append("-" * 60)
    lines.append("- target 只在有毒样本上为正，因此 target 分支必须条件于 toxic，不能作为全局概念。")
    lines.append("- expression=0 混有有毒样本，因此 toxicity gate 必须是 soft 的，不能硬编码 expression。")
    lines.append("- hate 与 offensive 的样本量/难度不同，type 分支应单独建模，不能用同一 flat 读出。")
    lines.append("- T 概念与 raw target 的映射已确认；T 概念本身有效，问题在读出是否独立使用它们。")
    lines.append("- 边界噪声和高不确定样本需要单独报告；0.84 目标必须考虑标注噪声上限。")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] 结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
