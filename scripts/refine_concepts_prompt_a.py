#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt A：对 INDEPENDENT-CBM 的 top-k 概念做 LLM 条件校验。

只实现 Prompt A，不做 Prompt B/C，不做重训。

流程
----
1. 读取 true-INDEPENDENT-CBM 的 validation_predictions.csv，拿到 g 的 21 个
   A-E 概念概率、f 的 pipeline 概率/预测，以及文本。
2. 每个 validation 样本选 top-k / 候选 / 不确定概念。
3. 对每个 (样本, 概念) 构造 Prompt A：
   原文 + 概念定义/边界 + g 的离散初判 + 其他激活概念。
4. 用 vLLM 跑 GLM-4-9B-Chat，沿用 run_items_v8.py 的“是/否”首 token
   约束与概率提取。
5. 输出 refinement_pairs.csv、prompts.jsonl、report.txt。
6. 可选诊断：把被查询概念的 p_g 替换成 p_llm，在同一个 validation 上
   跑 f，观察 Macro-F1 变化。该诊断是 in-sample，不作为最终结论。

重要
----
- 只用 validation，不碰 test。
- 不注入 gold label；prompt 里只有 g 的离散初判和 f 的预测（可选）。
- 不暴露 g 的具体概率，避免 LLM 锚定。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 复用 run_items_v8.py 的 vLLM 加载、chat template、是/否首 token 提取。
# 该模块顶层不导入 vLLM，只有调用函数时才导入，因此 dry-run 不需要 vLLM。
import run_items_v8 as v8  # noqa: E402

CBM_DIR_DEFAULT = (
    PROJECT_ROOT
    / "scripts"
    / "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0"
)
TRAIN_JSON_DEFAULT = CBM_DIR_DEFAULT / "data" / "train.json"
DICT_DEFAULT = CBM_DIR_DEFAULT / "data" / "v8_dictionary.csv"
OUTPUTS_DEFAULT = CBM_DIR_DEFAULT / "outputs" / "independent_true_seed20261122"
TRAIN_PRED_DEFAULT = OUTPUTS_DEFAULT / "train_predictions.csv"
PRED_DEFAULT = TRAIN_PRED_DEFAULT
SPLIT_DEFAULT = OUTPUTS_DEFAULT / "train_validation_split.csv"
F_CKPT_DEFAULT = CBM_DIR_DEFAULT / "outputs" / "independent_f_best.pt"

# 与 INDEPENDENT 脚本一致：26 个概念，f 只吃前 21 个 A-E。
CONCEPT_COLS = [
    "A01", "A02", "A03",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04",
    "D01", "D02",
    "E01", "E02", "E03", "E04", "E05",
    "T01", "T02", "T03", "T04", "T05",
]
TASK_CONCEPT_COLS = CONCEPT_COLS[:21]
CONCEPT_SET = set(CONCEPT_COLS)

PROMPT_VERSION = "prompt-a-concept-verify-v1.0.0"


def parse_args():
    p = argparse.ArgumentParser(
        description="Prompt A: LLM 条件校验 INDEPENDENT-CBM 的 top-k 概念"
    )
    p.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="本地 GLM-4-9B-Chat 目录；dry-run 可不填",
    )
    p.add_argument("--train_json", type=Path, default=TRAIN_JSON_DEFAULT)
    p.add_argument("--dict_path", type=Path, default=DICT_DEFAULT)
    p.add_argument(
        "--predictions",
        type=Path,
        default=PRED_DEFAULT,
        help="默认 train_predictions.csv；也可传 validation_predictions.csv",
    )
    p.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="在哪个 split 上做 LLM 精修；默认 train",
    )
    p.add_argument("--split_csv", type=Path, default=SPLIT_DEFAULT)
    p.add_argument("--f_ckpt", type=Path, default=F_CKPT_DEFAULT)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="默认 results/prompt_a_refinement/<timestamp>",
    )
    p.add_argument(
        "--concept_ids",
        type=str,
        default=",".join(TASK_CONCEPT_COLS),
        help=(
            "候选概念池；每个样本在该池内选 top-k 校验，"
            "默认全部 21 个 A-E"
        ),
    )
    p.add_argument(
        "--selection",
        choices=["gap", "prob", "union"],
        default="gap",
        help=(
            "gap: 按 |p_g - teacher_c| 取 top-k（推荐）；"
            "prob: 按 p_g 取 top-k；union: gap+prob+候选+不确定"
        ),
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--candidate_threshold", type=float, default=0.5)
    p.add_argument("--uncertain_low", type=float, default=0.2)
    p.add_argument("--uncertain_high", type=float, default=0.8)
    p.add_argument(
        "--max_concepts_per_sample",
        type=int,
        default=5,
        help="每个样本最多校验多少个概念；<=0 表示不限制",
    )
    p.add_argument(
        "--limit_samples",
        type=int,
        default=200,
        help="只取前 N 个 split 样本做 pilot；0 表示全部（train=7680, validation=1920）",
    )
    p.add_argument(
        "--include_f_decision",
        action="store_true",
        help="在 prompt 中加入 f 的预测（不是 gold label）；默认不加，避免锚定",
    )
    p.add_argument("--max_active_in_prompt", type=int, default=8)
    p.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85
    )
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="只构造 prompt 和选择统计，不加载 vLLM、不跑 f 诊断",
    )
    p.add_argument(
        "--skip_f_eval",
        action="store_true",
        help="跳过 f 的 in-sample 替换诊断",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_dictionary(path: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = ["条目id", "构念名", "条目问句", "概念定义", "边界说明"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"词典缺少列：{miss}")
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        cid = str(row["条目id"]).strip()
        out[cid] = {
            "name": str(row["构念名"]).strip(),
            "question": str(row["条目问句"]).strip(),
            "definition": str(row["概念定义"]).strip(),
            "boundary": str(row["边界说明"]).strip(),
        }
    return out


def _load_split(
    train_json: Path,
    predictions: Path,
    split_csv: Path,
    split_name: str,
) -> pd.DataFrame:
    raw = json.loads(train_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("train.json 顶层必须是 list")
    df = pd.DataFrame(raw)
    if len(df) != 9600:
        raise ValueError(f"train.json 行数不是 9600：{len(df)}")
    df.insert(0, "id", [f"TOXICN_train_{i:04d}" for i in range(len(df))])

    split = pd.read_csv(split_csv, encoding="utf-8-sig")
    if not {"id", "split"}.issubset(split.columns):
        raise ValueError("split_csv 缺少 id/split 列")
    pred = pd.read_csv(predictions, encoding="utf-8-sig")
    if "id" not in pred.columns:
        raise ValueError("predictions 缺少 id 列")
    # train.json 已有这两列，避免 merge 后出现 _x/_y 重复。
    pred = pred.drop(
        columns=[c for c in ("toxic_type", "expression") if c in pred.columns]
    )

    need_pred = [
        "pipeline_prob_toxic_f_of_g_x",
        "pipeline_pred_toxic",
        "toxic_true",
    ]
    miss = [c for c in need_pred if c not in pred.columns]
    if miss:
        raise ValueError(f"predictions 缺少列：{miss}")
    for cid in TASK_CONCEPT_COLS:
        for col in (
            f"predicted_concept_{cid}",
            f"concept_target_{cid}",
        ):
            if col not in pred.columns:
                raise ValueError(f"predictions 缺少列：{col}")

    df = df.merge(split[["id", "split"]], on="id", how="inner", validate="one_to_one")
    df = df.merge(pred, on="id", how="inner", validate="one_to_one")
    if "content" not in df.columns:
        raise ValueError("train.json 缺少 content 列")
    if df["content"].isna().any():
        raise ValueError("content 存在空值")
    out = df[df["split"] == split_name].copy()
    expected = 7680 if split_name == "train" else 1920
    if len(out) != expected:
        raise ValueError(
            f"{split_name} split 行数不是 {expected}：{len(out)}"
        )
    return out.sort_values("id", kind="mergesort").reset_index(drop=True)


def _compact_concept(cid: str, meta: Dict[str, Dict[str, str]]) -> str:
    return f"{cid} {meta[cid]['name']}"


def _build_prompt(
    row: pd.Series,
    cid: str,
    p_g: float,
    meta: Dict[str, Dict[str, str]],
    active_other: List[str],
    include_f_decision: bool,
) -> str:
    item = meta[cid]
    candidate = "候选" if p_g >= 0.5 else "非候选"
    if active_other:
        active_text = "；".join(_compact_concept(c, meta) for c in active_other)
    else:
        active_text = "无"
    f_line = ""
    if include_f_decision:
        f_prob = float(row["pipeline_prob_toxic_f_of_g_x"])
        f_pred = "toxic" if int(row["pipeline_pred_toxic"]) == 1 else "non-toxic"
        f_line = f"下游判断：{f_pred}（概率 {f_prob:.3f}）\n"

    return (
        "你是严格的中文文本概念校验助手。请只依据原文判断指定概念是否成立，"
        "不要因为“模型初判”而附和。\n\n"
        "【原文】\n"
        f"{row['content']}\n\n"
        "【概念】\n"
        f"{cid} {item['name']}\n"
        f"问句：{item['question']}\n"
        f"定义：{item['definition']}\n"
        f"边界：{item['boundary']}\n\n"
        "【上下文】\n"
        f"模型初判：{candidate}\n"
        f"其他激活概念：{active_text}\n"
        f"{f_line}"
        "\n【要求】\n"
        "1. 只有原文明确支持该概念时才回答“是”；证据不足回答“否”。\n"
        "2. 引用、转述、反对、假设、否定范围内的表述，不等于发言者使用。\n"
        "3. 先只回答“是”或“否”。\n\n"
        "回答："
    )


def _select_pairs(
    val: pd.DataFrame,
    concept_ids: List[str],
    meta: Dict[str, Dict[str, str]],
    args,
) -> List[dict]:
    pairs: List[dict] = []
    for _, row in val.iterrows():
        probs = np.array(
            [float(row[f"predicted_concept_{cid}"]) for cid in concept_ids],
            dtype=np.float64,
        )
        targets = np.array(
            [float(row[f"concept_target_{cid}"]) for cid in concept_ids],
            dtype=np.float64,
        )
        targets = np.where(np.isfinite(targets), targets, 0.5)
        gaps = np.abs(probs - targets)
        prob_order = np.argsort(-probs, kind="mergesort")
        gap_order = np.argsort(-gaps, kind="mergesort")
        prob_rank = {
            concept_ids[int(i)]: r + 1 for r, i in enumerate(prob_order)
        }
        gap_rank = {
            concept_ids[int(i)]: r + 1 for r, i in enumerate(gap_order)
        }

        active = [
            concept_ids[int(i)]
            for i in range(len(concept_ids))
            if probs[i] >= args.candidate_threshold
        ]

        k = max(1, args.top_k)
        if args.selection == "gap":
            selected = set(int(i) for i in gap_order[:k])
        elif args.selection == "prob":
            selected = set(int(i) for i in prob_order[:k])
        else:
            selected = set(int(i) for i in gap_order[:k])
            selected |= set(int(i) for i in prob_order[:k])
            selected |= {
                int(i)
                for i in range(len(concept_ids))
                if probs[i] >= args.candidate_threshold
            }
            selected |= {
                int(i)
                for i in range(len(concept_ids))
                if args.uncertain_low <= probs[i] <= args.uncertain_high
            }
        if not selected:
            selected = {int(gap_order[0])}

        if args.selection == "prob":
            selected_list = sorted(
                selected, key=lambda i: (-probs[i], concept_ids[i])
            )
        else:
            selected_list = sorted(
                selected, key=lambda i: (-gaps[i], -probs[i], concept_ids[i])
            )
        if args.max_concepts_per_sample > 0:
            selected_list = selected_list[: args.max_concepts_per_sample]

        for i in selected_list:
            cid = concept_ids[i]
            p_g = float(probs[i])
            active_other = [c for c in active if c != cid]
            if args.max_active_in_prompt > 0:
                active_other = active_other[: args.max_active_in_prompt]
            prompt = _build_prompt(
                row, cid, p_g, meta, active_other, args.include_f_decision
            )
            pairs.append(
                {
                    "sample_id": str(row["id"]),
                    "concept_id": cid,
                    "concept_name": meta[cid]["name"],
                    "p_g": p_g,
                    "teacher_c": float(targets[i]),
                    "gap": float(gaps[i]),
                    "g_rank": int(prob_rank[cid]),
                    "gap_rank": int(gap_rank[cid]),
                    "g_candidate": int(p_g >= args.candidate_threshold),
                    "active_concepts": "；".join(active) if active else "无",
                    "f_prob": float(row["pipeline_prob_toxic_f_of_g_x"]),
                    "f_pred": int(row["pipeline_pred_toxic"]),
                    "toxic_true": int(row["toxic_true"]),
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                }
            )
    return pairs


class _TaskPredictor(nn.Module):
    """与 INDEPENDENT 脚本中的 f 完全一致：21 -> 128 -> 1。"""

    def __init__(self, input_dim: int = len(TASK_CONCEPT_COLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, c):
        return self.net(c).squeeze(-1)


def _load_f(path: Path) -> Optional[_TaskPredictor]:
    if not path.is_file():
        return None
    model = _TaskPredictor()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_prob": float(np.mean(p)),
    }


def _eval_refined_f(
    val: pd.DataFrame,
    pairs: List[dict],
    f_ckpt: Path,
) -> Tuple[pd.DataFrame, Optional[str]]:
    model = _load_f(f_ckpt)
    if model is None:
        return pd.DataFrame(), f"找不到 f checkpoint：{f_ckpt}"

    x_orig = np.array(
        [
            [float(row[f"predicted_concept_{cid}"]) for cid in TASK_CONCEPT_COLS]
            for _, row in val.iterrows()
        ],
        dtype=np.float32,
    )
    y = val["toxic_true"].to_numpy(dtype=int)
    sid_to_pos = {str(sid): i for i, sid in enumerate(val["id"].astype(str))}
    cid_to_col = {cid: j for j, cid in enumerate(TASK_CONCEPT_COLS)}

    x_ref = x_orig.copy()
    replaced = 0
    for pair in pairs:
        p_llm = pair.get("p_llm")
        if p_llm is None or not math.isfinite(float(p_llm)):
            continue
        pos = sid_to_pos.get(pair["sample_id"])
        col = cid_to_col.get(pair["concept_id"])
        if pos is None or col is None:
            continue
        x_ref[pos, col] = float(p_llm)
        replaced += 1

    with torch.no_grad():
        p_orig = torch.sigmoid(model(torch.as_tensor(x_orig))).numpy()
        p_ref = torch.sigmoid(model(torch.as_tensor(x_ref))).numpy()

    rows = []
    for name, p in (
        ("original_f_g_x", p_orig),
        ("refined_replace_f", p_ref),
    ):
        m = _metrics(y, p)
        m["variant"] = name
        rows.append(m)

    # 可选：oracle f(c) 仅作参照，不作为模型输入。
    if "oracle_prob_toxic_f_of_c" in val.columns:
        p_oracle = val["oracle_prob_toxic_f_of_c"].to_numpy(dtype=float)
        m = _metrics(y, p_oracle)
        m["variant"] = "oracle_f_teacher_c"
        rows.append(m)

    df = pd.DataFrame(rows)[
        ["variant", "n", "macro_f1", "auc", "tp", "tn", "fp", "fn", "mean_prob"]
    ]
    note = (
        f"refined_replace 替换了 {replaced} 个 (sample, concept) 单元；"
        "该诊断与选择使用同一 split；若为 train 则是 in-sample 诊断，"
        "不作为最终结论。"
    )
    return df, note


def _agreement_report(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = pairs_df.dropna(subset=["p_llm"]).copy()
    for cid, sub in valid.groupby("concept_id", sort=True):
        pg = sub["p_g"].to_numpy(dtype=float)
        pl = sub["p_llm"].to_numpy(dtype=float)
        flip = float(
            np.mean((pg >= 0.5).astype(int) != (pl >= 0.5).astype(int))
        ) if len(sub) else float("nan")
        mae = float(np.mean(np.abs(pg - pl))) if len(sub) else float("nan")
        if len(sub) >= 2 and np.std(pg) > 1e-9 and np.std(pl) > 1e-9:
            corr = float(np.corrcoef(pg, pl)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "concept_id": cid,
                "n": int(len(sub)),
                "mean_p_g": float(np.mean(pg)) if len(sub) else float("nan"),
                "mean_p_llm": float(np.mean(pl)) if len(sub) else float("nan"),
                "mae": mae,
                "pearson": corr,
                "flip_rate": flip,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    args,
    pairs_df: pd.DataFrame,
    agreement: pd.DataFrame,
    f_metrics: pd.DataFrame,
    f_note: Optional[str],
    n_val: int,
    split_name: str,
) -> None:
    lines = []
    lines.append("Prompt A：INDEPENDENT-CBM 概念条件校验报告")
    lines.append("=" * 76)
    lines.append(f"时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"prompt：{PROMPT_VERSION}")
    lines.append(f"{split_name} 样本：{n_val}")
    lines.append(f"查询概念数：{len(args.concept_ids_list)}；pairs={len(pairs_df)}")
    lines.append(
        f"选择：selection={args.selection}, top_k={args.top_k}, "
        f"candidate>={args.candidate_threshold}, "
        f"uncertain=[{args.uncertain_low}, {args.uncertain_high}], "
        f"max_per_sample={args.max_concepts_per_sample}"
    )
    lines.append(f"include_f_decision={args.include_f_decision}")
    lines.append("")

    lines.append("一、选择分布")
    lines.append("-" * 76)
    if len(pairs_df):
        sel = (
            pairs_df.groupby("concept_id")
            .agg(
                n=("concept_id", "size"),
                mean_p_g=("p_g", "mean"),
                mean_gap=("gap", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        for _, r in sel.iterrows():
            lines.append(
                f"{r['concept_id']:>4s}  n={int(r['n']):5d}  "
                f"mean_p_g={r['mean_p_g']:.4f}  mean_gap={r['mean_gap']:.4f}"
            )
    else:
        lines.append("无 pairs")
    lines.append("")

    lines.append("二、LLM 校验一致性（只统计有效首 token）")
    lines.append("-" * 76)
    if len(agreement):
        lines.append(
            f"{'concept':>8s}{'n':>6s}{'mean_p_g':>12s}{'mean_p_llm':>13s}"
            f"{'MAE':>9s}{'pearson':>10s}{'flip':>9s}"
        )
        for _, r in agreement.iterrows():
            lines.append(
                f"{r['concept_id']:>8s}{int(r['n']):6d}{r['mean_p_g']:12.4f}"
                f"{r['mean_p_llm']:13.4f}{r['mae']:9.4f}"
                f"{r['pearson']:10.4f}{r['flip_rate']:9.4f}"
            )
    else:
        lines.append("无有效 p_llm")
    lines.append("")

    lines.append(f"三、f 的 {split_name} 替换诊断")
    lines.append("-" * 76)
    if len(f_metrics):
        lines.append(
            f"{'variant':<24s}{'MacroF1':>9s}{'AUC':>9s}"
            f"{'FP':>7s}{'FN':>7s}{'mean_prob':>11s}"
        )
        for _, r in f_metrics.iterrows():
            lines.append(
                f"{r['variant']:<24s}{r['macro_f1']:9.4f}{r['auc']:9.4f}"
                f"{int(r['fp']):7d}{int(r['fn']):7d}{r['mean_prob']:11.4f}"
            )
    else:
        lines.append(f_note or "未运行")
    if f_note:
        lines.append("")
        lines.append(f"说明：{f_note}")
    lines.append("")

    lines.append("四、限制")
    lines.append("-" * 76)
    lines.append(f"1. 只使用 {split_name} split，不碰 test；test 仍保持冻结。")
    lines.append("2. prompt 不含 gold label；include_f_decision 只含 f 的预测。")
    lines.append("3. refined_replace 是 in-sample 诊断，不能当作最终 F1 提升。")
    lines.append("4. p_llm 是首 token 是/否概率；cand_mass 低或 first_token 非是/否时记为无效。")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.concept_ids_list = [
        x.strip() for x in str(args.concept_ids).split(",") if x.strip()
    ]
    unknown = [c for c in args.concept_ids_list if c not in CONCEPT_SET]
    if unknown:
        raise ValueError(f"未知概念：{unknown}")
    if not args.concept_ids_list:
        raise ValueError("--concept_ids 为空")
    if args.top_k <= 0:
        raise ValueError("--top_k 必须为正")
    if not (0.0 <= args.uncertain_low <= args.uncertain_high <= 1.0):
        raise ValueError("uncertain 区间非法")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (
        PROJECT_ROOT
        / "results"
        / "prompt_a_refinement"
        / f"{args.split}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_dictionary(args.dict_path)
    missing = [c for c in args.concept_ids_list if c not in meta]
    if missing:
        raise ValueError(f"词典缺少概念：{missing}")

    split_df = _load_split(
        args.train_json, args.predictions, args.split_csv, args.split
    )
    if args.limit_samples and args.limit_samples > 0:
        split_df = split_df.head(args.limit_samples).copy()
    if not (split_df["id"].astype(str).is_unique):
        raise ValueError(f"{args.split} id 不唯一")

    pairs = _select_pairs(split_df, args.concept_ids_list, meta, args)
    pairs_df = pd.DataFrame(
        [{k: v for k, v in p.items() if k != "prompt"} for p in pairs]
    )
    prompts = [p["prompt"] for p in pairs]

    # 先落盘 prompt，便于 dry-run 审计和正式运行复用。
    with (out_dir / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "sample_id": p["sample_id"],
                        "concept_id": p["concept_id"],
                        "p_g": p["p_g"],
                        "prompt": p["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[info] split={args.split} n={len(split_df)} pairs={len(pairs)} "
        f"concepts={len(args.concept_ids_list)} dry_run={args.dry_run}",
        flush=True,
    )
    if prompts:
        lengths = np.array([len(x) for x in prompts])
        print(
            f"[info] prompt chars: mean={lengths.mean():.1f} "
            f"max={lengths.max()} p95={np.percentile(lengths, 95):.1f}",
            flush=True,
        )

    if args.dry_run:
        summary = pd.DataFrame(
            [
                {
                    "sample_id": p["sample_id"],
                    "concept_id": p["concept_id"],
                    "p_g": p["p_g"],
                    "teacher_c": p["teacher_c"],
                    "gap": p["gap"],
                    "g_rank": p["g_rank"],
                    "gap_rank": p["gap_rank"],
                    "g_candidate": p["g_candidate"],
                    "prompt_chars": p["prompt_chars"],
                }
                for p in pairs
            ]
        )
        summary.to_csv(
            out_dir / "selection_summary.csv", index=False, encoding="utf-8-sig"
        )
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "version": PROMPT_VERSION,
                    "dry_run": True,
                    "split": args.split,
                    "n_val": len(split_df),
                    "n_pairs": len(pairs),
                    "args": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[done] dry-run 输出：{out_dir}", flush=True)
        return

    if args.model_path is None:
        raise ValueError("正式运行必须提供 --model_path")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{args.model_path}")

    tokenizer, llm = v8.load_vllm_model(
        args.model_path,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
    )
    chat_kwargs = v8.detect_chat_kwargs(tokenizer)
    yes_ids = v8.get_candidate_ids(["是"], tokenizer)
    no_ids = v8.get_candidate_ids(["否"], tokenizer)
    sampling, restrict_mode = v8.build_sampling_params(yes_ids + no_ids)
    print(f"[info] 首 token 约束方式：{restrict_mode}", flush=True)

    rendered = v8.render_prompts(
        tokenizer, prompts, chat_kwargs, args.model_path.name.lower()
    )
    outputs = llm.generate(rendered, sampling, use_tqdm=True)
    if len(outputs) != len(pairs):
        raise RuntimeError(f"vLLM 输出数 {len(outputs)} != pairs {len(pairs)}")

    p_llm: List[Optional[float]] = []
    first_tokens: List[str] = []
    first_ids: List[int] = []
    cand_masses: List[float] = []
    for out in outputs:
        try:
            res = v8.get_probability_result(out, tokenizer, yes_ids, no_ids)
            p_llm.append(res["p_yes"])
            first_tokens.append(res["first_token"])
            first_ids.append(int(res["first_token_id"]))
            cand_masses.append(float(res["cand_mass"]))
        except Exception as exc:  # 单条失败不丢弃整轮结果
            p_llm.append(None)
            first_tokens.append(f"ERROR:{type(exc).__name__}")
            first_ids.append(-1)
            cand_masses.append(0.0)

    pairs_df["p_llm"] = p_llm
    pairs_df["first_token"] = first_tokens
    pairs_df["first_token_id"] = first_ids
    pairs_df["cand_mass"] = cand_masses
    pairs_df["valid_yes_no"] = [
        1 if t in {"是", "否"} and p is not None else 0
        for t, p in zip(first_tokens, p_llm)
    ]
    pairs_df.to_csv(
        out_dir / "refinement_pairs.csv", index=False, encoding="utf-8-sig"
    )

    agreement = _agreement_report(pairs_df)

    f_metrics = pd.DataFrame()
    f_note = None
    if not args.skip_f_eval:
        # 把 p_llm 回填到 pairs，供 f 诊断使用。
        for pair, pl in zip(pairs, p_llm):
            pair["p_llm"] = pl
        f_metrics, f_note = _eval_refined_f(split_df, pairs, args.f_ckpt)
        if len(f_metrics):
            f_metrics.to_csv(
                out_dir / "metrics_refined_f.csv",
                index=False,
                encoding="utf-8-sig",
            )

    _write_report(
        out_dir / "report.txt",
        args,
        pairs_df,
        agreement,
        f_metrics,
        f_note,
        n_val=len(split_df),
        split_name=args.split,
    )

    valid_n = int(pairs_df["valid_yes_no"].sum())
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "version": PROMPT_VERSION,
                "dry_run": False,
                "split": args.split,
                "predictions": str(args.predictions),
                "model_path": str(args.model_path),
                "n_val": len(split_df),
                "n_pairs": len(pairs),
                "n_valid_yes_no": valid_n,
                "valid_rate": valid_n / max(1, len(pairs)),
                "mean_cand_mass": float(np.mean(cand_masses))
                if cand_masses
                else None,
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
