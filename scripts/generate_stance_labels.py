#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用本地 GLM-4-9B-Chat 给 ToxiCN 子集生成 stance / target 标签。

只做离线标注，不训练模型，不碰 test。

输出：
  data/processed/TOXICN/stance_labels/stance_labels.csv
  data/processed/TOXICN/stance_labels/stance_label_meta.json

列：
  id, split, toxic_true, toxic_type, expression,
  stance, target, valid, raw_output

stance ∈ {use, mention, oppose, none}
target ∈ {LGBTQ, region, gender, race, other, none}

用法：
  python scripts/generate_stance_labels.py --model_path /root/autodl-tmp/glm-4-9b-chat
  python scripts/generate_stance_labels.py --dry_run
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

CBM_DIR = SCRIPT_DIR / "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0"
TRAIN_JSON = CBM_DIR / "data" / "train.json"
SPLIT_CSV = (
    CBM_DIR
    / "outputs"
    / "independent_true_seed20261122"
    / "train_validation_split.csv"
)
TRAIN_PRED = (
    CBM_DIR
    / "outputs"
    / "independent_true_seed20261122"
    / "train_predictions.csv"
)
VAL_PRED = (
    CBM_DIR
    / "outputs"
    / "independent_true_seed20261122"
    / "validation_predictions.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "data" / "processed" / "TOXICN" / "stance_labels"
    / "stance_labels.csv"
)

STANCES = {"use", "mention", "oppose", "none"}
TARGETS = {"LGBTQ", "region", "gender", "race", "other", "none"}
PROMPT_VERSION = "stance-target-v1.0.0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, default=None)
    p.add_argument("--train_json", type=Path, default=TRAIN_JSON)
    p.add_argument("--split_csv", type=Path, default=SPLIT_CSV)
    p.add_argument("--train_predictions", type=Path, default=TRAIN_PRED)
    p.add_argument("--val_predictions", type=Path, default=VAL_PRED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument(
        "--train_alloc",
        type=str,
        default="400,200,400",
        help="train 按 toxic_type 0/1/2 的采样数；hate 过采样",
    )
    p.add_argument(
        "--val_alloc",
        type=str,
        default="200,100,200",
        help="validation 按 toxic_type 0/1/2 的采样数",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_pool(args) -> pd.DataFrame:
    raw = json.loads(args.train_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 9600:
        raise ValueError("train.json 必须是 9600 行的 list")
    df = pd.DataFrame(raw)
    df.insert(0, "id", [f"TOXICN_train_{i:04d}" for i in range(len(df))])
    for col in ("content", "toxic", "toxic_type", "expression"):
        if col not in df.columns:
            raise ValueError(f"train.json 缺少列：{col}")

    split = pd.read_csv(args.split_csv, encoding="utf-8-sig")
    if not {"id", "split"}.issubset(split.columns):
        raise ValueError("split_csv 缺少 id/split")
    df = df.merge(split[["id", "split"]], on="id", how="inner", validate="one_to_one")
    df = df.rename(columns={"toxic": "toxic_true"})
    return df


def sample_by_type(
    df: pd.DataFrame, split: str, alloc: dict, seed: int
) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    picked = []
    for t, n in alloc.items():
        part = sub[sub["toxic_type"] == t]
        if n >= len(part):
            picked.append(part)
        else:
            picked.append(part.sample(n=n, random_state=seed + int(t)))
    out = pd.concat(picked, ignore_index=True) if picked else sub.iloc[0:0]
    # 去重并按 id 排序，保证可复现
    out = out.drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True)
    return out


def build_prompt(text: str) -> str:
    return (
        "你是中文毒性内容标注助手。只判断“发言者立场”和“攻击目标”，"
        "不要判断整条文本是否有毒。\n\n"
        "【立场 stance】\n"
        "- use：发言者自己在使用攻击、辱骂或歧视性表达；\n"
        "- mention：发言者在引用、转述或报道他人的攻击/歧视表达；\n"
        "- oppose：发言者在反对、谴责或反驳攻击/歧视表达；\n"
        "- none：文本没有攻击/歧视内容。\n\n"
        "【目标 target】\n"
        "- LGBTQ、region、gender、race、other、none；\n"
        "- 目标不明确时选 none。\n\n"
        "【原文】\n"
        f"{text}\n\n"
        "只输出一行 JSON，不要解释：\n"
        '{"stance":"use|mention|oppose|none",'
        '"target":"LGBTQ|region|gender|race|other|none"}'
    )


def parse_output(text: str):
    text = (text or "").strip()
    obj = None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        sm = re.search(r'"stance"\s*:\s*"([^"]+)"', text, re.I)
        tm = re.search(r'"target"\s*:\s*"([^"]+)"', text, re.I)
        if sm and tm:
            obj = {"stance": sm.group(1), "target": tm.group(1)}
    if obj is None:
        return None, None, 0
    stance = str(obj.get("stance", "")).strip().lower()
    target = str(obj.get("target", "")).strip()
    target_map = {x.lower(): x for x in TARGETS}
    target = target_map.get(target.lower(), target)
    valid = int(stance in STANCES and target in TARGETS)
    return stance, target, valid


def main():
    args = parse_args()
    if args.out.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"输出已存在：{args.out}；确认覆盖请加 --overwrite")

    df = load_pool(args)
    train_alloc = {
        i: int(x) for i, x in enumerate(str(args.train_alloc).split(","))
    }
    val_alloc = {
        i: int(x) for i, x in enumerate(str(args.val_alloc).split(","))
    }
    if set(train_alloc) != {0, 1, 2} or set(val_alloc) != {0, 1, 2}:
        raise ValueError("alloc 必须包含 toxic_type 0/1/2")
    sampled = pd.concat(
        [
            sample_by_type(df, "train", train_alloc, args.seed),
            sample_by_type(df, "validation", val_alloc, args.seed + 1),
        ],
        ignore_index=True,
    )
    if sampled["id"].duplicated().any():
        raise ValueError("采样后 id 重复")
    args.n_train = int((sampled["split"] == "train").sum())
    args.n_val = int((sampled["split"] == "validation").sum())

    prompts = [build_prompt(str(t)) for t in sampled["content"]]
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        preview = out_dir / "prompts_preview.jsonl"
        with preview.open("w", encoding="utf-8") as f:
            for sid, text, prompt in zip(sampled["id"], sampled["content"], prompts):
                f.write(
                    json.dumps(
                        {"id": sid, "text": text, "prompt": prompt},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        sampled[["id", "split", "toxic_true", "toxic_type", "expression"]].to_csv(
            out_dir / "sampled_ids.csv", index=False, encoding="utf-8-sig"
        )
        print(
            f"[done] dry-run: {len(sampled)} samples -> {preview}", flush=True
        )
        return

    if args.model_path is None:
        raise ValueError("正式运行必须提供 --model_path")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{args.model_path}")

    import run_items_v8 as v8
    from vllm import SamplingParams

    tokenizer, llm = v8.load_vllm_model(
        args.model_path,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
    )
    chat_kwargs = v8.detect_chat_kwargs(tokenizer)
    rendered = v8.render_prompts(
        tokenizer, prompts, chat_kwargs, args.model_path.name.lower()
    )
    sampling = SamplingParams(max_tokens=48, temperature=0)
    outputs = llm.generate(rendered, sampling, use_tqdm=True)
    if len(outputs) != len(sampled):
        raise RuntimeError(f"vLLM 输出数 {len(outputs)} != {len(sampled)}")

    stances, targets, valids, raws = [], [], [], []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        stance, target, valid = parse_output(text)
        stances.append(stance if stance else "invalid")
        targets.append(target if target else "invalid")
        valids.append(valid)
        raws.append(text[:300])

    result = sampled[
        ["id", "split", "toxic_true", "toxic_type", "expression"]
    ].copy()
    result["stance"] = stances
    result["target"] = targets
    result["valid"] = valids
    result["raw_output"] = raws
    result.to_csv(args.out, index=False, encoding="utf-8-sig")

    valid_df = result[result["valid"] == 1]
    meta = {
        "version": PROMPT_VERSION,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": str(args.model_path),
        "n_total": int(len(result)),
        "n_valid": int(len(valid_df)),
        "valid_rate": float(len(valid_df) / max(1, len(result))),
        "stance_dist": valid_df["stance"].value_counts().to_dict(),
        "target_dist": valid_df["target"].value_counts().to_dict(),
        "by_split": result.groupby("split")["valid"].agg(["size", "sum"]).to_dict(),
        "prompt_version": PROMPT_VERSION,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
    }
    (out_dir / "stance_label_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[done] {len(result)} rows, valid={len(valid_df)} "
        f"({meta['valid_rate']:.2%}) -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
