#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 true INDEPENDENT-CBM 生成 train split 的逐样本预测表。

输出 schema 与
outputs/independent_true_seed20261122/validation_predictions.csv 完全一致：

  id, toxic_true,
  pipeline_prob_toxic_f_of_g_x, pipeline_pred_toxic,
  oracle_prob_toxic_f_of_c, oracle_pred_toxic,
  toxic_type, expression,
  concept_target_A01, predicted_concept_A01, ...

只处理 train split（7,680 条），不读取 test。
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONCEPT_COLS = [
    "A01", "A02", "A03",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "C01", "C02", "C03", "C04",
    "D01", "D02",
    "E01", "E02", "E03", "E04", "E05",
    "T01", "T02", "T03", "T04", "T05",
]
TASK_CONCEPT_COLS = CONCEPT_COLS[:21]
EXPECTED_TRAIN_ROWS = 9600
EXPECTED_TRAIN_SUB = 7680
EXPECTED_VAL = 1920


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="生成 INDEPENDENT-CBM 的 train_predictions.csv"
    )
    p.add_argument(
        "--cbm_dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "INDEPENDENT_CBM_TrueIndependent_8_2_FinalTest_v1.0.0",
    )
    p.add_argument(
        "--model_path",
        type=Path,
        default=Path(r"D:\models\chinese-roberta-wwm-ext"),
    )
    p.add_argument(
        "--outputs_dir",
        type=Path,
        default=None,
        help="默认 <cbm_dir>/outputs/independent_true_seed20261122",
    )
    p.add_argument("--g_ckpt", type=Path, default=None)
    p.add_argument("--f_ckpt", type=Path, default=None)
    p.add_argument("--split_csv", type=Path, default=None)
    p.add_argument("--train_json", type=Path, default=None)
    p.add_argument("--matrix_csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 条 train 做 smoke；0 表示全部 7680",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="覆盖已存在的输出文件"
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定 cuda，但 CUDA 不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConceptPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.concept_head = nn.Linear(
            encoder.config.hidden_size, len(CONCEPT_COLS)
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.concept_head(h)


class TaskPredictor(nn.Module):
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


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.enc = tokenizer(
            self.df["content"].astype(str).tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        self.c = self.df[CONCEPT_COLS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "c": self.c[i],
            "y": int(row["toxic"]),
            "toxic_type": int(row["toxic_type"]),
            "expression": int(row["expression"]),
            "id": str(row["id"]),
        }


def make_collate(tokenizer):
    def fn(batch):
        toks = tokenizer.pad(
            [
                {
                    "input_ids": x["input_ids"],
                    "attention_mask": x["attention_mask"],
                }
                for x in batch
            ],
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "c": torch.tensor(
                np.stack([x["c"] for x in batch]), dtype=torch.float32
            ),
            "y": torch.tensor(
                [x["y"] for x in batch], dtype=torch.long
            ),
            "toxic_type": [x["toxic_type"] for x in batch],
            "expression": [x["expression"] for x in batch],
            "ids": [x["id"] for x in batch],
        }

    return fn


def load_data(args):
    cbm_dir = args.cbm_dir
    data_dir = cbm_dir / "data"
    outputs_dir = args.outputs_dir or (
        cbm_dir / "outputs" / "independent_true_seed20261122"
    )
    train_json = args.train_json or (data_dir / "train.json")
    matrix_csv = args.matrix_csv or (data_dir / "v8_train_matrix.csv")
    split_csv = args.split_csv or (outputs_dir / "train_validation_split.csv")
    # 两个 checkpoint 位于 outputs/ 根目录，不在 seed 子目录。
    g_ckpt = args.g_ckpt or (cbm_dir / "outputs" / "independent_g_best.pt")
    f_ckpt = args.f_ckpt or (cbm_dir / "outputs" / "independent_f_best.pt")
    out = args.out or (outputs_dir / "train_predictions.csv")

    for path in (train_json, matrix_csv, split_csv, g_ckpt, f_ckpt):
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件：{path}")

    raw = json.loads(train_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"train.json 必须是 {EXPECTED_TRAIN_ROWS} 行的 list"
        )
    df = pd.DataFrame(raw)
    df.insert(
        0, "id", [f"TOXICN_train_{i:04d}" for i in range(len(df))]
    )
    for col in ("toxic", "content", "toxic_type", "expression"):
        if col not in df.columns:
            raise ValueError(f"train.json 缺少列：{col}")

    matrix = pd.read_csv(matrix_csv, encoding="utf-8-sig")
    expected_cols = ["id", "toxic", "content"] + CONCEPT_COLS
    if list(matrix.columns) != expected_cols:
        raise ValueError("v8_train_matrix.csv 列名/顺序不符合预期")
    if len(matrix) != EXPECTED_TRAIN_ROWS:
        raise ValueError("v8_train_matrix.csv 行数不是 9600")
    if matrix["id"].astype(str).tolist() != df["id"].astype(str).tolist():
        raise ValueError("matrix 与 train.json 的 id 顺序不一致")
    if not np.array_equal(
        matrix["toxic"].to_numpy(dtype=int),
        df["toxic"].to_numpy(dtype=int),
    ):
        raise ValueError("matrix 与 train.json 的 toxic 不一致")
    if not np.array_equal(
        matrix["content"].astype(str).to_numpy(),
        df["content"].astype(str).to_numpy(),
    ):
        raise ValueError("matrix 与 train.json 的 content 不一致")

    split = pd.read_csv(split_csv, encoding="utf-8-sig")
    if not {"id", "split"}.issubset(split.columns):
        raise ValueError("split_csv 缺少 id/split")
    if len(split) != EXPECTED_TRAIN_ROWS:
        raise ValueError("split_csv 行数不是 9600")
    train_ids = split.loc[
        split["split"] == "train", "id"
    ].astype(str).tolist()
    val_ids = split.loc[
        split["split"] == "validation", "id"
    ].astype(str).tolist()
    if len(train_ids) != EXPECTED_TRAIN_SUB:
        raise ValueError(f"train split 不是 7680：{len(train_ids)}")
    if len(val_ids) != EXPECTED_VAL:
        raise ValueError(f"validation split 不是 1920：{len(val_ids)}")
    if set(train_ids) & set(val_ids):
        raise ValueError("train/validation id 有重叠")
    if set(train_ids) | set(val_ids) != set(df["id"].astype(str)):
        raise ValueError("split 与 train.json 的 id 集合不一致")

    df = df.merge(
        matrix[["id"] + CONCEPT_COLS], on="id", validate="one_to_one"
    )
    order = {sid: i for i, sid in enumerate(train_ids)}
    train_df = df[df["id"].astype(str).isin(order)].copy()
    train_df["__order"] = train_df["id"].astype(str).map(order)
    train_df = (
        train_df.sort_values("__order", kind="mergesort")
        .drop(columns="__order")
        .reset_index(drop=True)
    )
    if len(train_df) != EXPECTED_TRAIN_SUB:
        raise ValueError(f"train_df 行数不是 7680：{len(train_df)}")
    if train_df["content"].isna().any():
        raise ValueError("train_df 存在空 content")

    return train_df, out, g_ckpt, f_ckpt, {
        "train_json": train_json,
        "matrix_csv": matrix_csv,
        "split_csv": split_csv,
        "g_ckpt": g_ckpt,
        "f_ckpt": f_ckpt,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    train_df, out_path, g_ckpt, f_ckpt, inputs = load_data(args)

    if args.limit and args.limit > 0:
        train_df = train_df.head(args.limit).copy()
        print(
            f"[warn] --limit={args.limit}：只生成前 {len(train_df)} 条，"
            "不是完整 train_predictions.csv",
            flush=True,
        )
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出已存在：{out_path}；确认覆盖请加 --overwrite"
        )

    from transformers import AutoModel, AutoTokenizer

    if not args.model_path.is_dir():
        raise FileNotFoundError(f"backbone 目录不存在：{args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        str(args.model_path), local_files_only=True
    )

    g = ConceptPredictor(encoder)
    g_state = torch.load(g_ckpt, map_location="cpu", weights_only=False)
    if "state_dict" not in g_state:
        raise ValueError("g checkpoint 缺少 state_dict")
    g.load_state_dict(g_state["state_dict"])

    f = TaskPredictor()
    f.load_state_dict(
        torch.load(f_ckpt, map_location="cpu", weights_only=True)
    )

    g.to(device).eval()
    f.to(device).eval()
    if device.type == "cuda":
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        amp_dtype = None

    ds = TrainDataset(train_df, tokenizer, args.max_length)
    collate = make_collate(tokenizer)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    ids, ys, types, exprs = [], [], [], []
    preds, targets = [], []
    pipe_probs, oracle_probs = [], []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            c = batch["c"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(amp_dtype is not None),
            ):
                logits = g(inp, mask)
            pred = torch.sigmoid(logits.float())
            pipe_prob = torch.sigmoid(
                f(pred[:, : len(TASK_CONCEPT_COLS)])
            )
            oracle_prob = torch.sigmoid(
                f(c[:, : len(TASK_CONCEPT_COLS)])
            )
            preds.append(pred.cpu().numpy())
            targets.append(c.cpu().numpy())
            pipe_probs.append(pipe_prob.cpu().numpy())
            oracle_probs.append(oracle_prob.cpu().numpy())
            ids.extend(batch["ids"])
            ys.extend(batch["y"].numpy().tolist())
            types.extend(batch["toxic_type"])
            exprs.extend(batch["expression"])

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    pipe_probs = np.concatenate(pipe_probs, axis=0)
    oracle_probs = np.concatenate(oracle_probs, axis=0)
    y = np.asarray(ys, dtype=int)

    if len(ids) != len(train_df):
        raise RuntimeError(f"输出行数 {len(ids)} != {len(train_df)}")
    if not np.isfinite(preds).all() or preds.min() < 0 or preds.max() > 1:
        raise ValueError("predicted concepts 不是 [0,1] 的有限值")
    if not np.isfinite(targets).all() or targets.min() < 0 or targets.max() > 1:
        raise ValueError("teacher concepts 不是 [0,1] 的有限值")
    if not np.isfinite(pipe_probs).all() or not np.isfinite(oracle_probs).all():
        raise ValueError("pipeline/oracle 概率存在非有限值")

    out = pd.DataFrame(
        {
            "id": ids,
            "toxic_true": y,
            "pipeline_prob_toxic_f_of_g_x": pipe_probs,
            "pipeline_pred_toxic": (pipe_probs >= 0.5).astype(int),
            "oracle_prob_toxic_f_of_c": oracle_probs,
            "oracle_pred_toxic": (oracle_probs >= 0.5).astype(int),
            "toxic_type": types,
            "expression": exprs,
        }
    )
    for j, cid in enumerate(CONCEPT_COLS):
        out[f"concept_target_{cid}"] = targets[:, j]
        out[f"predicted_concept_{cid}"] = preds[:, j]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(out_path)

    from sklearn.metrics import f1_score, roc_auc_score

    pipe_pred = (pipe_probs >= 0.5).astype(int)
    oracle_pred = (oracle_probs >= 0.5).astype(int)
    pipe_f1 = float(
        f1_score(y, pipe_pred, average="macro", zero_division=0)
    )
    oracle_f1 = float(
        f1_score(y, oracle_pred, average="macro", zero_division=0)
    )
    pipe_auc = (
        float(roc_auc_score(y, pipe_probs))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    oracle_auc = (
        float(roc_auc_score(y, oracle_probs))
        if len(np.unique(y)) == 2
        else float("nan")
    )

    eps = 1e-6
    p = np.clip(preds, eps, 1 - eps)
    t = np.clip(targets, eps, 1 - eps)
    concept_mae = np.mean(np.abs(p - t), axis=0)
    concept_bce = np.mean(
        -(t * np.log(p) + (1 - t) * np.log(1 - p)), axis=0
    )

    lines = []
    lines.append("INDEPENDENT-CBM train split 预测表")
    lines.append("=" * 64)
    lines.append(f"时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}")
    lines.append(f"device：{device}")
    lines.append(f"backbone：{args.model_path}")
    lines.append(f"train rows：{len(out)}")
    lines.append(f"toxic rate：{y.mean():.4f}")
    lines.append(
        f"pipeline f(g(x))：MacroF1={pipe_f1:.4f} AUC={pipe_auc:.4f}"
        "（train in-sample）"
    )
    lines.append(
        f"oracle f(c)：MacroF1={oracle_f1:.4f} AUC={oracle_auc:.4f}"
        "（train in-sample）"
    )
    lines.append("")
    lines.append("概念蒸馏质量（vs LLM soft target）")
    lines.append("-" * 64)
    for j, cid in enumerate(CONCEPT_COLS):
        lines.append(
            f"{cid}: MAE={concept_mae[j]:.4f} BCE={concept_bce[j]:.4f}"
        )
    lines.append("")
    lines.append("输出")
    lines.append("-" * 64)
    lines.append(str(out_path))
    report = "\n".join(lines) + "\n"
    (out_path.parent / "train_predictions_report.txt").write_text(
        report, encoding="utf-8"
    )
    meta = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "backbone": str(args.model_path),
        "n_rows": int(len(out)),
        "toxic_rate": float(y.mean()),
        "pipeline_macro_f1_train_in_sample": pipe_f1,
        "pipeline_auc_train_in_sample": pipe_auc,
        "oracle_macro_f1_train_in_sample": oracle_f1,
        "oracle_auc_train_in_sample": oracle_auc,
        "inputs": {k: str(v) for k, v in inputs.items()},
        "input_sha256": {k: sha256(v) for k, v in inputs.items()},
        "output": str(out_path),
        "output_sha256": sha256(out_path),
    }
    (out_path.parent / "train_predictions_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(report, flush=True)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
