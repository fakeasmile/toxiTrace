#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLM + concept bottleneck：测量缺口 / sufficiency / leakage 诊断。

研究主张
--------
CBM 在中文毒性检测上的性能缺口主要是概念测量/充分性问题，而不是概念决策问题。
本脚本在同一 split、同一协议下比较四种模型：

  1. plm_only        raw text -> PLM -> binary/regime heads（约 0.82 锚点）
  2. llm_cbm         34 维 LLM 概念分数 -> heads（约 0.79 锚点）
  3. plm_cbm_strict  PLM -> 34 concept heads -> heads（只看概念）
  4. plm_cbm_hybrid  PLM -> 34 concept heads + h_cls -> heads（概念 + 文本残差）

关键量
------
- sufficiency gap:
    llm_cbm -> plm_cbm_strict  概念测量改进
    strict  -> hybrid          文本残差收益
    hybrid  -> plm_only        剩余瓶颈代价
- leakage（概念瓶颈是否真在起作用）：
    normal / zero_concepts / shuffle_concepts 三种测试
    hybrid 在 zero/shuffle 下仍然很高 => 残差路径泄漏标签
- block ablation：A/B/C/D/E/T/N 各概念区块置 0 后的 Macro-F1 变化
- regime operating point：val 选 hate recall>=90% 的阈值，test 报告 FP/FN/F1

用法
----
# 正式（BERT + RoBERTa）
python scripts/train_plm_concept_cbm.py

# 只跑 RoBERTa，3 seeds
python scripts/train_plm_concept_cbm.py --backbones chinese-roberta-wwm-ext

# 冒烟（tiny BERT + 每 regime 20 条 + 1 epoch）
python scripts/train_plm_concept_cbm.py --backbones bert-tiny-chinese \
  --limit 20 --epochs 1 --llm_epochs 5 --batch_size 8 --max_len 64 \
  --out_dir results/plm_concept_cbm_smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.n = 0

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, n: int = 1):
            self.n += n

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_regime_aware_cbm as base  # noqa: E402


PROJECT_ROOT = base.PROJECT_ROOT
OUT_ROOT = PROJECT_ROOT / "results" / "plm_concept_cbm"
MODEL_ROOT = Path(r"d:\models")
DEFAULT_BACKBONES = ["bert-base-chinese", "chinese-roberta-wwm-ext"]
SEED_DEFAULT = [729, 1136, 1196]
ALL_MODELS = [
    "llm_cbm",
    "plm_only",
    "plm_cbm_strict",
    "plm_cbm_strict_detached",
    "plm_cbm_hybrid",
]
REUSE_FILES = {
    "metrics": "metrics_by_run.csv",
    "strata": "regime_strata.csv",
    "concept": "concept_metrics.csv",
    "leakage": "leakage.csv",
    "block": "block_ablation.csv",
    "op": "operating_point.csv",
}


def load_reuse_rows(
    reuse_dir: Path, retrain_keys: set
) -> Dict[str, List[Dict[str, object]]]:
    """读取上一次结果，并剔除本次要重训的 (backbone, model) 行。"""
    out: Dict[str, List[Dict[str, object]]] = {}
    for key, fname in REUSE_FILES.items():
        path = reuse_dir / fname
        if not path.is_file():
            out[key] = []
            continue
        df = pd.read_csv(path)
        if "backbone" in df.columns and "model" in df.columns:
            keep = [
                (str(b), str(m)) not in retrain_keys
                for b, m in zip(df["backbone"], df["model"])
            ]
            df = df[pd.Series(keep, index=df.index)]
        out[key] = df.to_dict("records")
    return out


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------

def resolve_backbone(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    return MODEL_ROOT / name


def block_indices(concept_cols: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for name, cols in base.BLOCK_DEFS.items():
        idxs = [concept_cols.index(c) for c in cols if c in concept_cols]
        if idxs:
            out[name] = idxs
    return out


def select_hate_threshold(
    y_bin: np.ndarray,
    prob: np.ndarray,
    regime: np.ndarray,
    target: float = 0.90,
) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    prob = np.asarray(prob, dtype=float)
    regime = np.asarray(regime)
    best_key = None
    best_t = 0.5
    for t in [round(0.02 + 0.01 * i, 2) for i in range(97)]:
        pred = (prob >= t).astype(int)
        hate_recall = float(pred[regime == 1].mean()) if (regime == 1).any() else float("nan")
        if not math.isfinite(hate_recall) or hate_recall < target:
            continue
        fp = float(pred[regime == 0].mean()) if (regime == 0).any() else float("nan")
        f1 = base.macro_f1(y_bin, pred)
        key = (fp if math.isfinite(fp) else 1e9, -f1)
        if best_key is None or key < best_key:
            best_key = key
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

class PLMDataset(Dataset):
    def __init__(
        self,
        enc,
        idx: Sequence[int],
        y_bin: np.ndarray,
        y3: np.ndarray,
        concepts: np.ndarray,
    ):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        self.input_ids = enc["input_ids"][idx_t]
        self.attention_mask = enc["attention_mask"][idx_t]
        self.token_type_ids = (
            enc["token_type_ids"][idx_t] if "token_type_ids" in enc else None
        )
        self.y_bin = torch.tensor(np.asarray(y_bin)[idx], dtype=torch.float32)
        self.y3 = torch.tensor(np.asarray(y3)[idx], dtype=torch.long)
        self.concepts = torch.tensor(np.asarray(concepts)[idx], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_bin)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "y_bin": self.y_bin[i],
            "y3": self.y3[i],
            "concept_targets": self.concepts[i],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[i]
        return item


class ConceptDataset(Dataset):
    def __init__(self, concepts: np.ndarray, y_bin: np.ndarray, y3: np.ndarray):
        self.concepts = torch.tensor(np.asarray(concepts), dtype=torch.float32)
        self.y_bin = torch.tensor(np.asarray(y_bin), dtype=torch.float32)
        self.y3 = torch.tensor(np.asarray(y3), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y_bin)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "concept_input": self.concepts[i],
            "y_bin": self.y_bin[i],
            "y3": self.y3[i],
        }


def build_plm_loaders(
    backbone_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    y_bin_train: np.ndarray,
    y3_train: np.ndarray,
    C_train: np.ndarray,
    y_bin_test: np.ndarray,
    y3_test: np.ndarray,
    C_test: np.ndarray,
    args,
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone_path), local_files_only=True
    )
    enc_tr = tokenizer(
        train_df["content"].astype(str).tolist(),
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_tensors="pt",
    )
    enc_te = tokenizer(
        test_df["content"].astype(str).tolist(),
        truncation=True,
        max_length=args.max_len,
        padding="max_length",
        return_tensors="pt",
    )
    ds_tr = PLMDataset(enc_tr, tr_idx, y_bin_train, y3_train, C_train)
    ds_val = PLMDataset(enc_tr, val_idx, y_bin_train, y3_train, C_train)
    ds_te = PLMDataset(
        enc_te, np.arange(len(test_df)), y_bin_test, y3_test, C_test
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    return tokenizer, train_loader, val_loader, test_loader


def build_concept_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    y_bin_train: np.ndarray,
    y3_train: np.ndarray,
    C_train: np.ndarray,
    y_bin_test: np.ndarray,
    y3_test: np.ndarray,
    C_test: np.ndarray,
    args,
):
    ds_tr = ConceptDataset(C_train[tr_idx], y_bin_train[tr_idx], y3_train[tr_idx])
    ds_val = ConceptDataset(C_train[val_idx], y_bin_train[val_idx], y3_train[val_idx])
    ds_te = ConceptDataset(C_test, y_bin_test, y3_test)
    train_loader = DataLoader(
        ds_tr, batch_size=args.llm_batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.llm_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ds_te, batch_size=args.llm_batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class PLMEncoder(nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.hidden = int(self.encoder.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return out.last_hidden_state[:, 0, :]


class PLMOnly(nn.Module):
    def __init__(self, model_path: Path, dropout: float = 0.1):
        super().__init__()
        self.encoder = PLMEncoder(model_path)
        self.drop = nn.Dropout(dropout)
        self.head_bin = nn.Linear(self.encoder.hidden, 1)
        self.head_reg = nn.Linear(self.encoder.hidden, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        h = self.encoder(input_ids, attention_mask, token_type_ids)
        z = self.drop(h)
        return self.head_bin(z), self.head_reg(z), None, None


class PLMConceptCBM(nn.Module):
    def __init__(
        self,
        model_path: Path,
        n_concepts: int,
        mode: str = "strict",
        dropout: float = 0.1,
        detach_concepts: bool = False,
    ):
        super().__init__()
        assert mode in {"strict", "hybrid"}
        self.mode = mode
        # strict + detach：概念头只受 LLM concept loss 驱动，
        # 分类梯度不回流到概念头/encoder，避免概念被标签重塑。
        self.detach_concepts = bool(detach_concepts) and mode == "strict"
        self.encoder = PLMEncoder(model_path)
        self.concept_head = nn.Linear(self.encoder.hidden, n_concepts)
        in_dim = n_concepts if mode == "strict" else n_concepts + self.encoder.hidden
        self.head_bin = nn.Linear(in_dim, 1)
        self.head_reg = nn.Linear(in_dim, 3)
        self.drop = nn.Dropout(dropout)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        h = self.encoder(input_ids, attention_mask, token_type_ids)
        c_logits = self.concept_head(h)
        c = torch.sigmoid(c_logits)
        return h, c_logits, c

    def heads(self, h: torch.Tensor, c: torch.Tensor):
        # hybrid 训练时做 modality dropout，防止模型完全忽略概念或残差。
        if self.training and self.mode == "hybrid":
            if torch.rand(1, device=h.device).item() < 0.10:
                c = torch.zeros_like(c)
            if torch.rand(1, device=h.device).item() < 0.10:
                h = torch.zeros_like(h)
        c_for_heads = c.detach() if self.detach_concepts else c
        z = c_for_heads if self.mode == "strict" else torch.cat([c_for_heads, h], dim=1)
        z = self.drop(z)
        return self.head_bin(z), self.head_reg(z)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        h, c_logits, c = self.encode(input_ids, attention_mask, token_type_ids)
        logits_bin, logits_reg = self.heads(h, c)
        return logits_bin, logits_reg, c_logits, c


class LLMConceptCBM(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head_bin = nn.Linear(d, 1)
        self.head_reg = nn.Linear(d, 3)

    def forward(self, concept_input: torch.Tensor):
        z = self.drop(concept_input)
        return self.head_bin(z), self.head_reg(z), None, concept_input


# ---------------------------------------------------------------------------
# 训练 / 推理
# ---------------------------------------------------------------------------

def forward_batch(model, batch: Dict[str, torch.Tensor], device: torch.device, is_plm: bool):
    if is_plm:
        kwargs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(
                device, non_blocking=True
            )
        return model(**kwargs)
    return model(batch["concept_input"].to(device, non_blocking=True))


def compute_loss(
    outputs,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    regime_weights: torch.Tensor,
    regime_weight: float,
    concept_weight: float,
) -> torch.Tensor:
    logits_bin, logits_reg, c_logits, _ = outputs
    y_bin = batch["y_bin"].to(device, non_blocking=True)
    y3 = batch["y3"].to(device, non_blocking=True)
    loss = F.binary_cross_entropy_with_logits(logits_bin.squeeze(-1), y_bin)
    loss = loss + regime_weight * F.cross_entropy(
        logits_reg, y3, weight=regime_weights
    )
    if c_logits is not None:
        loss = loss + concept_weight * F.binary_cross_entropy_with_logits(
            c_logits, batch["concept_targets"].to(device, non_blocking=True)
        )
    return loss


def evaluate(model, loader, device: torch.device, is_plm: bool) -> Dict[str, np.ndarray]:
    model.eval()
    probs_bin, probs_reg, cs = [], [], []
    y_bin, y3, C = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = forward_batch(model, batch, device, is_plm)
            logits_bin, logits_reg, _, c = outputs
            probs_bin.append(torch.sigmoid(logits_bin.squeeze(-1)).cpu().numpy())
            probs_reg.append(torch.softmax(logits_reg, dim=1).cpu().numpy())
            if c is not None:
                cs.append(c.cpu().numpy())
            y_bin.append(batch["y_bin"].numpy())
            y3.append(batch["y3"].numpy())
            if "concept_targets" in batch:
                C.append(batch["concept_targets"].numpy())
    return {
        "prob_bin": np.concatenate(probs_bin),
        "prob_reg": np.concatenate(probs_reg),
        "c": np.concatenate(cs) if cs else None,
        "y_bin": np.concatenate(y_bin),
        "y3": np.concatenate(y3),
        "C": np.concatenate(C) if C else None,
    }


def train_plm_model(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    args,
    regime_weights: torch.Tensor,
    seed: int,
    use_fp16: bool,
    desc: str = "",
) -> Tuple[object, int, float]:
    base.set_seed(seed)
    backbone_params = list(model.encoder.parameters())
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.")
    ]
    opt = AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(
        opt, int(0.1 * total_steps), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad = 0

    pbar = tqdm(
        total=total_steps,
        desc=desc or "plm_train",
        leave=False,
        dynamic_ncols=True,
    )
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = forward_batch(model, batch, device, is_plm=True)
                loss = compute_loss(
                    outputs,
                    batch,
                    device,
                    regime_weights,
                    args.regime_weight,
                    args.concept_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            pbar.update(1)
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", ep=f"{epoch}/{args.epochs}"
                )

        val = evaluate(model, val_loader, device, is_plm=True)
        val_f1 = base.macro_f1(
            val["y_bin"], (val["prob_bin"] >= 0.5).astype(int)
        )
        print(
            f"[plm] {desc} epoch {epoch}/{args.epochs} "
            f"loss={loss.item():.4f} val_macroF1={val_f1:.4f}"
        )
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val


def train_llm_model(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    args,
    regime_weights: torch.Tensor,
    seed: int,
    desc: str = "",
) -> Tuple[object, int, float]:
    base.set_seed(seed)
    opt = AdamW(model.parameters(), lr=args.head_lr, weight_decay=1e-4)
    total_steps = max(1, args.llm_epochs * len(train_loader))
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad = 0

    pbar = tqdm(
        total=total_steps,
        desc=desc or "llm_cbm",
        leave=False,
        dynamic_ncols=True,
    )
    step = 0
    for epoch in range(1, args.llm_epochs + 1):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            outputs = forward_batch(model, batch, device, is_plm=False)
            loss = compute_loss(
                outputs,
                batch,
                device,
                regime_weights,
                args.regime_weight,
                args.concept_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            step += 1
            pbar.update(1)
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", ep=f"{epoch}/{args.llm_epochs}"
                )

        val = evaluate(model, val_loader, device, is_plm=False)
        val_f1 = base.macro_f1(
            val["y_bin"], (val["prob_bin"] >= 0.5).astype(int)
        )
        improved = val_f1 > best_val + 1e-6
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"[llm] {desc} epoch {epoch}/{args.llm_epochs} "
                f"loss={loss.item():.4f} val_macroF1={val_f1:.4f}"
            )
        if improved:
            best_val = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.llm_patience:
                break

    pbar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val


def encode_cbm(model: PLMConceptCBM, loader, device: torch.device):
    model.eval()
    hs, cs, y_bin, y3, C = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = {
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(
                    device, non_blocking=True
                ),
            }
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(
                    device, non_blocking=True
                )
            h, _, c = model.encode(**kwargs)
            hs.append(h.cpu().numpy())
            cs.append(c.cpu().numpy())
            y_bin.append(batch["y_bin"].numpy())
            y3.append(batch["y3"].numpy())
            C.append(batch["concept_targets"].numpy())
    return (
        np.concatenate(hs),
        np.concatenate(cs),
        np.concatenate(y_bin),
        np.concatenate(y3),
        np.concatenate(C),
    )


def heads_prob_cbm(
    model: PLMConceptCBM,
    h: np.ndarray,
    c: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
):
    model.eval()
    pb, pr = [], []
    with torch.no_grad():
        for start in range(0, len(h), batch_size):
            hb = torch.tensor(h[start : start + batch_size], device=device)
            cb = torch.tensor(c[start : start + batch_size], device=device)
            logits_bin, logits_reg = model.heads(hb, cb)
            pb.append(torch.sigmoid(logits_bin.squeeze(-1)).cpu().numpy())
            pr.append(torch.softmax(logits_reg, dim=1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pr)


def heads_prob_llm(
    model: LLMConceptCBM,
    c: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
):
    model.eval()
    pb, pr = [], []
    with torch.no_grad():
        for start in range(0, len(c), batch_size):
            cb = torch.tensor(c[start : start + batch_size], device=device)
            logits_bin, logits_reg, _, _ = model(cb)
            pb.append(torch.sigmoid(logits_bin.squeeze(-1)).cpu().numpy())
            pr.append(torch.softmax(logits_reg, dim=1).cpu().numpy())
    return np.concatenate(pb), np.concatenate(pr)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

def concept_quality_rows(
    backbone: str,
    model_name: str,
    seed: int,
    c_prob: np.ndarray,
    C_target: np.ndarray,
    concept_cols: List[str],
) -> List[Dict[str, object]]:
    rows = []
    eps = 1e-6
    for j, cname in enumerate(concept_cols):
        t = np.asarray(C_target[:, j], dtype=float)
        p = np.asarray(c_prob[:, j], dtype=float)
        y = (t >= 0.5).astype(int)
        auc = base.auc_or_nan(y, p)
        if np.std(t) > 1e-9 and np.std(p) > 1e-9:
            corr = float(np.corrcoef(t, p)[0, 1])
        else:
            corr = float("nan")
        bce = float(-np.mean(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)))
        rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "concept": cname,
                "auc": auc,
                "corr": corr,
                "bce": bce,
                "target_mean": float(np.mean(t)),
                "pred_mean": float(np.mean(p)),
            }
        )
    return rows


def leakage_rows_plm(
    backbone: str,
    model_name: str,
    seed: int,
    model: PLMConceptCBM,
    test_loader,
    device: torch.device,
    y_bin_test: np.ndarray,
    regime_test: np.ndarray,
) -> List[Dict[str, object]]:
    h, c, _, _, _ = encode_cbm(model, test_loader, device)
    normal, _ = heads_prob_cbm(model, h, c, device)
    zero, _ = heads_prob_cbm(model, h, np.zeros_like(c), device)
    perm = np.random.default_rng(0).permutation(len(c))
    shuffled, _ = heads_prob_cbm(model, h, c[perm], device)
    rows = []
    for mode, prob in [("normal", normal), ("zero_concepts", zero), ("shuffle_concepts", shuffled)]:
        m = base.binary_metrics(y_bin_test, prob, 0.5)
        rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "mode": mode,
                "macro_f1": m["macro_f1"],
                "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                "hate_fn_rate": float(np.mean(prob[regime_test == 1] < 0.5)),
                "offensive_fn_rate": float(np.mean(prob[regime_test == 2] < 0.5)),
            }
        )
    return rows


def leakage_rows_llm(
    backbone: str,
    model_name: str,
    seed: int,
    model: LLMConceptCBM,
    C_test: np.ndarray,
    device: torch.device,
    y_bin_test: np.ndarray,
    regime_test: np.ndarray,
) -> List[Dict[str, object]]:
    normal, _ = heads_prob_llm(model, C_test, device)
    zero, _ = heads_prob_llm(model, np.zeros_like(C_test), device)
    perm = np.random.default_rng(0).permutation(len(C_test))
    shuffled, _ = heads_prob_llm(model, C_test[perm], device)
    rows = []
    for mode, prob in [("normal", normal), ("zero_concepts", zero), ("shuffle_concepts", shuffled)]:
        m = base.binary_metrics(y_bin_test, prob, 0.5)
        rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "mode": mode,
                "macro_f1": m["macro_f1"],
                "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                "hate_fn_rate": float(np.mean(prob[regime_test == 1] < 0.5)),
                "offensive_fn_rate": float(np.mean(prob[regime_test == 2] < 0.5)),
            }
        )
    return rows


def block_rows_plm(
    backbone: str,
    model_name: str,
    seed: int,
    model: PLMConceptCBM,
    test_loader,
    device: torch.device,
    y_bin_test: np.ndarray,
    regime_test: np.ndarray,
    concept_cols: List[str],
) -> List[Dict[str, object]]:
    h, c, _, _, _ = encode_cbm(model, test_loader, device)
    normal, _ = heads_prob_cbm(model, h, c, device)
    base_m = base.binary_metrics(y_bin_test, normal, 0.5)
    blocks = block_indices(concept_cols)
    rows = []
    for block, idxs in blocks.items():
        c_abl = c.copy()
        c_abl[:, idxs] = 0.0
        prob, _ = heads_prob_cbm(model, h, c_abl, device)
        m = base.binary_metrics(y_bin_test, prob, 0.5)
        rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "block": block,
                "macro_f1": m["macro_f1"],
                "delta_macro_f1": m["macro_f1"] - base_m["macro_f1"],
                "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                "hate_fn_rate": float(np.mean(prob[regime_test == 1] < 0.5)),
                "offensive_fn_rate": float(np.mean(prob[regime_test == 2] < 0.5)),
            }
        )
    return rows


def block_rows_llm(
    backbone: str,
    model_name: str,
    seed: int,
    model: LLMConceptCBM,
    C_test: np.ndarray,
    device: torch.device,
    y_bin_test: np.ndarray,
    regime_test: np.ndarray,
    concept_cols: List[str],
) -> List[Dict[str, object]]:
    normal, _ = heads_prob_llm(model, C_test, device)
    base_m = base.binary_metrics(y_bin_test, normal, 0.5)
    blocks = block_indices(concept_cols)
    rows = []
    for block, idxs in blocks.items():
        c_abl = C_test.copy()
        c_abl[:, idxs] = 0.0
        prob, _ = heads_prob_llm(model, c_abl, device)
        m = base.binary_metrics(y_bin_test, prob, 0.5)
        rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "block": block,
                "macro_f1": m["macro_f1"],
                "delta_macro_f1": m["macro_f1"] - base_m["macro_f1"],
                "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                "hate_fn_rate": float(np.mean(prob[regime_test == 1] < 0.5)),
                "offensive_fn_rate": float(np.mean(prob[regime_test == 2] < 0.5)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", type=str, default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEED_DEFAULT))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--llm_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--llm_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--llm_batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--regime_weight", type=float, default=0.5)
    parser.add_argument("--concept_weight", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODELS),
        help="只训练指定模型，逗号分隔；可配合 --reuse_dir 复用其他模型结果",
    )
    parser.add_argument(
        "--reuse_dir",
        type=str,
        default="",
        help="复用上一次结果目录中的 metrics/leakage/block/concept/op CSV",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅用于冒烟：每个 regime 最多取 N 条")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown_models = [m for m in selected_models if m not in ALL_MODELS]
    if unknown_models:
        raise ValueError(
            f"未知模型：{unknown_models}；可选：{ALL_MODELS}"
        )
    if not selected_models:
        raise ValueError("--models 不能为空。")
    run_backbones = [b for b in backbones if resolve_backbone(b).is_dir()]
    missing_backbones = [b for b in backbones if b not in run_backbones]
    if missing_backbones:
        print(f"[warn] 以下 backbone 不存在，将跳过或不重训：{missing_backbones}")
    if not run_backbones and not args.reuse_dir:
        raise FileNotFoundError("没有可用的 backbone，且未提供 --reuse_dir。")

    split_seed = int(args.split_seed) if args.split_seed else seeds[0]
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_fp16 = device.type == "cuda" and not args.no_fp16
    print(
        f"[info] device={device}，fp16={use_fp16}，backbones={backbones}，"
        f"seeds={seeds}，split_seed={split_seed}，models={selected_models}"
    )

    train_df, concept_cols = base.load_split("train")
    test_df, test_concepts = base.load_split("test")
    if concept_cols != test_concepts:
        raise ValueError("train/test 概念列顺序不一致。")

    if args.limit:
        train_df = (
            train_df.groupby("regime", group_keys=False)
            .head(args.limit)
            .reset_index(drop=True)
        )
        test_df = (
            test_df.groupby("regime", group_keys=False)
            .head(args.limit)
            .reset_index(drop=True)
        )
        print(f"[info] smoke limit={args.limit}，train={len(train_df)}，test={len(test_df)}")

    tr_idx, val_idx = base.stratified_split_by_regime(
        train_df, seed=split_seed, val_ratio=args.val_ratio
    )
    y_bin_train = train_df["toxic"].to_numpy(dtype=int)
    y3_train = train_df["regime"].to_numpy(dtype=int)
    y_bin_test = test_df["toxic"].to_numpy(dtype=int)
    y3_test = test_df["regime"].to_numpy(dtype=int)
    C_train = train_df[concept_cols].to_numpy(dtype=np.float32)
    C_test = test_df[concept_cols].to_numpy(dtype=np.float32)
    regime_test = test_df["regime"].to_numpy(dtype=int)
    regime_val = y3_train[val_idx]

    counts = np.bincount(y3_train[tr_idx], minlength=3).astype(float)
    weights_sqrt = np.sqrt(len(tr_idx) / np.maximum(counts, 1.0))
    regime_weights = torch.tensor(weights_sqrt, dtype=torch.float32, device=device)
    print(
        f"[info] train regime={ {base.REGIME_NAMES[i]: int(counts[i]) for i in range(3)} }；"
        f"sqrt weights={np.round(weights_sqrt, 3).tolist()}"
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录：{out_dir}")

    retrain_keys = set()
    if "llm_cbm" in selected_models:
        retrain_keys.add(("none", "llm_cbm"))
    for bb in run_backbones:
        for model_name in selected_models:
            if model_name != "llm_cbm":
                retrain_keys.add((bb, model_name))

    reuse_rows: Dict[str, List[Dict[str, object]]] = {
        key: [] for key in REUSE_FILES
    }
    if args.reuse_dir:
        reuse_dir = Path(args.reuse_dir)
        if not reuse_dir.is_dir():
            raise FileNotFoundError(f"reuse_dir 不存在：{reuse_dir}")
        reuse_rows = load_reuse_rows(reuse_dir, retrain_keys)
        print(
            "[info] reuse_dir="
            f"{reuse_dir}，"
            + "，".join(f"{k}={len(v)}" for k, v in reuse_rows.items())
        )
        if reuse_dir.resolve() == out_dir.resolve():
            print("[warn] reuse_dir == out_dir；本次将合并后覆盖写回同一目录。")

    overview = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_concepts": int(len(concept_cols)),
        "concepts": concept_cols,
        "train_regime_counts": {
            base.REGIME_NAMES[k]: int(v)
            for k, v in train_df["regime"].value_counts().sort_index().items()
        },
        "test_regime_counts": {
            base.REGIME_NAMES[k]: int(v)
            for k, v in test_df["regime"].value_counts().sort_index().items()
        },
        "backbones": backbones,
        "run_backbones": run_backbones,
        "models": selected_models,
        "reuse_dir": str(args.reuse_dir) if args.reuse_dir else None,
        "seeds": seeds,
        "split_seed": split_seed,
        "val_ratio": args.val_ratio,
        "class_weight_sqrt": {
            base.REGIME_NAMES[i]: float(weights_sqrt[i]) for i in range(3)
        },
        "hyper": {
            "epochs": args.epochs,
            "llm_epochs": args.llm_epochs,
            "batch_size": args.batch_size,
            "llm_batch_size": args.llm_batch_size,
            "max_len": args.max_len,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "regime_weight": args.regime_weight,
            "concept_weight": args.concept_weight,
        },
    }
    with (out_dir / "overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    # 记录容器（可从上一次结果复用未重训模型）
    metrics_rows: List[Dict[str, object]] = list(reuse_rows["metrics"])
    strata_rows: List[Dict[str, object]] = list(reuse_rows["strata"])
    concept_rows: List[Dict[str, object]] = list(reuse_rows["concept"])
    leakage_rows: List[Dict[str, object]] = list(reuse_rows["leakage"])
    block_rows: List[Dict[str, object]] = list(reuse_rows["block"])
    op_rows: List[Dict[str, object]] = list(reuse_rows["op"])

    strata = base.strata_masks(test_df)

    def record_standard(
        backbone: str,
        model_name: str,
        seed: int,
        val_out: Dict[str, np.ndarray],
        test_out: Dict[str, np.ndarray],
        best_epoch: int,
    ) -> None:
        scores = [
            ("bin", np.asarray(val_out["prob_bin"]), np.asarray(test_out["prob_bin"])),
            (
                "reg",
                1.0 - np.asarray(val_out["prob_reg"])[:, 0],
                1.0 - np.asarray(test_out["prob_reg"])[:, 0],
            ),
        ]
        for score_type, val_prob, test_prob in scores:
            t_opt = base.best_threshold(val_out["y_bin"], val_prob)
            for policy, t in [("fixed_0.5", 0.5), ("val_opt", t_opt)]:
                m = base.binary_metrics(y_bin_test, test_prob, t)
                metrics_rows.append(
                    {
                        "backbone": backbone,
                        "model": model_name,
                        "seed": seed,
                        "score_type": score_type,
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
                        "hate_fn_rate": float(np.mean(test_prob[regime_test == 1] < t)),
                        "offensive_fn_rate": float(np.mean(test_prob[regime_test == 2] < t)),
                        "best_epoch": best_epoch,
                    }
                )
                if score_type == "bin":
                    for stratum_name in base.STRATA:
                        mask = strata[stratum_name]
                        if mask.sum() == 0:
                            continue
                        sm = base.stratum_metrics(y_bin_test, test_prob, t, mask)
                        strata_rows.append(
                            {
                                "backbone": backbone,
                                "model": model_name,
                                "seed": seed,
                                "threshold_policy": policy,
                                "stratum": stratum_name,
                                **sm,
                            }
                        )

        # operating point: val 选 hate recall>=90%，test 报告
        t_h = select_hate_threshold(
            val_out["y_bin"], val_out["prob_bin"], regime_val, 0.90
        )
        m = base.binary_metrics(y_bin_test, test_out["prob_bin"], t_h)
        op_rows.append(
            {
                "backbone": backbone,
                "model": model_name,
                "seed": seed,
                "threshold": t_h,
                "macro_f1": m["macro_f1"],
                "pos_f1": m["pos_f1"],
                "auc": m["auc"],
                "fp_rate_non_toxic": m["fp_rate_non_toxic"],
                "hate_fn_rate": float(np.mean(test_out["prob_bin"][regime_test == 1] < t_h)),
                "offensive_fn_rate": float(np.mean(test_out["prob_bin"][regime_test == 2] < t_h)),
            }
        )

    # ---------------- LLM-concept CBM（无 PLM） ----------------
    llm_seeds = seeds if "llm_cbm" in selected_models else []
    if llm_seeds:
        print("[info] === LLM-concept CBM ===")
    else:
        print("[info] skip llm_cbm（不在 --models 中）")
    llm_tr, llm_val, llm_te = build_concept_loaders(
        train_df, test_df, tr_idx, val_idx,
        y_bin_train, y3_train, C_train,
        y_bin_test, y3_test, C_test, args,
    )
    for seed in tqdm(llm_seeds, desc="llm_cbm", leave=False):
        model = LLMConceptCBM(len(concept_cols)).to(device)
        model, best_epoch, _ = train_llm_model(
            model, llm_tr, llm_val, device, args, regime_weights, seed,
            desc=f"llm_cbm/s{seed}",
        )
        val_out = evaluate(model, llm_val, device, is_plm=False)
        test_out = evaluate(model, llm_te, device, is_plm=False)
        record_standard("none", "llm_cbm", seed, val_out, test_out, best_epoch)
        leakage_rows.extend(
            leakage_rows_llm(
                "none", "llm_cbm", seed, model, C_test, device,
                y_bin_test, regime_test,
            )
        )
        block_rows.extend(
            block_rows_llm(
                "none", "llm_cbm", seed, model, C_test, device,
                y_bin_test, regime_test, concept_cols,
            )
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------- PLM 模型 ----------------
    for backbone_name in tqdm(run_backbones, desc="backbones", leave=False):
        backbone_path = resolve_backbone(backbone_name)
        if not backbone_path.is_dir():
            print(f"[warn] 跳过不存在的 backbone：{backbone_path}")
            continue
        print(f"[info] === backbone={backbone_name} ({backbone_path}) ===")
        tokenizer, plm_tr, plm_val, plm_te = build_plm_loaders(
            backbone_path, train_df, test_df, tr_idx, val_idx,
            y_bin_train, y3_train, C_train,
            y_bin_test, y3_test, C_test, args,
        )

        for model_name, mode, detach in [
            ("plm_only", None, False),
            ("plm_cbm_strict", "strict", False),
            ("plm_cbm_strict_detached", "strict", True),
            ("plm_cbm_hybrid", "hybrid", False),
        ]:
            if model_name not in selected_models:
                continue
            for seed in tqdm(seeds, desc=model_name, leave=False):
                if mode is None:
                    model = PLMOnly(backbone_path)
                else:
                    model = PLMConceptCBM(
                        backbone_path,
                        len(concept_cols),
                        mode=mode,
                        detach_concepts=detach,
                    )
                model = model.to(device)
                model, best_epoch, _ = train_plm_model(
                    model, plm_tr, plm_val, device, args,
                    regime_weights, seed, use_fp16,
                    desc=f"{backbone_name}/{model_name}/s{seed}",
                )
                val_out = evaluate(model, plm_val, device, is_plm=True)
                test_out = evaluate(model, plm_te, device, is_plm=True)
                record_standard(
                    backbone_name, model_name, seed, val_out, test_out, best_epoch
                )

                if mode is not None:
                    # concept distillation quality
                    if test_out["c"] is not None:
                        concept_rows.extend(
                            concept_quality_rows(
                                backbone_name, model_name, seed,
                                test_out["c"], C_test, concept_cols,
                            )
                        )
                    # leakage / block ablation
                    leakage_rows.extend(
                        leakage_rows_plm(
                            backbone_name, model_name, seed, model,
                            plm_te, device, y_bin_test, regime_test,
                        )
                    )
                    block_rows.extend(
                        block_rows_plm(
                            backbone_name, model_name, seed, model,
                            plm_te, device, y_bin_test, regime_test,
                            concept_cols,
                        )
                    )

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "backbone", "model", "seed", "score_type", "threshold_policy",
            "threshold", "acc", "macro_f1", "pos_f1", "precision", "recall",
            "auc", "fp_rate_non_toxic", "fn_rate_toxic",
            "hate_fn_rate", "offensive_fn_rate", "best_epoch",
        ],
    )
    metrics_df.to_csv(out_dir / "metrics_by_run.csv", index=False, encoding="utf-8-sig")

    report_backbones = [
        str(b) for b in pd.unique(metrics_df["backbone"]) if str(b) != "none"
    ] if len(metrics_df) else []

    agg = (
        metrics_df.groupby(
            ["backbone", "model", "score_type", "threshold_policy"]
        )[
            [
                "acc", "macro_f1", "pos_f1", "precision", "recall", "auc",
                "fp_rate_non_toxic", "fn_rate_toxic",
                "hate_fn_rate", "offensive_fn_rate", "threshold",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg.to_csv(out_dir / "metrics_aggregate.csv", index=False, encoding="utf-8-sig")

    strata_df = pd.DataFrame(
        strata_rows,
        columns=[
            "backbone", "model", "seed", "threshold_policy", "stratum",
            "n", "toxic_rate", "mean_prob", "macro_f1", "pos_f1",
            "precision", "recall", "auc", "fp_rate_non_toxic", "fn_rate_toxic",
        ],
    )
    strata_df.to_csv(out_dir / "regime_strata.csv", index=False, encoding="utf-8-sig")

    concept_df = pd.DataFrame(
        concept_rows,
        columns=[
            "backbone", "model", "seed", "concept", "auc", "corr", "bce",
            "target_mean", "pred_mean",
        ],
    )
    if len(concept_df):
        concept_df.to_csv(out_dir / "concept_metrics.csv", index=False, encoding="utf-8-sig")

    leakage_df = pd.DataFrame(
        leakage_rows,
        columns=[
            "backbone", "model", "seed", "mode", "macro_f1",
            "fp_rate_non_toxic", "hate_fn_rate", "offensive_fn_rate",
        ],
    )
    leakage_df.to_csv(out_dir / "leakage.csv", index=False, encoding="utf-8-sig")

    block_df = pd.DataFrame(
        block_rows,
        columns=[
            "backbone", "model", "seed", "block", "macro_f1",
            "delta_macro_f1", "fp_rate_non_toxic", "hate_fn_rate",
            "offensive_fn_rate",
        ],
    )
    block_df.to_csv(out_dir / "block_ablation.csv", index=False, encoding="utf-8-sig")

    op_df = pd.DataFrame(
        op_rows,
        columns=[
            "backbone", "model", "seed", "threshold", "macro_f1", "pos_f1",
            "auc", "fp_rate_non_toxic", "hate_fn_rate", "offensive_fn_rate",
        ],
    )
    op_df.to_csv(out_dir / "operating_point.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def mvals(backbone: str, model: str, score_type: str, policy: str, col: str) -> pd.Series:
        sub = metrics_df[
            (metrics_df["backbone"] == backbone)
            & (metrics_df["model"] == model)
            & (metrics_df["score_type"] == score_type)
            & (metrics_df["threshold_policy"] == policy)
        ][col].dropna()
        return sub

    def mtext(backbone: str, model: str, score_type: str, policy: str, col: str) -> str:
        return base.mean_std_text(mvals(backbone, model, score_type, policy, col))

    def mmean(backbone: str, model: str, score_type: str, policy: str, col: str) -> float:
        v = mvals(backbone, model, score_type, policy, col)
        return float(v.mean()) if len(v) else float("nan")

    lines: List[str] = []
    lines.append("PLM + Concept Bottleneck 诊断报告")
    lines.append("=" * 76)
    lines.append(f"时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"device：{device}，fp16={use_fp16}")
    lines.append(f"backbones：{report_backbones}")
    lines.append(
        f"本次训练 models={selected_models}；reuse_dir={args.reuse_dir or 'none'}"
    )
    lines.append(f"seeds：{seeds}，split_seed={split_seed}，val_ratio={args.val_ratio}")
    lines.append(
        f"train rows={len(train_df)}，test rows={len(test_df)}，concepts={len(concept_cols)}"
    )
    lines.append(
        f"regime weights（sqrt-inverse）："
        f"non_toxic={weights_sqrt[0]:.3f}，hate={weights_sqrt[1]:.3f}，"
        f"offensive={weights_sqrt[2]:.3f}"
    )
    lines.append(
        "模型：plm_only / llm_cbm / plm_cbm_strict / plm_cbm_strict_detached / "
        "plm_cbm_hybrid；binary head 为主，regime head 的 1-P(non_toxic) 为次。"
    )
    lines.append("")

    lines.append("一、二分类整体指标（binary head，threshold=0.5，seed 平均±std）")
    lines.append("-" * 76)
    lines.append(
        f"{'backbone':<24} {'model':<20} {'MacroF1':<17} {'PosF1':<17} "
        f"{'AUC':<17} {'FP_rate':<17} {'FN_hate':<17} {'FN_off'}"
    )
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_only", "plm_cbm_strict", "plm_cbm_strict_detached",
            "plm_cbm_hybrid"
        ]
        for model_name in models:
            lines.append(
                f"{backbone:<24} {model_name:<20} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'macro_f1'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'pos_f1'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'auc'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'fp_rate_non_toxic'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'hate_fn_rate'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'fixed_0.5', 'offensive_fn_rate')}"
            )
    lines.append("")

    lines.append("二、regime head 的 1-P(non_toxic) 二分类指标（threshold=0.5，seed 平均±std）")
    lines.append("-" * 76)
    lines.append(
        f"{'backbone':<24} {'model':<20} {'MacroF1':<17} {'PosF1':<17} "
        f"{'AUC':<17} {'FP_rate':<17} {'FN_hate':<17} {'FN_off'}"
    )
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_only", "plm_cbm_strict", "plm_cbm_strict_detached",
            "plm_cbm_hybrid"
        ]
        for model_name in models:
            lines.append(
                f"{backbone:<24} {model_name:<20} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'macro_f1'):<17} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'pos_f1'):<17} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'auc'):<17} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'fp_rate_non_toxic'):<17} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'hate_fn_rate'):<17} "
                f"{mtext(backbone, model_name, 'reg', 'fixed_0.5', 'offensive_fn_rate')}"
            )
    lines.append("")

    lines.append("三、val-opt 二分类指标（binary head）")
    lines.append("-" * 76)
    lines.append(
        f"{'backbone':<24} {'model':<20} {'MacroF1':<17} {'PosF1':<17} "
        f"{'AUC':<17} {'threshold'}"
    )
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_only", "plm_cbm_strict", "plm_cbm_strict_detached",
            "plm_cbm_hybrid"
        ]
        for model_name in models:
            lines.append(
                f"{backbone:<24} {model_name:<20} "
                f"{mtext(backbone, model_name, 'bin', 'val_opt', 'macro_f1'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'val_opt', 'pos_f1'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'val_opt', 'auc'):<17} "
                f"{mtext(backbone, model_name, 'bin', 'val_opt', 'threshold')}"
            )
    lines.append("")

    lines.append("四、sufficiency 分解（binary head，0.5；ΔMacro-F1）")
    lines.append("-" * 76)
    llm_mean = mmean("none", "llm_cbm", "bin", "fixed_0.5", "macro_f1")
    for backbone in report_backbones:
        strict = mmean(backbone, "plm_cbm_strict", "bin", "fixed_0.5", "macro_f1")
        detached = mmean(backbone, "plm_cbm_strict_detached", "bin", "fixed_0.5", "macro_f1")
        hybrid = mmean(backbone, "plm_cbm_hybrid", "bin", "fixed_0.5", "macro_f1")
        plm = mmean(backbone, "plm_only", "bin", "fixed_0.5", "macro_f1")
        lines.append(
            f"[{backbone}] llm_cbm={llm_mean:.4f}，"
            f"strict={strict:.4f}，detached={detached:.4f}，"
            f"hybrid={hybrid:.4f}，plm_only={plm:.4f}"
        )
        lines.append(
            f"  测量改进（strict - llm_cbm）={strict - llm_mean:+.4f}；"
            f"去耦代价（detached - strict）={detached - strict:+.4f}；"
            f"残差收益（hybrid - strict）={hybrid - strict:+.4f}；"
            f"剩余瓶颈（plm_only - hybrid）={plm - hybrid:+.4f}"
        )
    lines.append("")

    lines.append("五、regime 分层（binary head，0.5；seed 平均 Macro-F1 / FP / FN）")
    lines.append("-" * 76)
    strata_bin = strata_df[strata_df["threshold_policy"] == "fixed_0.5"]
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_only", "plm_cbm_strict", "plm_cbm_strict_detached",
            "plm_cbm_hybrid"
        ]
        for model_name in models:
            lines.append(f"[{backbone} / {model_name}]")
            for stratum_name in ["non_toxic", "hate", "offensive", "expression_0"]:
                s = strata_bin[
                    (strata_bin["backbone"] == backbone)
                    & (strata_bin["model"] == model_name)
                    & (strata_bin["stratum"] == stratum_name)
                ]
                if len(s) == 0:
                    continue
                lines.append(
                    f"  {stratum_name:<12} n={int(s['n'].iloc[0]):<5} "
                    f"MacroF1={base.fmt(s['macro_f1'].mean())} "
                    f"FP={base.fmt(s['fp_rate_non_toxic'].mean())} "
                    f"FN={base.fmt(s['fn_rate_toxic'].mean())}"
                )
    lines.append("")

    lines.append("六、概念蒸馏质量（PLM-CBM；与 LLM soft target 对比）")
    lines.append("-" * 76)
    if len(concept_df):
        cg = (
            concept_df.groupby(["backbone", "model"])[["auc", "corr", "bce"]]
            .mean()
            .reset_index()
        )
        for _, r in cg.iterrows():
            lines.append(
                f"{r['backbone']:<24} {r['model']:<20} "
                f"concept_AUC={base.fmt(r['auc'])} "
                f"corr={base.fmt(r['corr'])} BCE={base.fmt(r['bce'])}"
            )
    else:
        lines.append("（无）")
    lines.append("")

    lines.append("七、leakage：概念 zero / shuffle 后的 Macro-F1（0.5）")
    lines.append("-" * 76)
    lines.append(
        f"{'backbone':<24} {'model':<20} {'normal':<10} {'zero':<10} "
        f"{'shuffle':<10} {'zero_drop':<10} {'shuffle_drop'}"
    )
    lg = leakage_df.groupby(["backbone", "model", "mode"])["macro_f1"].mean().reset_index()
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_cbm_strict", "plm_cbm_strict_detached", "plm_cbm_hybrid"
        ]
        for model_name in models:
            sub = lg[(lg["backbone"] == backbone) & (lg["model"] == model_name)]
            if len(sub) == 0:
                continue
            vals = dict(zip(sub["mode"], sub["macro_f1"]))
            normal = vals.get("normal", float("nan"))
            zero = vals.get("zero_concepts", float("nan"))
            shuf = vals.get("shuffle_concepts", float("nan"))
            lines.append(
                f"{backbone:<24} {model_name:<20} "
                f"{base.fmt(normal):<10} {base.fmt(zero):<10} {base.fmt(shuf):<10} "
                f"{base.fmt(zero - normal):<10} {base.fmt(shuf - normal)}"
            )
    lines.append("")

    lines.append("八、block ablation：概念区块置 0 后的 ΔMacro-F1（0.5）")
    lines.append("-" * 76)
    bg = block_df.groupby(["backbone", "model", "block"])["delta_macro_f1"].mean().reset_index()
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_cbm_strict", "plm_cbm_strict_detached", "plm_cbm_hybrid"
        ]
        for model_name in models:
            sub = bg[(bg["backbone"] == backbone) & (bg["model"] == model_name)].copy()
            if len(sub) == 0:
                continue
            sub["abs"] = sub["delta_macro_f1"].abs()
            sub = sub.sort_values("abs", ascending=False)
            txt = "，".join(
                f"{r['block']}={r['delta_macro_f1']:+.4f}" for _, r in sub.iterrows()
            )
            lines.append(f"{backbone:<24} {model_name:<20} {txt}")
    lines.append("")

    lines.append("九、regime operating point（val 选 hate recall>=90%；test 报告，seed 平均±std）")
    lines.append("-" * 76)
    lines.append(
        f"{'backbone':<24} {'model':<20} {'MacroF1':<17} {'FP_rate':<17} "
        f"{'FN_hate':<17} {'FN_off'}"
    )
    for backbone in ["none"] + report_backbones:
        models = ["llm_cbm"] if backbone == "none" else [
            "plm_only", "plm_cbm_strict", "plm_cbm_strict_detached",
            "plm_cbm_hybrid"
        ]
        for model_name in models:
            sub = op_df[
                (op_df["backbone"] == backbone) & (op_df["model"] == model_name)
            ]
            if len(sub) == 0:
                continue
            lines.append(
                f"{backbone:<24} {model_name:<20} "
                f"{base.mean_std_text(sub['macro_f1']):<17} "
                f"{base.mean_std_text(sub['fp_rate_non_toxic']):<17} "
                f"{base.mean_std_text(sub['hate_fn_rate']):<17} "
                f"{base.mean_std_text(sub['offensive_fn_rate'])}"
            )
    lines.append("")

    # 结论
    lines.append("十、结论与判读")
    lines.append("-" * 76)
    def concept_auc(bb: str, model_name: str) -> float:
        if not len(concept_df):
            return float("nan")
        sub = concept_df[
            (concept_df["backbone"] == bb)
            & (concept_df["model"] == model_name)
        ]["auc"].dropna()
        return float(sub.mean()) if len(sub) else float("nan")

    for backbone in report_backbones:
        strict = mmean(backbone, "plm_cbm_strict", "bin", "fixed_0.5", "macro_f1")
        detached = mmean(backbone, "plm_cbm_strict_detached", "bin", "fixed_0.5", "macro_f1")
        hybrid = mmean(backbone, "plm_cbm_hybrid", "bin", "fixed_0.5", "macro_f1")
        plm = mmean(backbone, "plm_only", "bin", "fixed_0.5", "macro_f1")
        d_measure = strict - llm_mean
        d_detach = detached - strict
        d_resid = hybrid - strict
        d_gap = plm - hybrid
        strict_auc = concept_auc(backbone, "plm_cbm_strict")
        det_auc = concept_auc(backbone, "plm_cbm_strict_detached")
        lines.append(f"[{backbone}]")
        lines.append(
            f"- 测量改进（strict - llm_cbm）={d_measure:+.4f}；"
            f"去耦代价（detached - strict）={d_detach:+.4f}；"
            f"残差收益（hybrid - strict）={d_resid:+.4f}；"
            f"剩余瓶颈（plm_only - hybrid）={d_gap:+.4f}。"
        )
        lines.append(
            f"- 概念 AUC：strict={base.fmt(strict_auc)}，"
            f"detached={base.fmt(det_auc)}；"
            "strict 的概念头同时受标签梯度影响，detached 只受 LLM concept loss 驱动。"
        )
        if d_measure > 0 and d_gap <= 0.01:
            lines.append(
                "- 判读：PLM 概念测量缩小了 CBM 与黑箱 PLM 的差距，"
                "且 hybrid 已接近 plm_only；主张得到支持。"
            )
        elif d_measure > 0:
            lines.append(
                "- 判读：PLM 概念测量有改进，但 hybrid 仍与 plm_only 有明显差距；"
                "瓶颈不只在测量，概念集/覆盖或 bottleneck 本身仍有代价。"
            )
        else:
            lines.append(
                "- 判读：PLM 概念测量没有超过 LLM 概念 CBM；"
                "问题可能在概念集/覆盖或 concept head 训练，而不是单纯测量方式。"
            )
        if math.isfinite(d_detach) and math.isfinite(strict_auc) and math.isfinite(det_auc):
            if abs(d_detach) <= 0.01 and det_auc > strict_auc + 0.01:
                lines.append(
                    "- 判读：去耦后 F1 基本不变且概念 AUC 提升，"
                    "说明忠实概念本身足够；strict 的低 AUC 主要是标签重塑，而不是概念集不足。"
                )
            elif d_detach < -0.01:
                lines.append(
                    "- 判读：去耦后 F1 明显下降，说明 strict 的 F1 依赖概念被标签重塑；"
                    "faithfulness 存疑，下一步应改概念测量/概念集，而不是继续加 readout。"
                )
            else:
                lines.append(
                    "- 判读：去耦后 F1 与概念 AUC 变化有限，"
                    "概念层对标签和 LLM target 的耦合较弱。"
                )
        # leakage warning
        sub_l = lg[(lg["backbone"] == backbone) & (lg["model"] == "plm_cbm_hybrid")]
        if len(sub_l):
            vals = dict(zip(sub_l["mode"], sub_l["macro_f1"]))
            drop = vals.get("zero_concepts", float("nan")) - vals.get("normal", float("nan"))
            if math.isfinite(drop) and drop > -0.01:
                lines.append(
                    "- 警告：hybrid 在 zero_concepts 下几乎不掉点，残差路径可能泄漏标签；"
                    "需要进一步做 CTL/ICL 或概念 dropout 约束。"
                )
    lines.append("")
    lines.append("说明")
    lines.append("-" * 76)
    lines.append("- plm_only 用 raw text；llm_cbm 用 34 维 LLM 分数；plm_cbm_* 用 PLM 从原文测概念。")
    lines.append("- strict 只看概念；hybrid 用概念 + h_cls，训练时对概念/残差做 modality dropout。")
    lines.append("- zero/shuffle concepts 是 leakage 诊断，不是因果干预；block ablation 是敏感性分析。")
    lines.append("- T05 仍作为 34 维概念之一输入；其 target 映射不成立，block ablation 可观察其影响。")
    lines.append("- 所有阈值只在 val 选；test 只用于报告。")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] 结果已写入：{out_dir}")


if __name__ == "__main__":
    main()
