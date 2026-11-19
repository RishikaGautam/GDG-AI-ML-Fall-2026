"""Train the emotion model on CREMA-D voice-only soft labels.

Phase 1: encoder frozen, train layer mix + pooling + head (fast).
Phase 2: unfreeze the top N encoder layers at a low learning rate.
Then fit a temperature on the validation split and report test metrics.

Usage:
    python train.py --votes crema_voice_votes.csv --out runs/wavlm_base \
        --noise_dir data/musan/noise --rir_dir data/RIRS_NOISES
"""
import argparse
import contextlib
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from data import EMOTIONS, Augmenter, CremaSoftLabelDataset, actor_split, collate_pad
from model import SSLEmotionModel


# --------------------------------------------------------------------------- helpers
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def soft_kl(logits, target):
    """KL(target || softmax(logits)). KLDivLoss wants log-probs as input."""
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def warmup_cosine(opt, total_steps, warmup_frac=0.1):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# --------------------------------------------------------------------------- metrics
def compute_metrics(logits, targets, intended, temperature=1.0):
    logp = F.log_softmax(logits / temperature, dim=-1)
    pred = logp.argmax(-1)
    out = {
        "kl": F.kl_div(logp, targets, reduction="batchmean").item(),
        "acc_vs_crowd": (pred == targets.argmax(-1)).float().mean().item(),
    }
    m = intended >= 0
    p, y = pred[m], intended[m]
    recalls, f1s = [], []
    for c in range(len(EMOTIONS)):
        tp = ((p == c) & (y == c)).sum().item()
        fn = ((p != c) & (y == c)).sum().item()
        fp = ((p == c) & (y != c)).sum().item()
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out["acc_vs_intended"] = (p == y).float().mean().item() if len(y) else float("nan")
    out["uar_vs_intended"] = float(np.mean(recalls))
    out["macro_f1_vs_intended"] = float(np.mean(f1s))
    return out


def fit_temperature(logits, targets):
    """Find T minimising soft cross-entropy on held-out data (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = -(targets * F.log_softmax(logits / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


# --------------------------------------------------------------------------- loops
def run_epoch(model, loader, opt, sched, device, amp_dtype, grad_clip=1.0, log_every=20):
    model.train()
    total, n = 0.0, 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_batches, t0 = len(loader), time.time()
    for step, batch in enumerate(loader, 1):
        wav = batch["wav"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device)
        target = batch["target"].to(device)
        with autocast(device, amp_dtype):
            logits = model(wav, lengths)
        loss = soft_kl(logits.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        opt.step()
        sched.step()
        total += loss.item() * wav.size(0)
        n += wav.size(0)
        if step % log_every == 0 or step == n_batches:
            elapsed = time.time() - t0
            eta = elapsed / step * (n_batches - step)
            print(f"  batch {step}/{n_batches}  train_kl={total / n:.4f}  "
                  f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)
    return total / max(n, 1)


@torch.no_grad()
def collect(model, loader, device, amp_dtype):
    print(f"  evaluating {len(loader.dataset)} clips...", flush=True)
    model.eval()
    all_logits, all_t, all_i = [], [], []
    for batch in loader:
        with autocast(device, amp_dtype):
            logits = model(batch["wav"].to(device), batch["lengths"].to(device))
        all_logits.append(logits.float().cpu())
        all_t.append(batch["target"])
        all_i.append(batch["intended"])
    return torch.cat(all_logits), torch.cat(all_t), torch.cat(all_i)


def save_ckpt(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def train_phase(name, model, epochs, param_groups, loaders, device, amp_dtype,
                best, ckpt_path, meta, patience):
    if epochs <= 0:
        return best
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    sched = warmup_cosine(opt, epochs * len(loaders["train"]))
    bad = 0
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, loaders["train"], opt, sched, device, amp_dtype)
        m = compute_metrics(*collect(model, loaders["val"], device, amp_dtype))
        improved = m["kl"] < best["kl"]
        print(f"[{name}] epoch {ep}/{epochs}  train_kl={tr_loss:.4f}  val_kl={m['kl']:.4f}  "
              f"val_acc_crowd={m['acc_vs_crowd']:.3f}  val_uar_intended={m['uar_vs_intended']:.3f}"
              + ("  *" if improved else ""))
        if improved:
            best = {**m, "phase": name, "epoch": ep}
            save_ckpt(ckpt_path, model, meta)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop")
                break
    return best


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--votes", required=True, help="output of crema_votes.py")
    ap.add_argument("--out", default="runs/wavlm_base")
    ap.add_argument("--encoder", default="microsoft/wavlm-base-plus")
    ap.add_argument("--epochs_frozen", type=int, default=8)
    ap.add_argument("--epochs_finetune", type=int, default=12)
    ap.add_argument("--unfreeze_layers", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_seconds", type=float, default=4.0)
    ap.add_argument("--noise_dir", default=None)
    ap.add_argument("--rir_dir", default=None)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"device={device} amp={amp_dtype}")

    table = pd.read_csv(args.votes, dtype={"actor": str})
    table = actor_split(table, args.val_frac, args.test_frac, args.seed)
    table.to_csv(os.path.join(args.out, "splits.csv"), index=False)
    for s in ("train", "val", "test"):
        sub = table[table.split == s]
        print(f"{s}: {len(sub)} clips, {sub.actor.nunique()} actors")

    fe = AutoFeatureExtractor.from_pretrained(args.encoder)
    normalize = bool(getattr(fe, "do_normalize", True))
    use_mask = bool(getattr(fe, "return_attention_mask", False))

    augment = None if args.no_augment else Augmenter(args.noise_dir, args.rir_dir)
    ds = {
        "train": CremaSoftLabelDataset(table[table.split == "train"], True, args.max_seconds, augment, normalize),
        "val": CremaSoftLabelDataset(table[table.split == "val"], False, normalize=normalize),
        "test": CremaSoftLabelDataset(table[table.split == "test"], False, normalize=normalize),
    }
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"),
                             drop_last=(k == "train"), num_workers=args.num_workers,
                             collate_fn=collate_pad, pin_memory=device.type == "cuda")
               for k, v in ds.items()}

    model = SSLEmotionModel(args.encoder, num_classes=len(EMOTIONS), use_attention_mask=use_mask).to(device)
    meta = {"encoder": args.encoder, "emotions": EMOTIONS, "normalize": normalize,
            "use_attention_mask": use_mask, "temperature": 1.0, "args": vars(args)}
    ckpt_path = os.path.join(args.out, "best.pt")
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    best = {"kl": float("inf")}

    # Phase 1: frozen encoder
    best = train_phase("frozen", model, args.epochs_frozen,
                       [{"params": head_params, "lr": args.lr_head}],
                       loaders, device, amp_dtype, best, ckpt_path, meta, args.patience)

    # Phase 2: fine-tune top layers, starting from the best phase-1 weights
    if args.epochs_finetune > 0 and args.unfreeze_layers > 0:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
        model.unfreeze_top_layers(args.unfreeze_layers)
        enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
        print(f"Fine-tuning {sum(p.numel() for p in enc_params) / 1e6:.1f}M encoder params")
        best = train_phase("finetune", model, args.epochs_finetune,
                           [{"params": head_params, "lr": args.lr_head * 0.1},
                            {"params": enc_params, "lr": args.lr_encoder}],
                           loaders, device, amp_dtype, best, ckpt_path, meta, args.patience)

    # Calibrate on val, report on test
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    val_logits, val_t, _ = collect(model, loaders["val"], device, amp_dtype)
    T = fit_temperature(val_logits, val_t)
    meta["temperature"] = T
    save_ckpt(ckpt_path, model, meta)

    test = collect(model, loaders["test"], device, amp_dtype)
    results = {"best_val": best, "temperature": T,
               "test_T1": compute_metrics(*test, temperature=1.0),
               "test_calibrated": compute_metrics(*test, temperature=T)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved {ckpt_path} (temperature={T:.2f}). Note: this T is fitted on CREMA-D; "
          f"refit it on labelled debate segments before trusting debate probabilities.")


if __name__ == "__main__":
    main()
