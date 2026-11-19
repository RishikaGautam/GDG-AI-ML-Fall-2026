"""Score debate audio with a trained checkpoint.

Two modes:

1. Sliding windows over a whole recording (no transcript needed):
    python infer.py --ckpt runs/wavlm_base/best.pt --audio debate.wav --out windows.csv --ema 0.3

2. Per segment, using your teammates' timestamped transcript / diarization:
    python infer.py --ckpt runs/wavlm_base/best.pt --audio debate.wav \
        --segments debate_segments.csv --out segments.csv

   debate_segments.csv needs columns: start, end (seconds) and optionally speaker, text.
   Long segments are cut into windows; their logits are averaged before the softmax.
"""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from data import SR, load_audio, pad_batch, standardize
from model import SSLEmotionModel


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = SSLEmotionModel(meta["encoder"], num_classes=len(meta["emotions"]),
                            use_attention_mask=meta["use_attention_mask"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def window_spans(start, end, win, hop):
    """Sample spans of length `win` covering [start, end); short spans are kept whole."""
    if end - start <= win:
        return [(start, end)]
    spans = [(s, s + win) for s in range(start, end - win + 1, hop)]
    if spans[-1][1] < end:
        spans.append((end - win, end))
    return spans


@torch.no_grad()
def score_spans(model, wav, spans, meta, device, batch_size):
    out = []
    for i in range(0, len(spans), batch_size):
        chunk = [wav[s:e] for s, e in spans[i:i + batch_size]]
        if meta["normalize"]:
            chunk = [standardize(c) for c in chunk]
        batch, lengths = pad_batch(chunk)
        out.append(model(batch.to(device), lengths.to(device)).float().cpu())
    return torch.cat(out) if out else torch.empty(0, len(meta["emotions"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--segments", default=None, help="CSV with start,end[,speaker,text] in seconds")
    ap.add_argument("--out", default="emotions.csv")
    ap.add_argument("--win", type=float, default=3.0, help="window length, seconds")
    ap.add_argument("--hop", type=float, default=1.0, help="window hop, seconds")
    ap.add_argument("--min_seconds", type=float, default=0.5, help="skip segments shorter than this")
    ap.add_argument("--temperature", type=float, default=None, help="override the checkpoint's T")
    ap.add_argument("--ema", type=float, default=None, help="EMA weight on the newest window (window mode)")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.ckpt, device)
    emotions = meta["emotions"]
    T = args.temperature or meta.get("temperature", 1.0)
    wav = load_audio(args.audio)
    win, hop = int(args.win * SR), int(args.hop * SR)
    print(f"{args.audio}: {wav.numel() / SR:.1f}s, temperature={T:.2f}")

    if args.segments:
        seg = pd.read_csv(args.segments)
        rows, spans, owner = [], [], []
        for idx, r in seg.iterrows():
            s, e = int(r["start"] * SR), min(int(r["end"] * SR), wav.numel())
            if e - s < args.min_seconds * SR:
                continue
            for sp in window_spans(s, e, win, hop):
                spans.append(sp)
                owner.append(idx)
        logits = score_spans(model, wav, spans, meta, device, args.batch_size)
        owner = torch.tensor(owner)
        for idx in sorted(set(owner.tolist())):
            mean_logits = logits[owner == idx].mean(0)
            probs = F.softmax(mean_logits / T, dim=-1)
            row = seg.loc[idx].to_dict()
            row.update({f"p_{e}": float(p) for e, p in zip(emotions, probs)})
            row["top"] = emotions[int(probs.argmax())]
            rows.append(row)
        out = pd.DataFrame(rows)
        print(f"Scored {len(out)} of {len(seg)} segments")
    else:
        spans = window_spans(0, wav.numel(), win, hop)
        logits = score_spans(model, wav, spans, meta, device, args.batch_size)
        probs = F.softmax(logits / T, dim=-1)
        out = pd.DataFrame(probs.numpy(), columns=[f"p_{e}" for e in emotions])
        out.insert(0, "start", [s / SR for s, _ in spans])
        out.insert(1, "end", [e / SR for _, e in spans])
        if args.ema:
            smoothed, prev = [], None
            for p in probs:
                prev = p if prev is None else args.ema * p + (1 - args.ema) * prev
                smoothed.append(prev)
            sm = torch.stack(smoothed)
            for j, e in enumerate(emotions):
                out[f"ema_{e}"] = sm[:, j].numpy()
        out["top"] = [emotions[i] for i in probs.argmax(-1).tolist()]

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
