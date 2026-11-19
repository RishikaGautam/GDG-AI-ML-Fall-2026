"""Build voice-only soft labels for CREMA-D from finishedResponses.csv.

CREMA-D raters judged each clip three ways (queryType 1 = voice only,
2 = face only, 3 = audio-visual). An audio model should learn what listeners
*heard*, so only queryType == 1 responses are counted here.

Output: one row per clip with vote counts, smoothed probabilities, the
actor-intended emotion (from the filename) and the path to the WAV.

Usage:
    python crema_votes.py \
        --responses CREMA-D/finishedResponses.csv \
        --audio_dir CREMA-D/AudioWAV \
        --out crema_voice_votes.csv
"""
import argparse
import os

import pandas as pd

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
LETTER_TO_EMO = {"A": "ANG", "D": "DIS", "F": "FEA", "H": "HAP", "N": "NEU", "S": "SAD"}
WORD_TO_EMO = {
    "anger": "ANG", "angry": "ANG", "disgust": "DIS", "fear": "FEA",
    "happy": "HAP", "happiness": "HAP", "neutral": "NEU", "sad": "SAD", "sadness": "SAD",
}
VOICE_ONLY = 1


def normalize_emotion(x):
    """Map 'A' / 'ANG' / 'anger' style codes to the 3-letter code, else None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.upper() in EMOTIONS:
        return s.upper()
    if s.upper() in LETTER_TO_EMO:
        return LETTER_TO_EMO[s.upper()]
    return WORD_TO_EMO.get(s.lower())


def clip_id(name):
    """'1001_DFA_ANG_XX.flv' or a path -> '1001_DFA_ANG_XX'."""
    return os.path.splitext(os.path.basename(str(name).strip()))[0]


def build(responses_csv, audio_dir, alpha=0.5, min_raters=3):
    df = pd.read_csv(responses_csv, low_memory=False)
    required = {"queryType", "clipName", "respEmo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{responses_csv} is missing columns {sorted(missing)}; "
                         f"found {list(df.columns)}")

    df = df[pd.to_numeric(df["queryType"], errors="coerce") == VOICE_ONLY].copy()
    df["emo"] = df["respEmo"].map(normalize_emotion)
    n_bad = int(df["emo"].isna().sum())
    if n_bad:
        print(f"Dropping {n_bad} voice-only responses with unrecognised respEmo values: "
              f"{sorted(df.loc[df['emo'].isna(), 'respEmo'].astype(str).unique())[:10]}")
    df = df.dropna(subset=["emo"])
    df["clip"] = df["clipName"].map(clip_id)

    counts = (pd.crosstab(df["clip"], df["emo"])
              .reindex(columns=EMOTIONS, fill_value=0)
              .astype(int))
    n = counts.sum(axis=1)
    keep = n >= min_raters
    print(f"{len(counts)} clips have voice-only votes; "
          f"{int((~keep).sum())} dropped for having < {min_raters} raters")
    counts, n = counts[keep], n[keep]

    # Additive smoothing: with ~3-10 voice raters per clip, raw fractions are
    # noisy and full of hard zeros.
    probs = (counts + alpha).div(n + alpha * len(EMOTIONS), axis=0)

    out = pd.DataFrame(index=counts.index)
    out.index.name = "clip"
    parts = out.index.to_series().str.split("_")
    out["actor"] = parts.str[0]
    out["intended"] = parts.str[2].fillna("")
    out["intended_level"] = parts.str[3].fillna("")
    for e in EMOTIONS:
        out[f"n_{e}"] = counts[e]
    out["n_raters"] = n
    for e in EMOTIONS:
        out[f"p_{e}"] = probs[e]
    out["crowd_top"] = counts.idxmax(axis=1)
    out["path"] = [os.path.join(audio_dir, c + ".wav") for c in out.index]

    ok = out["path"].map(lambda p: os.path.isfile(p) and os.path.getsize(p) > 1024)
    if (~ok).any():
        print(f"Dropping {int((~ok).sum())} clips whose WAV is missing or empty "
              f"(e.g. {out.index[~ok][:3].tolist()})")
    out = out[ok]

    agree = (out["crowd_top"] == out["intended"]).mean()
    print(f"Kept {len(out)} clips from {out['actor'].nunique()} actors; "
          f"mean voice raters/clip = {out['n_raters'].mean():.1f}; "
          f"crowd majority matches intended emotion on {agree:.1%} of clips")
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="CREMA-D finishedResponses.csv")
    ap.add_argument("--audio_dir", required=True, help="CREMA-D AudioWAV folder")
    ap.add_argument("--out", default="crema_voice_votes.csv")
    ap.add_argument("--alpha", type=float, default=0.5, help="additive smoothing per class")
    ap.add_argument("--min_raters", type=int, default=3)
    args = ap.parse_args()
    table = build(args.responses, args.audio_dir, args.alpha, args.min_raters)
    table.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
