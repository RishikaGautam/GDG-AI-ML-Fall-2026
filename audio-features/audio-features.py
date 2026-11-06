"""
debate_features.py — extract speaker-quality features from debate .wav files.
 
Usage:
    python debate_features.py debate.wav                  # whole file = one speaker
    python debate_features.py debate.wav segments.csv     # per speaker turn
 
segments.csv columns: start,end,speaker   (seconds; e.g. from pyannote diarization)
 
Output: features.csv, one row per segment, plus speaker_summary.csv aggregated per speaker.
 
pip install librosa praat-parselmouth opensmile pandas numpy soundfile
"""
import sys
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import opensmile
 
SR = 16000  # CREMA-D is 16 kHz mono; standardize everything to match
 
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,      # 88 standard paralinguistic features
    feature_level=opensmile.FeatureLevel.Functionals,
)
 
 
def pause_features(y, sr, top_db=30, min_pause=0.25):
    """Pauses = gaps between non-silent intervals."""
    intervals = librosa.effects.split(y, top_db=top_db)
    total = len(y) / sr
    if len(intervals) == 0:
        return {"pause_count": 0, "pause_mean_s": 0, "pause_ratio": 1.0, "speech_ratio": 0}
    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sr
    gaps = gaps[gaps >= min_pause]
    speech = np.sum(intervals[:, 1] - intervals[:, 0]) / sr
    return {
        "pause_count": len(gaps),
        "pauses_per_min": len(gaps) / (total / 60) if total else 0,
        "pause_mean_s": float(gaps.mean()) if len(gaps) else 0.0,
        "pause_max_s": float(gaps.max()) if len(gaps) else 0.0,
        "speech_ratio": speech / total if total else 0,
    }
 
 
def voice_features(y, sr):
    """Pitch, jitter, shimmer, HNR via Praat."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    out = {
        "f0_mean": float(f0.mean()) if len(f0) else np.nan,
        "f0_std": float(f0.std()) if len(f0) else np.nan,          # low = monotone
        "f0_range": float(np.ptp(f0)) if len(f0) else np.nan,
    }
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        out["jitter"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        out["hnr"] = call(harm, "Get mean", 0, 0)
    except Exception:
        out.update(jitter=np.nan, shimmer=np.nan, hnr=np.nan)
    return out
 
 
def energy_features(y):
    rms = librosa.feature.rms(y=y)[0]
    return {"rms_mean": float(rms.mean()), "rms_std": float(rms.std())}  # vocal emphasis
 
 
def extract(y, sr):
    feats = {"duration_s": len(y) / sr}
    feats.update(pause_features(y, sr))
    feats.update(voice_features(y, sr))
    feats.update(energy_features(y))
    egemaps = smile.process_signal(y, sr).iloc[0].add_prefix("egemaps_").to_dict()
    feats.update(egemaps)
    return feats
 
 
def main(wav_path, seg_path=None):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    if seg_path:
        segs = pd.read_csv(seg_path)
    else:
        segs = pd.DataFrame([{"start": 0, "end": len(y) / sr, "speaker": "A"}])
 
    rows = []
    for _, s in segs.iterrows():
        clip = y[int(s.start * sr): int(s.end * sr)]
        if len(clip) < sr:  # skip turns under 1 s
            continue
        f = extract(clip, sr)
        f.update(speaker=s.speaker, start=s.start, end=s.end)
        rows.append(f)
 
    df = pd.DataFrame(rows)
    df.to_csv("features.csv", index=False)
 
    # per-speaker summary (duration-weighted mean of numeric features)
    num = df.select_dtypes("number").columns.drop(["start", "end"])
    summary = df.groupby("speaker").apply(
        lambda g: pd.Series(np.average(g[num], axis=0, weights=g["duration_s"]), index=num)
    )
    summary["total_talk_s"] = df.groupby("speaker")["duration_s"].sum()
    summary["turns"] = df.groupby("speaker").size()
    summary.to_csv("speaker_summary.csv")
    print(summary[["total_talk_s", "turns", "f0_std", "pauses_per_min", "hnr"]])
 
 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)