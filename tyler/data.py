"""Audio loading, augmentation, actor-disjoint splits and the CREMA-D dataset."""
import glob
import os
import random

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

SR = 16000
EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


# --------------------------------------------------------------------------- audio
def load_audio(path, sr=SR):
    """Load any WAV/FLAC as mono float32 at `sr`."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav


def standardize(wav):
    return (wav - wav.mean()) / (wav.std() + 1e-7)


def fft_convolve(x, h):
    n = x.numel() + h.numel() - 1
    nfft = 1 << (n - 1).bit_length()
    return torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)[:n]


def mix_at_snr(clean, noise, snr_db):
    p_clean = clean.pow(2).mean().clamp_min(1e-10)
    p_noise = noise.pow(2).mean().clamp_min(1e-10)
    scale = torch.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return clean + scale * noise


def _list_audio(folder):
    if not folder:
        return []
    files = []
    for ext in ("wav", "flac"):
        files += glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True)
    return sorted(files)


class Augmenter:
    """Make clean studio clips sound more like a debate hall.

    noise_dir: background noise (crowd, applause, room tone, babble), e.g. MUSAN.
    rir_dir:   room impulse responses, e.g. the OpenSLR RIRS_NOISES set.
    Without noise_dir it falls back to white noise, which is a weak proxy.
    """

    def __init__(self, noise_dir=None, rir_dir=None, p_noise=0.6, p_rir=0.4, snr_db=(5.0, 20.0)):
        self.noises = _list_audio(noise_dir)
        self.rirs = _list_audio(rir_dir)
        self.p_noise, self.p_rir, self.snr_db = p_noise, p_rir, snr_db
        print(f"Augmenter: {len(self.noises)} noise files, {len(self.rirs)} RIRs")

    def _noise(self, n):
        if not self.noises:
            return torch.randn(n)
        noise = load_audio(random.choice(self.noises))
        if noise.numel() < 160:  # empty or near-empty file: fall back to white noise
            return torch.randn(n)
        if noise.numel() < n:
            noise = noise.repeat(n // max(noise.numel(), 1) + 1)
        start = random.randint(0, noise.numel() - n)
        return noise[start:start + n]

    def __call__(self, wav):
        if self.rirs and random.random() < self.p_rir:
            rir = load_audio(random.choice(self.rirs))
            rir = rir / (rir.norm() + 1e-8)
            peak = int(rir.abs().argmax())  # keep the direct path aligned with the dry signal
            wav = fft_convolve(wav, rir)[peak:peak + wav.numel()]
        if random.random() < self.p_noise:
            wav = mix_at_snr(wav, self._noise(wav.numel()), random.uniform(*self.snr_db))
        return wav


# --------------------------------------------------------------------------- splits
def actor_split(table, val_frac=0.15, test_frac=0.15, seed=0):
    """Assign every clip of an actor to exactly one split (no speaker leakage)."""
    actors = sorted(table["actor"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(actors)
    n_test = round(len(actors) * test_frac)
    n_val = round(len(actors) * val_frac)
    split_of = {a: "test" for a in actors[:n_test]}
    split_of.update({a: "val" for a in actors[n_test:n_test + n_val]})
    split_of.update({a: "train" for a in actors[n_test + n_val:]})
    table = table.copy()
    table["split"] = table["actor"].astype(str).map(split_of)
    return table


# --------------------------------------------------------------------------- dataset
class CremaSoftLabelDataset(Dataset):
    """Returns (waveform, soft target over 6 emotions, intended-emotion index or -1)."""

    def __init__(self, table: pd.DataFrame, train: bool, max_seconds=4.0,
                 augment=None, normalize=True):
        self.paths = table["path"].tolist()
        self.targets = torch.tensor(table[[f"p_{e}" for e in EMOTIONS]].values, dtype=torch.float32)
        self.targets = self.targets / self.targets.sum(dim=1, keepdim=True)
        self.intended = torch.tensor(
            [EMOTIONS.index(e) if e in EMOTIONS else -1 for e in table["intended"]], dtype=torch.long)
        self.train = train
        self.max_len = int(max_seconds * SR)
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav = load_audio(self.paths[i])
        if self.train:
            if wav.numel() > self.max_len:
                start = random.randint(0, wav.numel() - self.max_len)
                wav = wav[start:start + self.max_len]
            if self.augment is not None:
                wav = self.augment(wav)
        if self.normalize:
            wav = standardize(wav)
        return wav, self.targets[i], self.intended[i]


def pad_batch(wavs):
    lengths = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    out = torch.zeros(len(wavs), int(lengths.max()))
    for i, w in enumerate(wavs):
        out[i, :w.numel()] = w
    return out, lengths


def collate_pad(batch):
    wavs, targets, intended = zip(*batch)
    wav, lengths = pad_batch(wavs)
    return {"wav": wav, "lengths": lengths,
            "target": torch.stack(targets), "intended": torch.stack(intended)}
