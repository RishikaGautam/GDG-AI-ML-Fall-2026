# Debate emotion recognition: CREMA-D baseline

A pretrained speech encoder (WavLM by default) with a learned layer mix, attention pooling and a
6-emotion head. It is trained on CREMA-D **voice-only** crowd votes as soft labels, using KL loss
and actor-disjoint splits, then calibrated with a fitted temperature.

## Setup
```bash
pip install -r requirements.txt
git lfs install && git clone https://github.com/CheyneyComputerScience/CREMA-D.git
```
Needs a GPU for reasonable training time (WavLM base+ fits in about 8 GB at batch size 16).

## 1. Build soft labels from voice-only votes
```bash
python crema_votes.py --responses CREMA-D/finishedResponses.csv \
    --audio_dir CREMA-D/AudioWAV --out crema_voice_votes.csv
```
Uses only `queryType == 1` (voice-only) responses, smooths the counts (`--alpha 0.5`) and drops
clips with fewer than 3 voice raters. It also prints how often the crowd majority matches the
actor's intended emotion.

## 2. Train
```bash
python train.py --votes crema_voice_votes.csv --out runs/wavlm_base \
    --noise_dir data/musan/noise --rir_dir data/RIRS_NOISES
```
- Phase 1 trains only the head with the encoder frozen. Phase 2 fine-tunes the top 4 encoder layers.
- Augmentation: room reverb plus background noise at 5–20 dB SNR. Point `--noise_dir` at crowd,
  applause and babble recordings (e.g. MUSAN) and `--rir_dir` at room impulse responses
  (e.g. OpenSLR 28). Without them it falls back to white noise and no reverb.
- Outputs `best.pt` (with the fitted temperature), `splits.csv` and `metrics.json`.

Metrics: `kl` (distance to the crowd distribution) is the one used to pick the best checkpoint.
`acc_vs_crowd` compares against the listeners' majority vote. `uar_vs_intended` and
`macro_f1_vs_intended` compare against the actor's intended emotion. Expect intended-label
scores to be modest: human listeners only reach about 41% audio-only.

## 3. Run on debate audio
Per segment, from your timestamped transcript or diarization (columns `start,end[,speaker,text]`):
```bash
python infer.py --ckpt runs/wavlm_base/best.pt --audio debate.wav \
    --segments debate_segments.csv --out segments_scored.csv
```
Or sliding windows over the whole file, with EMA smoothing:
```bash
python infer.py --ckpt runs/wavlm_base/best.pt --audio debate.wav --out windows.csv --ema 0.3
```

## Next steps
1. **Label a debate test set**: a few hundred segments, each rated by 2–3 people. Measure on it,
   and refit the temperature on it (the checkpoint's T is fitted on CREMA-D).
2. **Add naturalistic training data** (MSP-Podcast, IEMOCAP). Convert it to the same table format
   (`path, actor, intended, p_ANG..p_SAD`) and concatenate it with the CREMA-D table.
3. **Text branch**: encode each transcript segment and fuse it with `model.embed(...)`. Train it on
   corpora with real transcripts; CREMA-D's 12 fixed sentences carry no emotional meaning.
4. **Per-speaker normalization** of the pooled embeddings, applied the same way in training and
   inference.
