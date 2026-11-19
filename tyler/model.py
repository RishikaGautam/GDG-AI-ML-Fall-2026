"""Pretrained speech encoder + learned layer mix + attention pooling + emotion head.

Works with Hugging Face WavLM, HuBERT and wav2vec2 checkpoints, e.g.
    microsoft/wavlm-base-plus      (~95M params, good default)
    microsoft/wavlm-large          (~315M, better, needs a bigger GPU)
    facebook/hubert-base-ls960
The head outputs raw logits; apply log_softmax in the loss and softmax(z / T) at inference.
"""
import contextlib

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class AttentionPool(nn.Module):
    """Weighted average over time; the weights are learned and padding is masked out."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, x, mask):  # x: (B, T, D), mask: (B, T) True = real frame
        s = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=-1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1), w


class SSLEmotionModel(nn.Module):
    def __init__(self, encoder_name="microsoft/wavlm-base-plus", num_classes=6,
                 head_hidden=256, dropout=0.3, use_attention_mask=True, spec_augment=False):
        super().__init__()
        config = AutoConfig.from_pretrained(encoder_name)
        config.layerdrop = 0.0                    # keep every layer's hidden state
        config.apply_spec_augment = spec_augment  # optional time/feature masking
        self.encoder = AutoModel.from_pretrained(encoder_name, config=config)
        self.use_attention_mask = use_attention_mask

        dim = config.hidden_size
        # Emotion lives in middle layers more than the last one, so learn the mix.
        self.layer_logits = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout),
            nn.Linear(dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self.freeze_encoder()

    # ----------------------------------------------------------- freezing
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, k):
        """Fine-tune the top k transformer blocks; the CNN feature extractor stays frozen."""
        self.freeze_encoder()
        if k <= 0:
            return
        for layer in self.encoder.encoder.layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
        if getattr(self.encoder.config, "do_stable_layer_norm", False):
            for p in self.encoder.encoder.layer_norm.parameters():  # final LN in "large" models
                p.requires_grad = True

    @property
    def encoder_frozen(self):
        return not any(p.requires_grad for p in self.encoder.parameters())

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()  # deterministic features while only the head trains
        return self

    # ----------------------------------------------------------- forward
    def embed(self, wav, lengths):
        """wav: (B, S) float at 16 kHz, zero-padded; lengths: (B,) samples. Returns (B, D)."""
        attn = None
        if self.use_attention_mask:
            attn = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None]).long()
        ctx = torch.no_grad() if self.encoder_frozen else contextlib.nullcontext()
        with ctx:
            out = self.encoder(wav, attention_mask=attn, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)                    # (L+1, B, T, D)
        w = torch.softmax(self.layer_logits, dim=0).to(hs.dtype)
        x = (w[:, None, None, None] * hs).sum(dim=0)                  # (B, T, D)

        frame_lens = self.encoder._get_feat_extract_output_lengths(lengths).clamp(min=1)
        mask = torch.arange(x.shape[1], device=x.device)[None] < frame_lens[:, None]
        pooled, _ = self.pool(x, mask)
        return pooled

    def forward(self, wav, lengths):
        return self.head(self.embed(wav, lengths))
