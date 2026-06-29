import torch
import torch.nn as nn

from transformers import AutoModel


# Sentence-Transformers model: trained so mean-pooled output is a strong
# sentence/paragraph embedding. Much better frozen representation than vanilla
# distilbert-base-uncased, and a 768-dim drop-in replacement.
DEFAULT_TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def masked_mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _transformer_layers(text_encoder):
    """Return the ModuleList of transformer blocks across HF architectures."""
    if hasattr(text_encoder, "encoder") and hasattr(text_encoder.encoder, "layer"):
        return text_encoder.encoder.layer        # BERT / RoBERTa / MPNet / MiniLM
    if hasattr(text_encoder, "transformer") and hasattr(text_encoder.transformer, "layer"):
        return text_encoder.transformer.layer    # DistilBERT
    raise ValueError("Unsupported text encoder architecture for partial unfreezing")


def set_text_trainable(text_encoder, num_layers):
    """Freeze the text encoder, then unfreeze its top `num_layers` transformer blocks."""
    for p in text_encoder.parameters():
        p.requires_grad = False
    if num_layers and num_layers > 0:
        layers = _transformer_layers(text_encoder)
        for layer in layers[-num_layers:]:
            for p in layer.parameters():
                p.requires_grad = True


class VisualTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dropout=0.2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, dim) * 0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        B, N, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = x + self.pos_emb[:, :N+1]
        x = self.encoder(x, src_key_padding_mask=full_mask)

        x = self.norm(x)
        
        return x, full_mask


FUSION_OPTIONS = ("cross_attn", "concat", "add", "film")


class MultiModalSceneClassifier(nn.Module):
    def __init__(self, clip_dim, num_tags, audio_dim, fusion="cross_attn", text_model=DEFAULT_TEXT_MODEL):
        super().__init__()
        assert fusion in FUSION_OPTIONS, f"fusion must be one of {FUSION_OPTIONS}"
        self.fusion = fusion

        # vision is the primary ("content") stream
        self.frame_encoder = VisualTransformer(clip_dim)

        # text and audio are pooled "context" streams that condition vision
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, clip_dim)

        self.audio_encoder = VisualTransformer(audio_dim)
        self.audio_proj = nn.Linear(audio_dim, clip_dim)

        # combine the text and audio context vectors into a single conditioner
        self.context_proj = nn.Linear(clip_dim * 2, clip_dim)

        if fusion == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(embed_dim=clip_dim, num_heads=8, batch_first=True)
        elif fusion == "concat":
            self.fusion_proj = nn.Linear(clip_dim * 2, clip_dim)
        elif fusion == "film":
            # text generates per-dimension scale and shift to modulate visual
            self.film_scale = nn.Linear(clip_dim, clip_dim)
            self.film_shift = nn.Linear(clip_dim, clip_dim)
            # identity init: scale->1, shift->0 so fusion starts as a pure
            # visual pass-through and learns text modulation gradually
            nn.init.zeros_(self.film_scale.weight)
            nn.init.ones_(self.film_scale.bias)
            nn.init.zeros_(self.film_shift.weight)
            nn.init.zeros_(self.film_shift.bias)
        # "add" needs no extra layers

        self.norm = nn.LayerNorm(clip_dim)

        self.head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, frames, visual_masks, input_ids, attention_mask, audio, audio_masks):
        v_tokens, v_mask = self.frame_encoder(frames, visual_masks)
        v_cls = v_tokens[:, 0]

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_pooled = masked_mean_pool(text_out.last_hidden_state, attention_mask)
        t_cls = self.text_proj(t_pooled)

        a_tokens, _ = self.audio_encoder(audio, audio_masks)
        a_cls = self.audio_proj(a_tokens[:, 0])

        # text + audio -> single conditioning context for the visual stream
        ctx = self.context_proj(torch.cat([t_cls, a_cls], dim=-1))

        if self.fusion == "cross_attn":
            fused, _ = self.cross_attn(
                query=ctx.unsqueeze(1),
                key=v_tokens,
                value=v_tokens,
                key_padding_mask=v_mask,
            )
            fused = self.norm(fused.squeeze(1))
        elif self.fusion == "concat":
            fused = self.norm(self.fusion_proj(torch.cat([v_cls, ctx], dim=-1)))
        elif self.fusion == "add":
            fused = self.norm(v_cls + ctx)
        elif self.fusion == "film":
            scale = self.film_scale(ctx)
            shift = self.film_shift(ctx)
            fused = self.norm(scale * v_cls + shift)

        return self.head(fused)


class TextSceneClassifier(nn.Module):
    def __init__(self, num_tags, text_model=DEFAULT_TEXT_MODEL):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(text_out.last_hidden_state, attention_mask)
        return self.head(pooled)


class VisionSceneClassifier(nn.Module):
    def __init__(self, clip_dim, num_tags):
        super().__init__()

        self.frame_encoder = VisualTransformer(clip_dim)

        self.head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, frames, visual_masks):
        v_tokens, v_mask = self.frame_encoder(frames, visual_masks)
        v_cls = v_tokens[:, 0]
        logits = self.head(v_cls)

        return logits


class AudioSceneClassifier(nn.Module):
    def __init__(self, audio_dim, num_tags):
        super().__init__()

        # CLAP embeddings form a per-window sequence; reuse the generic
        # sequence transformer (same architecture as the visual encoder).
        self.audio_encoder = VisualTransformer(audio_dim)

        self.head = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, audio, audio_masks):
        a_tokens, a_mask = self.audio_encoder(audio, audio_masks)
        a_cls = a_tokens[:, 0]
        logits = self.head(a_cls)

        return logits