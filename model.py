import torch
import torch.nn as nn

from transformers import AutoModel


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


class MultiModalSceneClassifier(nn.Module):
    def __init__(self, clip_dim, num_tags):
        super().__init__()
        
        self.frame_encoder = VisualTransformer(clip_dim)
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, clip_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=clip_dim,
            num_heads=8,
            batch_first=True
        )

        self.norm = nn.LayerNorm(clip_dim)

        self.head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_tags),
        )

    def forward(self, frames, visual_masks, input_ids, attention_mask):
        v_tokens, v_mask = self.frame_encoder(frames, visual_masks)

        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        t_cls = text_out.last_hidden_state[:, 0]
        t_cls = self.text_proj(t_cls).unsqueeze(1)

        fused, _ = self.cross_attn(
            query=t_cls,
            key=v_tokens,
            value=v_tokens,
            key_padding_mask=v_mask
        )

        fused = self.norm(fused.squeeze(1))
        logits = self.head(fused)

        return logits


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