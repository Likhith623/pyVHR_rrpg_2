# ═══════════════════════════════════════════════════════════════════════════
# predict.py  —  NeuroPulse  |  FastAPI backend + inference logic
#
# Run:  uvicorn predict:app --host 0.0.0.0 --port 8000
#
# File layout expected:
#   backend/
#     predict.py              ← this file
#     ensemble_model.pkl
#     models/
#       efficientnet_model.pth
#       swin_model.pth
#       xception_model.pth
#     rppg/
#       best_rppg_ml_model.joblib
#       rppg_scaler.joblib
#       rppg_selector.joblib
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os, math, tempfile, shutil, cv2, joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit, logit
from scipy.signal import butter, filtfilt, welch
import albumentations as A
import timm
import mediapipe as mp

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NeuroPulse Deepfake Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS           = 1e-7
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD  = [0.5, 0.5, 0.5]

# ═════════════════════════════════════════════════════════════════════════════
# MODEL CLASS DEFINITIONS
# Extracted verbatim from training notebooks — DO NOT alter hyperparameters.
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EfficientNet-B4  (from model_efficientnet notebook)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Multi-head self-attention for temporal sequence modeling."""
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        key_padding_mask = ~mask if mask is not None else None
        attn_out, attn_weights = self.attention(x, x, x,
                                                key_padding_mask=key_padding_mask)
        x = self.layer_norm(x + self.dropout(attn_out))
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return pooled, attn_weights


class SpatioTemporalDeepfakeCNN(nn.Module):
    """
    EfficientNet-B4 backbone + BiLSTM + Multi-Head Attention.
    Trained config: hidden_dim=256, lstm_hidden=256, lstm_layers=2,
                    attention_heads=4, dropout=0.5
    """
    def __init__(self, model_name='efficientnet_b4', hidden_dim=256, dropout=0.3,
                 pretrained=True, temporal_type='bilstm_attention',
                 lstm_hidden=256, lstm_layers=2, attention_heads=4,
                 freeze_backbone=False):
        super().__init__()
        self.temporal_type = temporal_type

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            global_pool='avg', drop_path_rate=0.2)
        self.backbone_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if temporal_type in ['bilstm', 'bilstm_attention']:
            self.temporal = nn.LSTM(
                input_size=self.backbone_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0)
            temporal_out_dim = lstm_hidden * 2
            self.temporal_dropout = nn.Dropout(p=dropout)

            if temporal_type == 'bilstm_attention':
                self.temporal_attention = TemporalAttention(
                    feature_dim=lstm_hidden * 2,
                    num_heads=attention_heads,
                    dropout=dropout)
        elif temporal_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.backbone_dim, nhead=attention_heads,
                dim_feedforward=self.backbone_dim * 2,
                dropout=dropout, activation='gelu', batch_first=True)
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=lstm_layers)
            self.temporal_attention = TemporalAttention(
                feature_dim=self.backbone_dim,
                num_heads=attention_heads, dropout=dropout)
            temporal_out_dim = self.backbone_dim
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.temporal_out_dim = temporal_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames, mask=None):
        B, T, C, H, W = frames.shape
        flat_frames = frames.view(B * T, C, H, W)
        features    = self.backbone(flat_frames).view(B, T, -1)

        if self.temporal_type == 'bilstm':
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (lstm_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = lstm_out.mean(dim=1)
        elif self.temporal_type == 'bilstm_attention':
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            lstm_out = self.temporal_dropout(lstm_out)
            pooled, _ = self.temporal_attention(lstm_out, mask)
        elif self.temporal_type == 'transformer':
            attn_mask = ~mask if mask is not None else None
            trans_out = self.temporal(features, src_key_padding_mask=attn_mask)
            pooled, _ = self.temporal_attention(trans_out, mask)

        return self.classifier(pooled).squeeze(-1)

    def get_attention_weights(self, frames, mask=None):
        with torch.no_grad():
            B, T, C, H, W = frames.shape
            flat_frames = frames.view(B * T, C, H, W)
            features    = self.backbone(flat_frames).view(B, T, -1)
            if self.temporal_type == 'bilstm_attention':
                with torch.backends.cudnn.flags(enabled=False):
                    lstm_out, _ = self.temporal(features)
                lstm_out = self.temporal_dropout(lstm_out)
                _, attn_weights = self.temporal_attention(lstm_out, mask)
                return attn_weights
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Shared ECA module  (used by both Swin and Xception)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientChannelAttention(nn.Module):
    """ECA-Net: 1D conv over channel dimension."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.reshape(B * T, C)
        y = x.unsqueeze(1)
        y = self.conv(y)
        y = torch.sigmoid(y.squeeze(1))
        x = x * y
        if len(orig_shape) == 3:
            x = x.reshape(B, T, C)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Xception  (from model_xception notebook)
# ─────────────────────────────────────────────────────────────────────────────

class SpatioTemporalBiLSTM(nn.Module):
    """
    Xception backbone + ECA + Frequency Branch + BiLSTM + MHA.
    Trained config: hidden_dim=256, num_layers=2, dropout=0.3, attention_heads=4
    """
    def __init__(self, cnn_backbone='xception', pretrained=True,
                 hidden_dim=256, num_layers=2, dropout=0.3,
                 attention_heads=4, target_size=1):
        super().__init__()
        self.cnn_backbone_name = cnn_backbone

        if cnn_backbone in ('xception', 'legacy_xception'):
            self.backbone = timm.create_model(
                'legacy_xception', pretrained=pretrained,
                num_classes=0, global_pool='avg')
            cnn_out_dim = self.backbone.num_features   # 2048
        elif cnn_backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(
                cnn_backbone, pretrained=pretrained,
                num_classes=0, global_pool='avg')
            cnn_out_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")

        self.input_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.channel_attention = EfficientChannelAttention(cnn_out_dim)

        self.freq_branch = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.temporal = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        temporal_out_dim = hidden_dim * 2
        self.temporal_dropout = nn.Dropout(p=dropout)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_out_dim,
            num_heads=attention_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(temporal_out_dim)

        fused_dim = temporal_out_dim + hidden_dim   # 512 + 256 = 768
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, target_size),
        )
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'backbone' not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat           = x.reshape(B * T, C, H, W)
        spatial_features = self.backbone(x_flat).reshape(B, T, -1)
        spatial_features = self.channel_attention(spatial_features)

        freq_features = self.freq_branch(spatial_features)
        if mask is not None:
            freq_mask  = mask.unsqueeze(-1).float()
            freq_pooled = (freq_features * freq_mask).sum(dim=1) / freq_mask.sum(dim=1).clamp(min=1)
        else:
            freq_pooled = freq_features.mean(dim=1)

        projected = self.input_proj(spatial_features)

        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.temporal(projected)
        lstm_out = self.temporal_dropout(lstm_out)

        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.temporal_attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask)
        temporal_features = self.attn_norm(lstm_out + attn_out)

        if mask is not None:
            temp_mask = mask.unsqueeze(-1).float()
            temporal_pooled = (temporal_features * temp_mask).sum(dim=1) / temp_mask.sum(dim=1).clamp(min=1)
        else:
            temporal_pooled = temporal_features.mean(dim=1)

        fused  = torch.cat([temporal_pooled, freq_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Swin Transformer  (from model_swin notebook)
# ─────────────────────────────────────────────────────────────────────────────

class SpatioTemporalSwinCNN(nn.Module):
    """
    Swin-Tiny backbone + ECA + on-the-fly DCT branch + pack_padded BiLSTM + MHA.
    Trained config: hidden_dim=192, lstm_hidden=256, num_layers=2,
                    attention_heads=4, dropout=0.3, drop_path_rate=0.2
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True,
                 hidden_dim=192, lstm_hidden=256, num_layers=2, dropout=0.3,
                 attention_heads=4, drop_path_rate=0.2, freeze_backbone=False):
        super().__init__()

        # Pre-compute DCT matrix as a non-trainable buffer
        dct_m = np.empty((64, 64))
        for k in range(64):
            for n in range(64):
                dct_m[k, n] = math.cos(math.pi * k * (2.0 * n + 1) / 128.0)
        dct_m[0, :] /= math.sqrt(2.0)
        dct_m *= math.sqrt(2.0 / 64)
        self.register_buffer('dct_m', torch.from_numpy(dct_m).float())

        self.register_buffer('imagenet_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg',
            drop_path_rate=drop_path_rate)
        cnn_out_dim = self.backbone.num_features   # 768 for Swin-Tiny

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.channel_attention = EfficientChannelAttention(cnn_out_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # DCT freq branch: 128 raw DCT features → hidden_dim (192)
        self.freq_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.temporal = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        temporal_out_dim = lstm_hidden * 2          # 512
        self.temporal_dropout = nn.Dropout(p=dropout)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_out_dim,
            num_heads=attention_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(temporal_out_dim)

        fused_dim = temporal_out_dim + hidden_dim   # 512 + 192 = 704
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'backbone' not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param.data)
                        n = param.data.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)  # forget-gate bias = 1

    def _rgb_to_dct_features(self, x):
        x_denorm = x * self.imagenet_std + self.imagenet_mean
        gray = 0.299 * x_denorm[:, 0] + 0.587 * x_denorm[:, 1] + 0.114 * x_denorm[:, 2]
        down = F.interpolate(gray.unsqueeze(1), size=(64, 64),
                             mode='bilinear', align_corners=False).squeeze(1)
        dct_feat = torch.matmul(torch.matmul(self.dct_m, down), self.dct_m.t())
        dct_feat = torch.log(torch.abs(dct_feat) + 1e-6)
        blocks   = dct_feat.unfold(1, 8, 8).unfold(2, 8, 8)
        means    = blocks.mean(dim=(3, 4)).reshape(x.size(0), 64)
        stds     = blocks.std(dim=(3, 4)).reshape(x.size(0), 64)
        return torch.cat([means, stds], dim=1)

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)

        # Skip padded frames in backbone
        if mask is not None:
            real_mask_flat = mask.view(-1)
            real_spatial   = self.backbone(x_flat[real_mask_flat])
            spatial_flat   = torch.zeros(B * T, real_spatial.shape[-1],
                                         device=x.device, dtype=real_spatial.dtype)
            spatial_flat[real_mask_flat] = real_spatial
            spatial = spatial_flat.view(B, T, -1)
        else:
            spatial = self.backbone(x_flat).view(B, T, -1)

        spatial = self.channel_attention(spatial)

        # On-the-fly DCT branch
        dct_feats = self._rgb_to_dct_features(x_flat)
        freq      = self.freq_encoder(dct_feats).view(B, T, -1)

        if mask is not None:
            fm         = mask.unsqueeze(-1).float()
            freq_pooled = (freq * fm).sum(dim=1) / fm.sum(dim=1).clamp(min=1)
        else:
            freq_pooled = freq.mean(dim=1)

        projected = self.input_proj(spatial)

        # pack_padded_sequence → no gradient through padding tokens
        if mask is not None:
            lengths  = mask.sum(dim=1).clamp(min=1).cpu().long()
            packed   = torch.nn.utils.rnn.pack_padded_sequence(
                projected, lengths, batch_first=True, enforce_sorted=False)
            with torch.backends.cudnn.flags(enabled=False):
                packed_out, _ = self.temporal(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T)
        else:
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(projected)

        lstm_out = self.temporal_dropout(lstm_out)
        kpm      = ~mask if mask is not None else None
        attn_out, _ = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=kpm)
        temporal_features = self.attn_norm(lstm_out + attn_out)

        if mask is not None:
            tm = mask.unsqueeze(-1).float()
            temporal_pooled = (temporal_features * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)
        else:
            temporal_pooled = temporal_features.mean(dim=1)

        fused = torch.cat([temporal_pooled, freq_pooled], dim=-1)
        return self.classifier(fused).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL LOADING  (once at server startup)
# ═════════════════════════════════════════════════════════════════════════════

print("[NeuroPulse] Loading models...", flush=True)

# ── Ensemble ──────────────────────────────────────────────────────────────────
_ENSEMBLE    = joblib.load(os.path.join(BASE_DIR, "ensemble_model.pkl"))
W            = np.array(_ENSEMBLE["optuna_weights"])    # [0.1063, 0.3435, 0.5501]
T            = float(_ENSEMBLE["temperature_T"])         # 0.7109
THRESHOLD    = float(_ENSEMBLE["threshold_youden"])      # 0.4377
THRESHOLD_F1 = float(_ENSEMBLE["threshold_cv_f1"])       # 0.3240
MODEL_NAME   = str(_ENSEMBLE.get("ensemble_method", "optuna_multi_obj+T0.71"))
print(f"  Ensemble: W={[round(w,4) for w in W.tolist()]}  T={T:.4f}  THR={THRESHOLD:.4f}")

# ── rPPG artifacts ────────────────────────────────────────────────────────────
_RPPG_SCALER   = joblib.load(os.path.join(BASE_DIR, "rppg", "rppg_scaler.joblib"))
_RPPG_SELECTOR = joblib.load(os.path.join(BASE_DIR, "rppg", "rppg_selector.joblib"))
_RPPG_MODEL    = joblib.load(os.path.join(BASE_DIR, "rppg", "best_rppg_ml_model.joblib"))
print("  rPPG artifacts loaded")

# ── MTCNN ─────────────────────────────────────────────────────────────────────
from facenet_pytorch import MTCNN as _MTCNN
_mtcnn_224 = _MTCNN(
    image_size=224, margin=20, min_face_size=60,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=False, device=DEVICE, keep_all=False,
)
_mtcnn_299 = _MTCNN(
    image_size=299, margin=20, min_face_size=60,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=False, device=DEVICE, keep_all=False,
)
print("  MTCNN loaded")

# ── MediaPipe FaceMesh (rPPG) ─────────────────────────────────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
print("  MediaPipe FaceMesh loaded")

# ── EfficientNet-B4 ───────────────────────────────────────────────────────────
_eff_model = SpatioTemporalDeepfakeCNN(
    model_name="efficientnet_b4", pretrained=False,
    hidden_dim=256, lstm_hidden=256, lstm_layers=2,
    attention_heads=4, dropout=0.5, freeze_backbone=False,
).to(DEVICE)
_eff_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "efficientnet_model.pth"),
        map_location=DEVICE, weights_only=True,
    )
)
_eff_model.eval()
print("  EfficientNet-B4 loaded")

# ── Swin-Tiny ──────────────────────────────────────────────────────────────────
_swin_model = SpatioTemporalSwinCNN(
    model_name="swin_tiny_patch4_window7_224", pretrained=False,
    hidden_dim=192, lstm_hidden=256, num_layers=2,
    attention_heads=4, dropout=0.3, drop_path_rate=0.2, freeze_backbone=False,
).to(DEVICE)
_swin_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "swin_model.pth"),
        map_location=DEVICE, weights_only=True,
    )
)
_swin_model.eval()
print("  Swin-Tiny loaded")

# ── Xception ───────────────────────────────────────────────────────────────────
_xcep_model = SpatioTemporalBiLSTM(
    cnn_backbone="xception", pretrained=False,
    hidden_dim=256, num_layers=2, dropout=0.3,
    attention_heads=4,
).to(DEVICE)
_xcep_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "xception_model.pth"),
        map_location=DEVICE, weights_only=True,
    )
)
_xcep_model.eval()
print("  Xception loaded")

print("[NeuroPulse] All models ready.", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# FACE EXTRACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _center_crop(rgb_frame: np.ndarray, target: int) -> np.ndarray:
    h, w = rgb_frame.shape[:2]
    s    = min(h, w)
    y0   = (h - s) // 2
    x0   = (w - s) // 2
    return cv2.resize(rgb_frame[y0:y0+s, x0:x0+s], (target, target))


def _eye_align_rgb(rgb: np.ndarray, landmarks) -> np.ndarray:
    lm        = landmarks[0]
    left_eye  = lm[0]
    right_eye = lm[1]
    dy        = right_eye[1] - left_eye[1]
    dx        = right_eye[0] - left_eye[0]
    angle     = np.degrees(np.arctan2(dy, dx))
    if abs(angle) <= 2.0:
        return rgb
    eye_center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2),
    )
    h, w = rgb.shape[:2]
    M    = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)


@torch.no_grad()
def _extract_face(
    mtcnn_detector,
    rgb_frame: np.ndarray,
    img_size: int,
    use_alignment: bool = False,
) -> np.ndarray:
    from PIL import Image as _PIL_Image
    pil_img = _PIL_Image.fromarray(rgb_frame)

    try:
        if use_alignment:
            boxes, probs, lms = mtcnn_detector.detect(pil_img, landmarks=True)
            if lms is not None and len(lms) > 0 and probs[0] is not None and probs[0] >= 0.9:
                rgb_frame = _eye_align_rgb(rgb_frame, lms)
                pil_img   = _PIL_Image.fromarray(rgb_frame)

        face_tensor = mtcnn_detector(pil_img)
        if face_tensor is not None:
            # MTCNN post_process=False → raw pixels, just cast to uint8
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gray    = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= 20.0:
                return face_np
        return _center_crop(rgb_frame, img_size)
    except Exception:
        return _center_crop(rgb_frame, img_size)


def _extract_faces_from_video(
    video_path: str,
    n_frames: int,
    img_size: int,
    use_alignment: bool = False,
) -> np.ndarray:
    mtcnn  = _mtcnn_299 if img_size == 299 else _mtcnn_224
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return np.zeros((n_frames, img_size, img_size, 3), dtype=np.uint8)

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    faces   = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = _extract_face(mtcnn, rgb, img_size, use_alignment)
        faces.append(face)

    cap.release()
    return np.array(faces, dtype=np.uint8)   # (T, H, W, 3)


# ═════════════════════════════════════════════════════════════════════════════
# CNN TTA INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _build_transform(img_size: int, mean: list, std: list, n_frames: int) -> A.Compose:
    extra = {f"image{i}": "image" for i in range(1, n_frames)}
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
    ], additional_targets=extra)


def _run_cnn_tta(
    model: nn.Module,
    faces_np: np.ndarray,
    img_size: int,
    mean: list,
    std: list,
    n_tta_passes: int,
) -> float:
    """
    N-pass TTA:
      0 – standard
      1 – horizontal flip
      2 – brightness +15 (additive)
      3 – brightness −15 (additive)
      4 – Gaussian blur 3×3
      5 – 93 % centre-crop + resize
    """
    tf        = _build_transform(img_size, mean, std, len(faces_np))
    all_probs = []

    for tta_idx in range(n_tta_passes):
        frames = []
        for f in faces_np:
            aug = f.copy()
            if tta_idx == 1:
                aug = np.fliplr(aug).copy()
            elif tta_idx == 2:
                aug = np.clip(aug.astype(np.int32) + 15, 0, 255).astype(np.uint8)
            elif tta_idx == 3:
                aug = np.clip(aug.astype(np.int32) - 15, 0, 255).astype(np.uint8)
            elif tta_idx == 4:
                aug = cv2.GaussianBlur(aug, (3, 3), 0)
            elif tta_idx == 5:
                h, w   = aug.shape[:2]
                ch, cw = int(h * 0.93), int(w * 0.93)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                aug    = cv2.resize(aug[y0:y0+ch, x0:x0+cw], (w, h))
            frames.append(aug)

        kwargs = {"image": frames[0]}
        for k in range(1, len(frames)):
            kwargs[f"image{k}"] = frames[k]
        result = tf(**kwargs)

        tensors = [result["image"]] + [result[f"image{k}"] for k in range(1, len(frames))]
        tensor_list = []
        for t in tensors:
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t.transpose(2, 0, 1)).float()
            tensor_list.append(t)

        fstack = torch.stack(tensor_list).unsqueeze(0).to(DEVICE)   # (1, T, C, H, W)
        mask   = torch.ones(1, len(frames), dtype=torch.bool).to(DEVICE)

        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            logit_out = model(fstack, mask)

        prob = torch.sigmoid(logit_out.squeeze()).item()
        all_probs.append(float(prob))

    return float(np.mean(all_probs))


# ═════════════════════════════════════════════════════════════════════════════
# rPPG PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

ROI_LANDMARKS = {
    "forehead":       [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
    "left_cheek":     [50,187,123,116,143,156,70,63,105,66,107,55,65,52,53,46],
    "right_cheek":    [280,411,352,345,372,383,301,293,334,296,336,285,295,282,283,276],
    "chin":           [175,171,208,199,428,395,200,175,152,377,400,378],
    "nose":           [4,5,6,168,195,197,1,2,98,327,326,0],
    "left_jaw":       [172,136,150,149,176,148,152,377,378,400],
    "right_jaw":      [397,365,379,378,400,377,152,148,176,149,150],
    "left_forehead":  [105,66,107,55,65,52,53,46,70,63],
    "right_forehead": [334,296,336,285,295,282,283,276,301,293],
}


def _extract_roi_rgb(frame_bgr, landmarks_px, indices):
    pts = np.array(
        [[landmarks_px[i].x * frame_bgr.shape[1],
          landmarks_px[i].y * frame_bgr.shape[0]]
         for i in indices if i < len(landmarks_px)],
        dtype=np.float32,
    )
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if mask.sum() < 1000:
        return None
    roi      = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    rgb      = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(float)
    mean_rgb = rgb[mask > 0].mean(axis=0)
    return mean_rgb


def _chrom_rppg(rgb_signals, fps=30.0):
    sig = np.array(rgb_signals, dtype=float)
    if len(sig) < 10:
        return np.zeros(len(sig))
    mn    = sig.mean(axis=0) + EPS
    sig_n = sig / mn
    Xs    = 3*sig_n[:,0] - 2*sig_n[:,1]
    Ys    = 1.5*sig_n[:,0] + sig_n[:,1] - 1.5*sig_n[:,2]
    rppg  = Xs - (Xs.std() + EPS) / (Ys.std() + EPS) * Ys
    nyq   = fps / 2
    lo, hi = 0.7 / nyq, min(4.0 / nyq, 0.99)
    try:
        b, a = butter(3, [lo, hi], btype="band")
        rppg = filtfilt(b, a, rppg)
    except Exception:
        pass
    return rppg


def _extract_117_features(roi_signals_dict, fps=30.0):
    features     = []
    roi_names    = list(roi_signals_dict.keys())
    rppg_signals = {}

    for roi_name, rgb_seq in roi_signals_dict.items():
        rppg = _chrom_rppg(rgb_seq, fps)
        rppg_signals[roi_name] = rppg

        if len(rppg) < 5:
            features.extend([0.0] * 10)
            continue

        freqs, psd = welch(rppg, fs=fps, nperseg=min(len(rppg), 64), nfft=1024)
        hr_mask    = (freqs >= 0.7) & (freqs <= 4.0)
        psd_hr     = psd[hr_mask]

        snr      = float(psd_hr.sum() / (psd.sum() + EPS))
        purity   = float(psd_hr.max() / (psd_hr.sum() + EPS))
        entropy  = float(-np.sum((psd_hr / (psd_hr.sum() + EPS)) *
                                  np.log(psd_hr / (psd_hr.sum() + EPS) + EPS)))
        dom_f    = float(freqs[hr_mask][psd_hr.argmax()]) if psd_hr.size > 0 else 0.0
        centroid = float(np.sum(freqs[hr_mask] * psd_hr) / (psd_hr.sum() + EPS))
        rmssd    = float(np.sqrt(np.mean(np.diff(rppg)**2)))
        sdnn     = float(rppg.std())
        energy   = float(np.sum(rppg**2))
        crest    = float(np.abs(rppg).max() / (np.sqrt(np.mean(rppg**2)) + EPS))
        bpm      = dom_f * 60

        features.extend([snr, purity, entropy, dom_f, centroid, rmssd, sdnn, energy, crest, bpm])

    rppg_list = [rppg_signals[r] for r in roi_names]
    min_len   = min(len(s) for s in rppg_list)
    if min_len >= 5:
        for ii in range(len(rppg_list)):
            for jj in range(ii+1, len(rppg_list)):
                a, b = rppg_list[ii][:min_len], rppg_list[jj][:min_len]
                corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > EPS and b.std() > EPS else 0.0
                features.append(corr)
    else:
        features.extend([0.0] * 36)

    features = features[:117]
    while len(features) < 117:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


def predict_rppg(video_path: str, max_frames: int = 60) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = np.linspace(0, max(total - 1, 0), min(max_frames, total), dtype=int)

    roi_rgb_seqs = {k: [] for k in ROI_LANDMARKS}

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 10:
            continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            continue
        lms = result.multi_face_landmarks[0].landmark
        for roi_name, roi_indices in ROI_LANDMARKS.items():
            mean_rgb = _extract_roi_rgb(frame, lms, roi_indices)
            if mean_rgb is not None:
                roi_rgb_seqs[roi_name].append(mean_rgb)

    cap.release()

    valid_rois = {k: v for k, v in roi_rgb_seqs.items() if len(v) >= 10}
    if len(valid_rois) < 3:
        return 0.5

    features          = _extract_117_features(valid_rois, fps=fps)
    features_scaled   = _RPPG_SCALER.transform(features.reshape(1, -1))
    features_selected = _RPPG_SELECTOR.transform(features_scaled)
    prob              = float(_RPPG_MODEL.predict_proba(features_selected)[0, 1])
    return float(np.clip(prob, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# CNN MODEL WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

def predict_efficientnet(video_path: str) -> float:
    faces = _extract_faces_from_video(video_path, n_frames=16, img_size=224, use_alignment=False)
    return _run_cnn_tta(_eff_model, faces, img_size=224,
                        mean=IMAGENET_MEAN, std=IMAGENET_STD, n_tta_passes=5)


def predict_swin(video_path: str) -> float:
    faces = _extract_faces_from_video(video_path, n_frames=16, img_size=224, use_alignment=True)
    return _run_cnn_tta(_swin_model, faces, img_size=224,
                        mean=IMAGENET_MEAN, std=IMAGENET_STD, n_tta_passes=6)


def predict_xception(video_path: str) -> float:
    faces = _extract_faces_from_video(video_path, n_frames=16, img_size=299, use_alignment=True)
    return _run_cnn_tta(_xcep_model, faces, img_size=299,
                        mean=XCEPTION_MEAN, std=XCEPTION_STD, n_tta_passes=6)


# ═════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════

def run_ensemble(p_rppg: float, p_eff: float, p_swin: float) -> tuple[float, str, float]:
    probs = np.array([p_rppg, p_eff, p_swin]).clip(EPS, 1 - EPS)
    p_raw = float(W @ probs)

    if abs(T - 1.0) > 1e-4:
        p_final = float(expit(logit(np.clip(p_raw, EPS, 1 - EPS)) / T))
    else:
        p_final = p_raw

    is_fake    = p_final >= THRESHOLD
    confidence = p_final if is_fake else (1.0 - p_final)
    label      = "FAKE" if is_fake else "REAL"

    return p_final, label, round(confidence * 100, 2)


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_NAME}


@app.post("/api/v1/predict")
async def predict_endpoint(video: UploadFile = File(...)):
    ext = (video.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in {"mp4", "avi", "mov", "mkv", "webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"video.{ext}")
    try:
        content = await video.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        p_rppg = predict_rppg(tmp_path)
        p_eff  = predict_efficientnet(tmp_path)
        p_swin = predict_swin(tmp_path)
        p_xcep = predict_xception(tmp_path)   # display only — not in ensemble

        p_final, label, confidence = run_ensemble(p_rppg, p_eff, p_swin)

        return {
            "probability":     round(p_final, 4),
            "label":           label,
            "prediction":      1 if label == "FAKE" else 0,
            "confidence":      confidence,
            "model_name":      MODEL_NAME,
            "P_rPPG":          round(p_rppg,  4),
            "P_efficientnet":  round(p_eff,   4),
            "P_xception":      round(p_xcep,  4),
            "P_swin":          round(p_swin,  4),
            "p_fake":          round(p_final, 4),
            "threshold":       THRESHOLD,
            "threshold_f1":    THRESHOLD_F1,
            "weights":         [round(w, 4) for w in W.tolist()],
            "temperature_T":   round(T, 4),
            "ensemble_inputs": ["rppg", "efficientnet", "swin"],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=False)