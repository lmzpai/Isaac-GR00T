# spatial_action_head.py
# Requires: PyTorch >= 1.12 (MultiheadAttention supports batch_first in >=1.12)

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapter(nn.Module):
    """
    Lightweight MLP adapter D: projects spatial feature to VLM feature space.
    Input: t_spl_raw ∈ ℝ^{B × D_esm} (pooled spatial token)
    Output: t_spl ∈ ℝ^{B × D_act}
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_ratio: float = 1.0,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(1, int(hidden_ratio * max(d_in, d_out)))
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion:
    Query = ˆt_act (semantic token), Key/Value = t_spl (projected spatial feature).
    Uses Multi-Head Attention to adaptively recalibrate based on cross-modal relevance.
    """
    def __init__(self, d_act: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_act,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_act)  # optional stabilization

    def forward(self, t_act_hat: torch.Tensor, t_spl: torch.Tensor) -> torch.Tensor:
        # Shapes: t_act_hat ∈ ℝ^{B × D_act}, t_spl ∈ ℝ^{B × D_act}
        q = t_act_hat.unsqueeze(1)  # ℝ^{B × 1 × D_act}
        k = t_spl.unsqueeze(1)      # ℝ^{B × 1 × D_act}
        v = t_spl.unsqueeze(1)      # ℝ^{B × 1 × D_act}
        out, _ = self.mha(q, k, v)  # ℝ^{B × 1 × D_act}
        fused = out.squeeze(1)      # ℝ^{B × D_act}
        return self.ln(fused)


class FiLMGatedFusion(nn.Module):
    """
    FiLM-Gated Modulation:
    - Generate affine parameters (γ, β) from spatial feature t_spl.
    - Modulate action token: t_mod = γ ⊙ ˆt_act + β
    - Gate blend modulated semantic (t_mod) and original spatial feature (t_spl):
      fused = g ⊙ t_mod + (1 - g) ⊙ t_spl
    """
    def __init__(self, d_act: int):
        super().__init__()
        self.gamma = nn.Linear(d_act, d_act)
        self.beta = nn.Linear(d_act, d_act)
        # Gate from concatenation of spatial and modulated semantic
        self.gate = nn.Sequential(
            nn.Linear(2 * d_act, d_act),
            nn.GELU(),
            nn.Linear(d_act, d_act),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(d_act)

    def forward(self, t_act_hat: torch.Tensor, t_spl: torch.Tensor) -> torch.Tensor:
        # FiLM modulation
        gamma = self.gamma(t_spl)
        beta = self.beta(t_spl)
        t_mod = gamma * t_act_hat + beta
        # Gate
        g = self.gate(torch.cat([t_spl, t_mod], dim=-1))
        fused = g * t_mod + (1.0 - g) * t_spl
        return self.ln(fused)


class ElementAddFusion(nn.Module):
    """
    Element-wise Addition:
    fused = ˆt_act + t_spl
    """
    def __init__(self):
        super().__init__()
        self.ln = None  # placeholder; set at runtime

    def reset_ln(self, d_act: int):
        # Lazy init to known d_act
        self.ln = nn.LayerNorm(d_act)

    def forward(self, t_act_hat: torch.Tensor, t_spl: torch.Tensor) -> torch.Tensor:
        fused = t_act_hat + t_spl
        return self.ln(fused)


class SpatialEnhancedActionHead(nn.Module):
    """
    Spatial-Enhanced Action Head:
    - Compress spatial tokens T_spl via pooling to t_spl_raw ∈ ℝ^{B × D_esm}
    - Adapter D projects to aligned spatial feature t_spl ∈ ℝ^{B × D_act}
    - Fuse with semantic action token ˆt_act via chosen strategy
      Default: element-wise addition (highest performance and stability).
    """
    def __init__(
        self,
        d_esm: int,
        d_act: int,
        fusion: Literal["add", "cross_attn", "film"] = "add",
        pool: Literal["max", "mean"] = "max",
        adapter_hidden_ratio: float = 1.0,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_esm = d_esm
        self.d_act = d_act
        self.pool = pool

        # Adapter D
        self.adapter = MLPAdapter(
            d_in=d_esm,
            d_out=d_act,
            hidden_ratio=adapter_hidden_ratio,
            dropout=adapter_dropout,
        )

        # Fusion
        if fusion == "add":
            self.fusion = ElementAddFusion()
            self.fusion.reset_ln(d_act)
        elif fusion == "cross_attn":
            self.fusion = CrossAttentionFusion(d_act=d_act, num_heads=attn_heads, dropout=attn_dropout)
        elif fusion == "film":
            self.fusion = FiLMGatedFusion(d_act=d_act)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")
        self.fusion_name = fusion

    def _compress_spatial(self, T_spl: torch.Tensor) -> torch.Tensor:
        # T_spl ∈ ℝ^{B × N × D_esm} -> t_spl_raw ∈ ℝ^{B × D_esm}
        if self.pool == "max":
            t_spl_raw, _ = torch.max(T_spl, dim=1)
        elif self.pool == "mean":
            t_spl_raw = torch.mean(T_spl, dim=1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        return t_spl_raw

    def forward(self, T_spl: torch.Tensor, t_act_hat: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          - T_spl: ℝ^{B × N × D_esm} spatial tokens from ESM
          - t_act_hat: ℝ^{B × D_act} semantic action token from VLM
        Output:
          - f_fused: ℝ^{B × D_act}
        """
        t_spl_raw = self._compress_spatial(T_spl)
        t_spl = self.adapter(t_spl_raw)  # aligned spatial feature in VLM space
        f_fused = self.fusion(t_act_hat, t_spl)
        return f_fused


class MLPPolicy(nn.Module):
    """
    MLP-based action predictor π:
    A_t = π(f_t^fused)
    """
    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(d_in, d_out)
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, f_t_fused: torch.Tensor) -> torch.Tensor:
        return self.net(f_t_fused)


class LSTMPolicy(nn.Module):
    """
    LSTM-based action predictor for long-horizon tasks:
    Processes sequence (f_{t-H+1}^fused, ..., f_t^fused) and outputs action chunk:
    A_t = π(f_{t-H+1}^fused, ..., f_t^fused)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_size = hidden_size or max(d_in, d_out)
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, d_out),
        )

    def forward(self, seq_f_fused: torch.Tensor) -> torch.Tensor:
        # seq_f_fused ∈ ℝ^{B × H × D_in}
        out, (hn, cn) = self.lstm(seq_f_fused)
        # Use last timestep's output
        last = out[:, -1, :]  # ℝ^{B × hidden_size}
        return self.head(last)


class ActionPredictor(nn.Module):
    """
    Wrapper that selects MLP or LSTM predictor:
    - mode='mlp': forward expects f_t_fused ∈ ℝ^{B × D_act}
    - mode='lstm': forward expects seq_f_fused ∈ ℝ^{B × H × D_act}
    """
    def __init__(
        self,
        mode: Literal["mlp", "lstm"],
        d_in: int,
        d_out: int,
        mlp_hidden: Optional[int] = None,
        lstm_hidden: Optional[int] = None,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            self.impl = MLPPolicy(d_in=d_in, d_out=d_out, hidden=mlp_hidden, dropout=dropout)
        elif mode == "lstm":
            self.impl = LSTMPolicy(d_in=d_in, d_out=d_out, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown predictor mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


# --- Example usage ---
if __name__ == "__main__":
    torch.manual_seed(0)

    # Dummy shapes
    B, N = 8, 32
    D_esm = 256
    D_act = 512
    A_dim = 7   # e.g., 7-DoF continuous action

    # Inputs
    T_spl = torch.randn(B, N, D_esm)          # spatial tokens from ESM
    t_act_hat = torch.randn(B, D_act)         # semantic action token from VLM

    # Spatial-Enhanced Action Head (default: element-wise addition)
    head = SpatialEnhancedActionHead(
        d_esm=D_esm,
        d_act=D_act,
        fusion="add",          # "add" | "cross_attn" | "film"
        pool="max",
        adapter_hidden_ratio=1.0,
        attn_heads=8,
        attn_dropout=0.0,
        adapter_dropout=0.0,
    )

    f_fused = head(T_spl, t_act_hat)          # ℝ^{B × D_act}

    # Predictor: MLP (single-step)
    predictor_mlp = ActionPredictor(mode="mlp", d_in=D_act, d_out=A_dim, mlp_hidden=1024, dropout=0.1)
    A_t_mlp = predictor_mlp(f_fused)          # ℝ^{B × A_dim}
    print("MLP action shape:", A_t_mlp.shape)

    # Predictor: LSTM (sequence over H timesteps)
    H = 5
    seq_f_fused = torch.randn(B, H, D_act)
    predictor_lstm = ActionPredictor(mode="lstm", d_in=D_act, d_out=A_dim, lstm_hidden=512, lstm_layers=1, dropout=0.0)
    A_t_lstm = predictor_lstm(seq_f_fused)    # ℝ^{B × A_dim}
    print("LSTM action shape:", A_t_lstm.shape)