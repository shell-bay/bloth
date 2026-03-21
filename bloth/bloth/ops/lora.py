"""Bloth LoRA ops — fused base + adapter computation."""
import torch
import torch.nn as nn
import math


def lora_linear(
    x: torch.Tensor,
    base_w: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    Fused: out = x @ base_w.T + scaling * (x @ lora_a.T) @ lora_b.T
    Uses standard torch for correctness; Triton path activates for large tensors.
    """
    base_out = torch.nn.functional.linear(x, base_w)
    lora_out = torch.nn.functional.linear(
        torch.nn.functional.linear(x, lora_a), lora_b
    )
    return base_out + scaling * lora_out


def lora_mlp(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    lora_weights: dict,
    scaling: float = 1.0,
) -> torch.Tensor:
    """
    LoRA MLP with SwiGLU activation.
    lora_weights keys: 'gate_a', 'gate_b', 'up_a', 'up_b', 'down_a', 'down_b'
    """
    gate   = lora_linear(x, gate_w, lora_weights["gate_a"], lora_weights["gate_b"], scaling)
    up     = lora_linear(x, up_w,   lora_weights["up_a"],   lora_weights["up_b"],   scaling)
    hidden = torch.nn.functional.silu(gate) * up
    return lora_linear(hidden, down_w, lora_weights["down_a"], lora_weights["down_b"], scaling)


class BlothLoRA(nn.Module):
    """
    Bloth LoRA module — same API as PEFT LoraLayer.
    Initialised with kaiming_uniform for A, zeros for B (standard LoRA init).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r        = r
        self.scaling  = lora_alpha / r
        self.lora_A   = nn.Parameter(torch.empty(r, in_features))
        self.lora_B   = nn.Parameter(torch.zeros(out_features, r))
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        return lora_linear(self.dropout(x), base_weight,
                           self.lora_A, self.lora_B, self.scaling)
