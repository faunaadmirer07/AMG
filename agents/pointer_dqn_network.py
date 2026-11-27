"""
Pointer Q-Network — Flash-Attention dueling Q-network for variable-size BB sets.

Architecture:
  1. bb_encoder:   BB_feats [N, bb_dim] → bb_h [N, H]       (3-layer MLP)
  2. bb_self_attn: bb_h [N, H] → bb_h [N, H]                (multi-head self-attention via
                   F.scaled_dot_product_attention — O(N) memory, Flash kernel on CUDA)
                   Each BB attends to all selectable BBs — learns relative ranking.
  3. ctx_encoder:  context [ctx_dim] → ctx_h [H]            (2-layer MLP)
  4. Dueling heads (per-BB):
       V(s)       = value_head(ctx_h)                        scalar
       A(s, BB_i) = advantage_head(cat[bb_h_i, ctx_h])      per-BB scalar
       Q(s, BB_i) = V(s) + A(s, BB_i) - mean_j A(s, BB_j)

mask convention: True = selectable/real, False = already-selected or padding.
Non-selectable slots get Q = -inf so argmax always picks a valid BB.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerQNetwork(nn.Module):

    def __init__(self, bb_dim=30, ctx_dim=9, hidden_dim=256, use_dueling=True, num_attn_heads=4):
        super().__init__()
        assert hidden_dim % num_attn_heads == 0, "hidden_dim must be divisible by num_attn_heads"
        self.hidden_dim  = hidden_dim
        self.use_dueling = use_dueling
        self.num_heads   = num_attn_heads
        self.head_dim    = hidden_dim // num_attn_heads

        self.bb_encoder = nn.Sequential(
            nn.Linear(bb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Flash Attention projections (no bias — standard practice)
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if use_dueling:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, bb_feats, context, mask=None):
        """
        Args:
            bb_feats : [B, N, bb_dim]  all eligible BBs (fixed N per episode)
            context  : [B, ctx_dim]
            mask     : [B, N] bool — True = selectable, False = already-selected/padding

        Returns:
            q_values : [B, N]  Q-value per BB; non-selectable slots = -inf
        """
        B, N, _ = bb_feats.shape
        H        = self.hidden_dim

        bb_h = self.bb_encoder(bb_feats.view(B * N, -1)).view(B, N, H)  # [B, N, H]

        # Flash-Attention self-attention over BB embeddings
        # Q/K/V: [B, heads, N, head_dim]
        q = self.q_proj(bb_h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(bb_h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(bb_h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Additive attention bias: 0 for attend, -inf for ignore
        # mask [B, N] True=attend → bias [B, 1, 1, N] (broadcast over heads & query pos)
        if mask is not None:
            attn_bias = torch.zeros(B, 1, 1, N, device=bb_feats.device, dtype=bb_feats.dtype)
            attn_bias = attn_bias.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        else:
            attn_bias = None

        # Use the built-in attention path when available; fall back to a
        # manual implementation on older Torch versions.
        if hasattr(F, 'scaled_dot_product_attention'):
            bb_h_att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # [B, heads, N, hd]
        else:
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                attn_scores = attn_scores + attn_bias
            attn_weights = torch.softmax(attn_scores, dim=-1)
            bb_h_att = torch.matmul(attn_weights, v)
        bb_h_att = bb_h_att.transpose(1, 2).contiguous().view(B, N, H)           # [B, N, H]
        bb_h_att = self.out_proj(bb_h_att)
        bb_h     = self.attn_norm(bb_h + bb_h_att)                               # residual + LN

        ctx_h    = self.ctx_encoder(context)                                       # [B, H]
        ctx_exp  = ctx_h.unsqueeze(1).expand(B, N, -1)                           # [B, N, H]
        combined = torch.cat([bb_h, ctx_exp], dim=-1)                            # [B, N, 2H]

        if self.use_dueling:
            V = self.value_head(ctx_h)                                            # [B, 1]
            A = self.advantage_head(combined).squeeze(-1)                         # [B, N]
            if mask is not None:
                A_valid = A.masked_fill(~mask, 0.0)
                n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                A_mean  = A_valid.sum(dim=1, keepdim=True) / n_valid
            else:
                A_mean = A.mean(dim=1, keepdim=True)
            Q = V + (A - A_mean)
        else:
            Q = self.q_head(combined).squeeze(-1)                                 # [B, N]

        if mask is not None:
            Q = Q.masked_fill(~mask, float('-inf'))

        return Q
