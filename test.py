import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMoEAttention(nn.Module):
    """
    Input:
        x: (B, D, S, F)

    Example:
        B = batch
        D = dateback
        S = number of stocks, ~5000
        F = number of features, 407

    Flow:
        (B, D, S, F)

        Cross-sectional Attention + MoE:
        -> (B*D, S, F)
        -> (B*D, S, F)

        Restore:
        -> (B, D, S, F)

        Temporal Linear:
        D -> 1
        -> (B, S, F)

        Feature Linear:
        F -> 1
        -> (B, S, 1)
    """

    def __init__(
        self,
        n_features=407,
        dateback=20,

        # Cross-sectional attention
        n_heads=4,
        head_dim=16,

        # MoE
        n_experts=4,
        top_k=2,
        expert_rank=16,

        # residual branch strength
        residual_scale_init=0.0,

        bias=True,
    ):
        super().__init__()

        self.n_features = n_features
        self.dateback = dateback

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_dim = n_heads * head_dim

        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_rank = expert_rank

        assert 1 <= top_k <= n_experts

        # ============================================================
        # 1. Cross-sectional low-rank attention
        #
        # Main representation remains 407.
        #
        # Internally:
        #   407 -> attn_dim
        #
        # Example:
        #   407 -> 64
        # ============================================================

        self.q_proj = nn.Linear(
            n_features,
            self.attn_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            n_features,
            self.attn_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            n_features,
            self.attn_dim,
            bias=False
        )

        self.attn_out = nn.Linear(
            self.attn_dim,
            n_features,
            bias=False
        )

        # Residual gate:
        #
        # x <- x + alpha * Attention(x)
        #
        # alpha = 0:
        # attention branch initially has zero effect.
        self.attn_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

        # ============================================================
        # 2. Router
        #
        # Each stock token:
        #   407 -> n_experts
        #
        # Example:
        #   407 -> 4
        # ============================================================

        self.router = nn.Linear(
            n_features,
            n_experts,
            bias=True
        )

        # ============================================================
        # 3. Low-rank LINEAR experts
        #
        # Each expert:
        #
        #   407 -> rank -> 407
        #
        # Example:
        #
        #   407 -> 16 -> 407
        #
        # IMPORTANT:
        # no activation here.
        #
        # Therefore each expert itself remains a linear transformation.
        #
        # expert_down:
        #   (E, F, R)
        #
        # expert_up:
        #   (E, R, F)
        # ============================================================

        self.expert_down = nn.Parameter(
            torch.empty(
                n_experts,
                n_features,
                expert_rank
            )
        )

        self.expert_up = nn.Parameter(
            torch.empty(
                n_experts,
                expert_rank,
                n_features
            )
        )

        # x <- x + beta * MoE(x)
        self.moe_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

        # ============================================================
        # 4. Temporal linear
        #
        # D -> 1
        #
        # Applied independently for:
        # every stock × every feature
        # ============================================================

        self.temporal = nn.Linear(
            dateback,
            1,
            bias=bias
        )

        # ============================================================
        # 5. Final feature linear
        #
        # 407 -> 1
        # ============================================================

        self.head = nn.Linear(
            n_features,
            1,
            bias=bias
        )

        self._init_weights()

    def _init_weights(self):

        # ------------------------------------------------------------
        # Attention projections
        # ------------------------------------------------------------
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        # Keep attention output relatively mild initially
        nn.init.xavier_uniform_(
            self.attn_out.weight,
            gain=0.5
        )

        # ------------------------------------------------------------
        # Router
        #
        # Small random initialization:
        # avoid strongly preferring one expert at initialization.
        # ------------------------------------------------------------
        nn.init.normal_(
            self.router.weight,
            mean=0.0,
            std=0.01
        )
        nn.init.zeros_(self.router.bias)

        # ------------------------------------------------------------
        # Low-rank experts
        # ------------------------------------------------------------
        for i in range(self.n_experts):
            nn.init.xavier_uniform_(
                self.expert_down[i]
            )

            nn.init.xavier_uniform_(
                self.expert_up[i],
                gain=0.5
            )

        # ------------------------------------------------------------
        # Strong linear backbone
        # ------------------------------------------------------------
        nn.init.xavier_uniform_(
            self.temporal.weight
        )

        if self.temporal.bias is not None:
            nn.init.zeros_(self.temporal.bias)

        nn.init.xavier_uniform_(
            self.head.weight
        )

        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x:
            (B, D, S, F)

        return:
            (B, S, 1)
        """

        B, D, S, feature_dim = x.shape

        assert feature_dim == self.n_features
        assert D == self.dateback

        # ============================================================
        # Cross-sectional stage
        #
        # Every historical date becomes one independent market snapshot.
        #
        # (B, D, S, F)
        # ->
        # (B*D, S, F)
        # ============================================================

        h = x.reshape(
            B * D,
            S,
            feature_dim
        )

        # ============================================================
        # 1. Cross-sectional Attention
        # ============================================================

        # (BD, S, F)
        # ->
        # (BD, S, H*Dh)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # ->
        # (BD, S, H, Dh)
        # ->
        # (BD, H, S, Dh)

        q = q.reshape(
            B * D,
            S,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            B * D,
            S,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            B * D,
            S,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        # Full cross-sectional attention.
        #
        # Mathematically each head performs:
        #
        #     (S, S)
        #
        # attention.
        #
        # output:
        #     (BD, H, S, Dh)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

        # (BD, H, S, Dh)
        # ->
        # (BD, S, H, Dh)
        # ->
        # (BD, S, H*Dh)

        attn = attn.transpose(1, 2).reshape(
            B * D,
            S,
            self.attn_dim
        )

        # ->
        # (BD, S, F)

        attn = self.attn_out(attn)

        # Residual
        h = h + self.attn_scale * attn

        # ============================================================
        # 2. MoE Router
        # ============================================================

        # (BD, S, F)
        # ->
        # (BD, S, E)

        router_logits = self.router(h)

        # Select Top-K experts for each stock token.
        #
        # top_values:
        #   (BD, S, K)
        #
        # top_indices:
        #   (BD, S, K)

        top_values, top_indices = torch.topk(
            router_logits,
            k=self.top_k,
            dim=-1
        )

        # Normalize only among selected experts.
        #
        # (BD, S, K)

        top_weights = F.softmax(
            top_values,
            dim=-1
        )

        # Convert to full expert weights:
        #
        # (BD, S, E)

        router_weights = torch.zeros_like(
            router_logits
        )

        router_weights.scatter_(
            dim=-1,
            index=top_indices,
            src=top_weights
        )

        # ============================================================
        # 3. Low-rank linear experts
        #
        # No activation.
        #
        # h:
        #   (BD, S, F)
        #
        # expert_down:
        #   (E, F, R)
        #
        # result:
        #   (BD, S, E, R)
        # ============================================================

        expert_hidden = torch.einsum(
            "bsf,efr->bser",
            h,
            self.expert_down
        )

        # (BD, S, E, R)
        # ->
        # (BD, S, E, F)

        expert_output = torch.einsum(
            "bser,erf->bsef",
            expert_hidden,
            self.expert_up
        )

        # Router weighted combination:
        #
        # (BD, S, E, F)
        # *
        # (BD, S, E, 1)
        #
        # ->
        # (BD, S, F)

        moe_output = (
            expert_output
            * router_weights.unsqueeze(-1)
        ).sum(dim=2)

        # Residual
        h = h + self.moe_scale * moe_output

        # ============================================================
        # Restore time dimension
        #
        # (BD, S, F)
        # ->
        # (B, D, S, F)
        # ============================================================

        h = h.reshape(
            B,
            D,
            S,
            feature_dim
        )

        # ============================================================
        # Temporal linear
        #
        # Need D at the final dimension:
        #
        # (B, D, S, F)
        # ->
        # (B, S, F, D)
        # ============================================================

        h = h.permute(
            0,
            2,
            3,
            1
        )

        # D -> 1
        #
        # (B, S, F, D)
        # ->
        # (B, S, F, 1)

        h = self.temporal(h)

        # ->
        # (B, S, F)

        h = h.squeeze(-1)

        # ============================================================
        # Feature linear
        #
        # F -> 1
        #
        # (B, S, F)
        # ->
        # (B, S, 1)
        # ============================================================

        y = self.head(h)

        return y
