# app/models/fancy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.registry import register_models


# ============================================================
# Cross-Section Modules
# ============================================================

class CrossSectionIdentity(nn.Module):
    """
    Identity cross-section module.

    Input:
        x: [B, D, S, C]

    Output:
        x: [B, D, S, C]
    """
    def forward(self, x, adj=None):
        return x


class StockSelfAttention(nn.Module):
    """
    Full self-attention across stock dimension S.

    For each day independently:

        [B, D, S, C]
            ->
        [B*D, S, C]
            ->
        Self-Attention over S
            ->
        [B, D, S, C]

    Attention does NOT change feature dimension C.
    """
    def __init__(
        self,
        dim: int,
        heads: int=4,
        dropout: float=0.0,
    ):
        super().__init__()

        assert dim % heads == 0, \
            f"dim={dim} must be divisible by heads={heads}"

        self.norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        """
        Args:
            x: [B, D, S, C]

        Returns:
            x: [B, D, S, C]
        """
        B, D, S, C = x.shape

        residual = x

        x = self.norm(x)
        x = x.reshape(B * D, S, C)

        x, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
        )

        x = x.reshape(B, D, S, C)

        return residual + self.dropout(x)


class GraphConvBlock(nn.Module):
    """
    Simple GCN-style cross-section block.

    Equation:

        H' = A_hat @ H @ W

    where A_hat should already contain:
        - self-loop
        - normalization

    adj can be either:
        dense Tensor:  [S, S]
        sparse Tensor: [S, S]

    Input:
        x: [B, D, S, C]

    Output:
        x: [B, D, S, C]
    """
    def __init__(
        self,
        dim: int,
        dropout: float=0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        if adj is None:
            raise ValueError(
                "adj must be provided when cs_type='gnn'."
            )

        B, D, S, C = x.shape

        if adj.shape != (S, S):
            raise ValueError(
                f"adj shape must be ({S}, {S}), "
                f"but got {tuple(adj.shape)}."
            )

        residual = x

        # PreNorm + feature projection
        x = self.norm(x)
        x = self.proj(x)

        # ----------------------------------------------------
        # [B,D,S,C]
        # ->
        # [S,B,D,C]
        # ->
        # [S,B*D*C]
        #
        # This lets sparse.mm process all B,D,C together,
        # avoiding Python loops over B or D.
        # ----------------------------------------------------
        x = x.permute(2, 0, 1, 3)
        x = x.reshape(S, -1)

        if adj.layout == torch.strided:
            x = adj @ x
        else:
            x = torch.sparse.mm(adj, x)

        x = x.reshape(S, B, D, C)
        x = x.permute(1, 2, 0, 3)

        x = F.silu(x)

        return residual + self.dropout(x)


def get_cross_section_module(
    cs_type: str,
    dim: int,
    **kwargs,
):
    """
    Cross-section module factory.

    cs_type:
        "attention"
        "gnn"
        "none"
    """
    if cs_type == "attention":
        return StockSelfAttention(
            dim=dim,
            **kwargs,
        )

    elif cs_type == "gnn":
        return GraphConvBlock(
            dim=dim,
            **kwargs,
        )

    elif cs_type == "none":
        return CrossSectionIdentity()

    else:
        raise ValueError(
            f"Unknown cs_type: {cs_type}, "
            f"choose from ['attention', 'gnn', 'none']."
        )


# ============================================================
# Market-Level MoE
# ============================================================

class MarketMoE(nn.Module):
    """
    Market-level Mixture-of-Experts.

    Instead of routing every stock independently,
    first summarize the whole market cross-section:

        Mean_S(H)
        Std_S(H)

    Then:

        [B,D,2C]
            ->
        Router
            ->
        [B,D,E]

    All stocks on the same day share the same expert weights.

    Input:
        x: [B, D, S, C]

    Output:
        x:     [B, D, S, C]
        gates: [B, D, E]
    """
    def __init__(
        self,
        dim: int,
        num_experts: int=4,
        expert_hidden_dim: int=64,
        top_k: int=None,
        dropout: float=0.0,
    ):
        super().__init__()

        assert num_experts > 0

        if top_k is not None:
            assert 0 < top_k <= num_experts

        self.dim = dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.top_k = top_k

        self.norm = nn.LayerNorm(dim)

        # market state:
        #
        # mean [C]
        # std  [C]
        #
        # -> [2C]
        self.router = nn.Linear(
            dim * 2,
            num_experts,
        )

        # ----------------------------------------------------
        # Vectorized experts
        #
        # Instead of:
        #
        #   for expert in experts:
        #       ...
        #
        # use batched expert parameters.
        #
        # W1: [E, C, H]
        # W2: [E, H, C]
        # ----------------------------------------------------
        self.w1 = nn.Parameter(
            torch.empty(
                num_experts,
                dim,
                expert_hidden_dim,
            )
        )

        self.b1 = nn.Parameter(
            torch.zeros(
                num_experts,
                expert_hidden_dim,
            )
        )

        self.w2 = nn.Parameter(
            torch.empty(
                num_experts,
                expert_hidden_dim,
                dim,
            )
        )

        self.b2 = nn.Parameter(
            torch.zeros(
                num_experts,
                dim,
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, D, S, C]

        Returns:
            x:
                [B, D, S, C]

            gates:
                [B, D, E]
        """
        residual = x

        x = self.norm(x)

        # ----------------------------------------------------
        # Market-level state
        # ----------------------------------------------------
        market_mean = x.mean(
            dim=2
        )  # [B,D,C]

        market_std = x.std(
            dim=2,
            unbiased=False,
        )  # [B,D,C]

        market_state = torch.cat(
            [
                market_mean,
                market_std,
            ],
            dim=-1,
        )  # [B,D,2C]

        router_logits = self.router(
            market_state
        )  # [B,D,E]

        # ----------------------------------------------------
        # Dense MoE by default.
        #
        # If top_k is given:
        #     keep only Top-K experts.
        # ----------------------------------------------------
        if (
            self.top_k is not None
            and self.top_k < self.num_experts
        ):
            top_value, top_index = torch.topk(
                router_logits,
                k=self.top_k,
                dim=-1,
            )

            sparse_logits = torch.full_like(
                router_logits,
                float("-inf"),
            )

            sparse_logits.scatter_(
                -1,
                top_index,
                top_value,
            )

            gates = torch.softmax(
                sparse_logits,
                dim=-1,
            )

        else:
            gates = torch.softmax(
                router_logits,
                dim=-1,
            )

        # ----------------------------------------------------
        # Expert layer 1
        #
        # x:
        # [B,D,S,C]
        #
        # w1:
        # [E,C,H]
        #
        # ->
        # [B,D,S,E,H]
        # ----------------------------------------------------
        expert_hidden = torch.einsum(
            "bdsc,ech->bdseh",
            x,
            self.w1,
        )

        expert_hidden = (
            expert_hidden
            + self.b1[None, None, None, :, :]
        )

        expert_hidden = F.silu(
            expert_hidden
        )

        expert_hidden = self.dropout(
            expert_hidden
        )

        # ----------------------------------------------------
        # Expert layer 2
        #
        # [B,D,S,E,H]
        # ->
        # [B,D,S,E,C]
        # ----------------------------------------------------
        expert_out = torch.einsum(
            "bdseh,ehc->bdsec",
            expert_hidden,
            self.w2,
        )

        expert_out = (
            expert_out
            + self.b2[None, None, None, :, :]
        )

        # ----------------------------------------------------
        # Mixture
        #
        # gates:
        # [B,D,E]
        #
        # ->
        # [B,D,1,E,1]
        # ----------------------------------------------------
        x = (
            expert_out
            * gates[:, :, None, :, None]
        ).sum(
            dim=3
        )

        return (
            residual + self.dropout(x),
            gates,
        )


# ============================================================
# Temporal Module
# ============================================================

class TemporalConvBlock(nn.Module):
    """
    Lightweight temporal TCN.

    Only operates along D dimension.

    Input:
        x: [B, D, S, C]

    Internal:
        [B,D,S,C]
            ->
        [B*S,C,D]
            ->
        causal depthwise Conv1D
            ->
        pointwise Conv1D
            ->
        [B,D,S,C]

    No cross-stock mixing happens here.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int=3,
        dilation: int=1,
        dropout: float=0.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        self.norm = nn.LayerNorm(dim)

        # One temporal filter for each latent feature
        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
            padding=0,
        )

        # Mix latent feature channels
        self.pointwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, D, S, C]

        Returns:
            x: [B, D, S, C]
        """
        B, D, S, C = x.shape

        residual = x

        x = self.norm(x)

        # [B,D,S,C]
        # ->
        # [B,S,C,D]
        # ->
        # [B*S,C,D]
        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = x.reshape(
            B * S,
            C,
            D,
        )

        # causal padding
        left_padding = (
            self.dilation
            * (self.kernel_size - 1)
        )

        x = F.pad(
            x,
            (
                left_padding,
                0,
            ),
        )

        x = self.depthwise(x)
        x = self.pointwise(x)

        x = F.silu(x)
        x = self.dropout(x)

        # [B*S,C,D]
        # ->
        # [B,S,C,D]
        # ->
        # [B,D,S,C]
        x = x.reshape(
            B,
            S,
            C,
            D,
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        return residual + x


class TemporalIdentity(nn.Module):
    def forward(self, x):
        return x


def get_temporal_module(
    temporal_type: str,
    dim: int,
    **kwargs,
):
    if temporal_type == "tcn":
        return TemporalConvBlock(
            dim=dim,
            **kwargs,
        )

    elif temporal_type == "none":
        return TemporalIdentity()

    else:
        raise ValueError(
            f"Unknown temporal_type: {temporal_type}, "
            f"choose from ['tcn', 'none']."
        )


# ============================================================
# Main Model
# ============================================================

@register_models("fancy")
class FancyModel(nn.Module):
    """
    Daily Stock Prediction Model.

    Main architecture:

        Feature Encoder
            ->
        Cross-Section Module
            ->
        Market MoE
            ->
        Temporal Module
            ->
        Prediction Head

    plus strong linear backbone:

        y = y_linear + alpha * y_fancy


    ========================================================
    Input
    ========================================================

        x:
            [B, D, S, F]

        B:
            batch size

        D:
            lookback days

        S:
            number of stocks

        F:
            feature dimension


    ========================================================
    Shape Flow
    ========================================================

        [B,D,S,407]

            ↓ Feature Encoder

        [B,D,S,C]

            ↓ Attention / GNN

        [B,D,S,C]

            ↓ Market MoE

        [B,D,S,C]

            ↓ Temporal TCN

        [B,D,S,C]

            ↓ last day

        [B,S,C]

            ↓ MLP Head

        [B,S,1]


    Linear branch:

        x[:, -1]
        [B,S,407]

            ↓

        Linear(407,1)

            ↓

        [B,S,1]
    """
    def __init__(
        self,

        # -----------------------------------------
        # Input
        # -----------------------------------------
        feature_dim: int=407,
        hidden_dim: int=32,

        # -----------------------------------------
        # Cross Section
        # -----------------------------------------
        cs_type: str="attention",
        cs_kwargs: dict=None,

        # -----------------------------------------
        # MoE
        # -----------------------------------------
        use_moe: bool=True,
        num_experts: int=4,
        expert_hidden_dim: int=64,
        top_k: int=None,
        moe_dropout: float=0.0,

        # -----------------------------------------
        # Temporal
        # -----------------------------------------
        temporal_type: str="tcn",
        temporal_kernel_size: int=3,
        temporal_dilation: int=1,
        temporal_dropout: float=0.0,

        # -----------------------------------------
        # Head
        # -----------------------------------------
        head_hidden_dim: int=16,
        head_dropout: float=0.0,

        # -----------------------------------------
        # Linear backbone
        # -----------------------------------------
        use_linear_backbone: bool=True,
        freeze_linear_backbone: bool=False,
        residual_scale_init: float=0.1,

        # -----------------------------------------
        # Init
        # -----------------------------------------
        init: str="xavier",
    ):
        super().__init__()

        cs_kwargs = cs_kwargs or {}

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.cs_type = cs_type
        self.temporal_type = temporal_type

        self.use_moe = use_moe
        self.use_linear_backbone = (
            use_linear_backbone
        )

        # ====================================================
        # 1. Feature Encoder
        #
        # 407 -> C
        # ====================================================
        self.encoder = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.SiLU(),
        )

        # ====================================================
        # 2. Cross-Section
        #
        # attention / gnn / none
        # ====================================================
        self.cross_section = (
            get_cross_section_module(
                cs_type=cs_type,
                dim=hidden_dim,
                **cs_kwargs,
            )
        )

        # ====================================================
        # 3. Market MoE
        # ====================================================
        if use_moe:
            self.moe = MarketMoE(
                dim=hidden_dim,
                num_experts=num_experts,
                expert_hidden_dim=(
                    expert_hidden_dim
                ),
                top_k=top_k,
                dropout=moe_dropout,
            )

        else:
            self.moe = None

        # ====================================================
        # 4. Temporal
        # ====================================================
        self.temporal = (
            get_temporal_module(
                temporal_type=temporal_type,
                dim=hidden_dim,
                kernel_size=(
                    temporal_kernel_size
                ),
                dilation=(
                    temporal_dilation
                ),
                dropout=(
                    temporal_dropout
                ),
            )
            if temporal_type == "tcn"
            else TemporalIdentity()
        )

        # ====================================================
        # 5. Prediction Head
        #
        # C -> head_hidden -> 1
        # ====================================================
        self.head = nn.Sequential(
            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                head_hidden_dim,
            ),

            nn.SiLU(),

            nn.Dropout(
                head_dropout
            ),

            nn.Linear(
                head_hidden_dim,
                1,
            ),
        )

        # ====================================================
        # 6. Strong Linear Backbone
        # ====================================================
        if use_linear_backbone:
            self.linear_backbone = nn.Linear(
                feature_dim,
                1,
            )

            if freeze_linear_backbone:
                for p in (
                    self.linear_backbone.parameters()
                ):
                    p.requires_grad = False

            # ------------------------------------------------
            # Do NOT initialize exactly at zero.
            #
            # alpha=0 would block gradient from entering
            # the nonlinear branch at the beginning.
            # ------------------------------------------------
            self.residual_scale = nn.Parameter(
                torch.tensor(
                    float(residual_scale_init)
                )
            )

        else:
            self.linear_backbone = None
            self.residual_scale = None

        self._init_weights(init)

    # ========================================================
    # Initialization
    # ========================================================

    def _init_weights(self, init: str):
        for m in self.modules():

            if isinstance(
                m,
                (nn.Linear, nn.Conv1d),
            ):
                if init == "xavier":
                    nn.init.xavier_uniform_(
                        m.weight
                    )

                elif init == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        nonlinearity="relu",
                    )

                elif init == "normal":
                    nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=0.02,
                    )

                else:
                    raise ValueError(
                        f"Unknown init type: {init}."
                    )

                if m.bias is not None:
                    nn.init.zeros_(
                        m.bias
                    )

            elif isinstance(
                m,
                nn.MultiheadAttention,
            ):
                if init == "xavier":
                    nn.init.xavier_uniform_(
                        m.in_proj_weight
                    )

                elif init == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.in_proj_weight,
                        nonlinearity="linear",
                    )

                elif init == "normal":
                    nn.init.normal_(
                        m.in_proj_weight,
                        mean=0.0,
                        std=0.02,
                    )

                if m.in_proj_bias is not None:
                    nn.init.zeros_(
                        m.in_proj_bias
                    )

        # ----------------------------------------------------
        # Vectorized MoE weights are raw nn.Parameter,
        # so initialize them separately.
        # ----------------------------------------------------
        if self.use_moe:

            if init == "xavier":
                nn.init.xavier_uniform_(
                    self.moe.w1
                )

                nn.init.xavier_uniform_(
                    self.moe.w2
                )

            elif init == "kaiming":
                nn.init.kaiming_uniform_(
                    self.moe.w1,
                    nonlinearity="relu",
                )

                nn.init.kaiming_uniform_(
                    self.moe.w2,
                    nonlinearity="linear",
                )

            elif init == "normal":
                nn.init.normal_(
                    self.moe.w1,
                    mean=0.0,
                    std=0.02,
                )

                nn.init.normal_(
                    self.moe.w2,
                    mean=0.0,
                    std=0.02,
                )

            nn.init.zeros_(
                self.moe.b1
            )

            nn.init.zeros_(
                self.moe.b2
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor=None,
        return_aux: bool=False,
    ):
        """
        Args:
            x:
                [B, D, S, F]

            adj:
                [S, S]

                Only required when:
                    cs_type == "gnn"

                Prefer normalized sparse adjacency.

            return_aux:
                If True, additionally return:
                    MoE gates
                    residual prediction
                    linear prediction
                    residual scale

        Returns:
            pred:
                [B, S, 1]
        """

        if x.dim() != 4:
            raise ValueError(
                f"x must have shape [B,D,S,F], "
                f"but got {tuple(x.shape)}."
            )

        B, D, S, F_dim = x.shape

        if F_dim != self.feature_dim:
            raise ValueError(
                f"feature dim must be "
                f"{self.feature_dim}, "
                f"but got {F_dim}."
            )

        # ====================================================
        # Linear Backbone
        #
        # only use current / latest day
        # ====================================================
        x_last = x[:, -1]

        if self.use_linear_backbone:
            linear_pred = (
                self.linear_backbone(
                    x_last
                )
            )  # [B,S,1]

        else:
            linear_pred = None

        # ====================================================
        # Fancy Branch
        # ====================================================

        # ----------------------------------------------------
        # Feature Encoder
        #
        # [B,D,S,F]
        # ->
        # [B,D,S,C]
        # ----------------------------------------------------
        h = self.encoder(x)

        # ----------------------------------------------------
        # Cross-Section
        #
        # Attention:
        #     ignores adj
        #
        # GNN:
        #     requires adj
        # ----------------------------------------------------
        h = self.cross_section(
            h,
            adj=adj,
        )

        # ----------------------------------------------------
        # Market MoE
        # ----------------------------------------------------
        if self.use_moe:
            h, gates = self.moe(h)

        else:
            gates = None

        # ----------------------------------------------------
        # Temporal
        #
        # [B,D,S,C]
        # ->
        # [B,D,S,C]
        # ----------------------------------------------------
        h = self.temporal(h)

        # ----------------------------------------------------
        # Latest temporal state
        #
        # [B,D,S,C]
        # ->
        # [B,S,C]
        # ----------------------------------------------------
        h = h[:, -1]

        # ----------------------------------------------------
        # Prediction Head
        #
        # [B,S,C]
        # ->
        # [B,S,1]
        # ----------------------------------------------------
        fancy_pred = self.head(h)

        # ====================================================
        # Linear + Nonlinear Residual
        # ====================================================
        if self.use_linear_backbone:

            pred = (
                linear_pred
                + self.residual_scale
                * fancy_pred
            )

        else:
            pred = fancy_pred

        # ====================================================
        # Optional diagnostics
        # ====================================================
        if return_aux:

            aux = {
                "gates": gates,
                "linear_pred": linear_pred,
                "fancy_pred": fancy_pred,
                "residual_scale": (
                    self.residual_scale
                    if self.use_linear_backbone
                    else None
                ),
            }

            return pred, aux

        return pred


# end of app/models/fancy.py
