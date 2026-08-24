对，你这个判断是对的：**不要用 `Linear(5000, 32)` 去压股票维度**。

因为那相当于：

[
[z_1,\dots,z_{32}]^T=W_{32\times5000}[h_1,\dots,h_{5000}]^T
]

其中 (W) 的第 17 列永远对应“第 17 个股票位置”。一旦股票排序变化，含义就变了，会把 **stock sequence/index 写死**。

你真正要做的是：

[
\boxed{\text{Learned Attentive Pooling}}
]

它比完整 MultiheadAttention 还容易实现。

---

## 1. 最简单的 5000 → 32 方法

Encoder 后，你现在一天的数据是：

[
H_d\in\mathbb R^{5000\times64}
]

整个 10 天：

[
H\in\mathbb R^{10\times5000\times64}
]

因为 (B=1)，我后面直接省掉 batch 轴。

定义 32 个 learnable vectors：

[
Q\in\mathbb R^{32\times64}
]

也就是：

```python
self.latents = nn.Parameter(
    torch.randn(32, 64) * 0.02
)
```

然后让每个 latent 和所有 5000 只股票计算 similarity：

[
score_{i,m}
===========

h_i^\top q_m
]

所以：

[
Score\in\mathbb R^{5000\times32}.
]

然后**沿股票维度做 softmax**：

[
a_{i,m}
=======

\frac{\exp(score_{i,m})}
{\sum_{j=1}^{5000}\exp(score_{j,m})}
]

这样对于每一个 latent (m)：

[
\sum_{i=1}^{5000}a_{i,m}=1.
]

最后：

[
z_m
===

\sum_{i=1}^{5000}
a_{i,m}h_i.
]

于是：

[
\boxed{
[5000,64]
\rightarrow
[32,64]
}
]

这就是全部逻辑。

---

# 2. 最重要的一点：这里没有 stock-specific parameter

例如你交换股票 1 和股票 3000：

[
H' = PH
]

attention weight 也会跟着交换。

但是最后：

[
Z
]

不会因为“股票放在第几个位置”而改变。

所以这个 aggregation 对股票集合是：

[
\boxed{\text{permutation invariant}}
]

这正是你要的。

而：

```python
Linear(5000, 32)
```

做不到这一点。

---

# 3. PyTorch 可以简单到这个程度

假设输入：

```text
x: [1, 10, 5000, 64]
```

代码：

```python
class LatentPooling(nn.Module):
    def __init__(self, embed_dim=64, num_latents=32):
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(num_latents, embed_dim) * 0.02
        )

        self.scale = embed_dim ** -0.5

    def forward(self, x):
        """
        x: [1, D, S, E]
        """

        # B = 1
        x = x.squeeze(0)
        # [D, S, E]

        # similarity between stocks and latent queries
        # [D, S, E] x [M, E]
        # -> [D, S, M]
        score = torch.einsum(
            "dse,me->dsm",
            x,
            self.latents
        ) * self.scale

        # Each latent distributes attention over 5000 stocks
        weight = torch.softmax(
            score,
            dim=1
        )
        # [D, S, M]

        # weighted aggregation:
        # [D,S,M] + [D,S,E]
        # -> [D,M,E]
        z = torch.einsum(
            "dsm,dse->dme",
            weight,
            x
        )

        return z
```

输出：

[
\boxed{
[1,10,5000,64]
\rightarrow
[10,32,64]
}
]

如果想保留 batch：

```python
return z.unsqueeze(0)
```

就是：

[
[1,10,32,64].
]

### 整个过程甚至没有一次 `transpose/permute`。

---

# 4. 这个代码实际上在干什么？

例如：

```text
latent 0
```

可能对 5000 股票产生：

```text
Stock 1       0.0001
Stock 2       0.0004
Stock 3       0.0021
...
Stock 827     0.031
Stock 1242    0.024
...
```

然后：

[
z_0
===

0.0001h_1
+0.0004h_2
+\cdots
+0.031h_{827}
+\cdots
]

另一个：

```text
latent 1
```

有自己的 query：

[
q_1
]

所以会得到完全不同的一组股票权重。

最终：

[
z_0,z_1,\ldots,z_{31}
]

就是 32 种不同的 market summaries。

---

# 5. 这其实比直接 MultiHeadAttention 更适合你的第一版

完整 Cross-Attention 是：

[
QW_Q,\quad KW_K,\quad VW_V
]

然后：

[
Attention(Q,K,V).
]

你现在这个简化版本其实相当于：

[
Q=\text{learnable latent},
\quad
K=V=H
]

没有复杂的 multi-head projection。

优点是：

* 极简单；
* 显存低；
* 没有 transpose；
* permutation invariant；
* 容易 debug；
* 很容易看每个 latent 到底关注哪些股票。

所以如果你明天就要实现：

[
\boxed{\text{我反而建议先用这个版本。}}
]

等证明有效，再升级 Multihead Cross-Attention。

---

# 6. 还有一个小改进我会加上

计算 similarity 之前，可以做 LayerNorm：

```python
self.norm = nn.LayerNorm(64)
```

然后：

```python
h = self.norm(x)
```

甚至如果你特别希望 similarity 更接近 cosine：

```python
h = F.normalize(h, dim=-1)
q = F.normalize(self.latents, dim=-1)

score = torch.einsum(
    "dse,me->dsm",
    h,
    q
)
```

这样：

[
score_{i,m}
===========

\cos(h_i,q_m).
]

这和你后面 GNN 用 cosine 构图的逻辑还非常一致：

### GNN

[
\cos(h_i,h_j)
]

决定：

> stock-stock local relationship。

### Latent Pooling

[
\cos(h_i,q_m)
]

决定：

> stock-global latent relationship。

这个统一性其实挺漂亮。

---

# 7. 所以你整个 Cross Section 可以有两条完全不同的路

Encoder：

[
[10,5000,64].
]

### Local branch

股票和股票：

[
\cos(h_i,h_j)
]

↓

Top-(K_g=32)

↓

GNN

↓

[
H_{local}\in[10,5000,64].
]

### Global branch

股票和 latent：

[
\cos(h_i,q_m)
]

↓

softmax over 5000 stocks

↓

[
\boxed{
5000\rightarrow M=32
}
]

↓

[
Z_{global}\in[10,32,64].
]

如果需要再把 global 信息回写股票，就再做：

[
32\rightarrow5000.
]

这两条逻辑其实非常统一：

[
\boxed{
\begin{aligned}
Local &: Stock\leftrightarrow Stock\
Global &: Stock\leftrightarrow Latent
\end{aligned}}
]

---

## 最后一个概念一定要区分

你不是：

[
\boxed{64\ features\rightarrow32\ features}
]

那才是普通：

```python
nn.Linear(64, 32)
```

你现在做的是：

[
\boxed{
5000\ stock\ tokens
\rightarrow
32\ latent\ tokens
}
]

每个 token 仍然：

[
64\text{ dimensions}.
]

所以：

[
[5000,64]
\rightarrow
[32,64].
]

**(E=64) 没变，变的是 token/set dimension (S:5000\to M:32)。**

如果追求第一版简单、稳、没有股票顺序约束，我会直接采用上面的 **learnable queries + cosine/dot-product + softmax pooling**，而不是 `Linear(S,M)`。
