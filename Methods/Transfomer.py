import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal=False):
        # max_T: maximal sequence length
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_T = max_T
        self.causal = causal
        self.q_net = nn.Linear(ff_dim, head_dim*n_heads)
        self.k_net = nn.Linear(ff_dim, head_dim*n_heads)
        self.v_net = nn.Linear(ff_dim, head_dim*n_heads)
        self.proj_net = nn.Linear(head_dim*n_heads, ff_dim)
        self.drop_p = drop_p

    def forward(self, x):
        B, T, _ = x.shape # batch size, seq length, ff_dim
        E, D = self.n_heads, self.head_dim

        # Divide the tensors for multi head dot product
        q = self.q_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        k = self.k_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        v = self.v_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d

        inner = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p, is_causal=self.causal)
        inner = inner.transpose(1, 2).contiguous().view(B, T, E*D) # b e t d -> b t (e d) Combine results from multi heads
        return self.proj_net(inner)

class Block(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(ff_dim)
        self.attn = Attention(ff_dim, head_dim, max_T, n_heads, drop_p, causal)
        self.ln2 = nn.LayerNorm(ff_dim)
        self.ff = nn.Sequential(
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, ff_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        # Pre LN
        x = x + self.attn(self.ln1(x)) # residual
        x = x + self.ff(self.ln2(x)) # residual
        return x

class Transformer(nn.Module):

    def __init__(self, token_dim, ff_dim, head_dim, n_heads, n_blocks, max_T, drop_p, causal):
        super().__init__()
        self.proj_token = nn.Linear(token_dim, ff_dim)
        self.blocks = nn.ModuleList([Block(ff_dim, head_dim, max_T, n_heads, drop_p, causal) for _ in range(n_blocks)])

        # initialize the position embedding and the [CLS] token
        devisor = 1 / torch.sqrt(torch.tensor(ff_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, max_T, ff_dim) * devisor)

    def forward(self, x):
        B, T, C = x.shape # B: batch size, T: sequence length, C: token dim

        # token and pos embedding
        x = self.proj_token(x)
        x += self.pos_emb[:, :T, :]

        for block in self.blocks:
            x = block(x)
        return x
