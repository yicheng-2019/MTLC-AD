import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from copy import deepcopy


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiScaleCompressedAttention(nn.Module):
    """
    Multi-scale Compressed Attention Module.
    """

    def __init__(self, embed_dim=512, seq_length=256, num_heads=8, dropout_p=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.comp_embed_dim = embed_dim // 2
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lora_dim = embed_dim // 2

        self.to_qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        self.split_heads = Rearrange('b l (h d) -> b h l d', h=self.num_heads)
        self.merge_heads = Rearrange('b h l d -> b l (h d)')

        self.norm = nn.LayerNorm(self.embed_dim)

        self.dropout_p = dropout_p

        self.comp_blk1 = CompressionBlock(
            self.head_dim, self.seq_length, comp_block_size=32,
            is_causal=True, top_k=2, dropout_p=self.dropout_p)
        self.comp_blk2 = CompressionBlock(
            self.head_dim, self.seq_length, comp_block_size=8,
            is_causal=True, top_k=8, dropout_p=self.dropout_p)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # self.gate = nn.Sequential(nn.Linear(self.head_dim, 5), nn.Softmax(dim=-1))
        out_combine_mlp = nn.Linear(self.embed_dim, self.num_heads * 3)
        nn.init.zeros_(out_combine_mlp.weight)
        out_combine_mlp.bias.data.copy_(torch.tensor([-2., -2., 2.,] * self.num_heads))

        self.gate = nn.Sequential(
            out_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b l (h s) -> b h l s', h=self.num_heads)
        )

    def forward(self, inp):

        B, Seq, Dim = inp.shape

        inp = self.norm(inp)

        q, k, v = self.to_qkv(inp).chunk(3, dim=-1)

        q, k, v = map(self.split_heads, (q, k, v))

        # 计算全局注意力global attention，设置dropout率为0.1
        glb_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # compression-selection block 1
        out1 = self.comp_blk1(q, k, v)
        # compression-selection block 2
        out2 = self.comp_blk2(q, k, v)

        # 构建门控网络权重 gate_weights: [B, num_heads, Seq, 5]
        gate_weights = self.gate(inp)

        # 输出组合 merged_outputs: [5, B, num_heads, Seq, head_dim]
        merged_out = torch.stack([
            out1,
            out2,
            glb_out])

        # 加权求和 [B, num_heads, Seq, 5] @ [5, B, num_heads, Seq, head_dim]^T -> [B, num_heads, Seq, head_dim]
        # 恢复原来形状 [B, num_heads, Seq, head_dim] -> [B, Seq, num_heads * head_dim]
        out = torch.einsum('b h l s, s b h l d -> b h l d', gate_weights, merged_out)

        out = self.merge_heads(out)

        # 这可能不是那么必要
        out = self.norm(out)

        out = self.out_proj(out)

        return out


class CompressionBlock(nn.Module):

    def __init__(
            self,
            dim,
            seq_length,
            comp_block_size=32,
            expand_factor=1,
            is_causal=False,
            top_k=4,
            dropout_p=0.1
        ):
        super().__init__()

        self.head_dim = dim
        self.seq_length = seq_length
        self.comp_blk_size = comp_block_size
        self.is_causal = is_causal  # 是否使用因果注意力机制
        self.top_k = top_k
        self.expand_factor = expand_factor
        self.lora_dim = self.head_dim // 2
        self.dropout_p = dropout_p

        # 压缩函数
        comp_mlp = nn.Sequential(
            nn.Linear(self.comp_blk_size * self.lora_dim, self.lora_dim * self.expand_factor),
            QuickGELU(),
            nn.Linear(self.lora_dim * self.expand_factor, self.lora_dim)
        )

        self.k_comp_mlp = deepcopy(comp_mlp)

        self.v_comp_mlp = deepcopy(comp_mlp)

    def make_comp_causal_mask(self, q_seq, k_seq, dtype, device):
        """
        构建针对于压缩注意力机制的因果注意力mask
        q_seq > k_seq
        q_seq: q序列长度
        comp_block_size：压缩块大小
        :return:
        """
        mask = torch.zeros((q_seq, k_seq), dtype=dtype, device=device)  # 构建mask矩阵

        # 如果需要因果mask，则填充-inf
        if self.is_causal:
            mask.fill_(float("-inf"))
            # 对于每个k序列，只允许看到前面的q序列部分
            for i in range(k_seq):
                mask[i * self.comp_blk_size: (i + 1) * self.comp_blk_size, : i + 1] = 0

        return mask

    def MultiHeadAttn(self, q, k, v, attn_mask=None, is_causal=False):
        """
        Multi-head Scaled dot-product attention with optional masking and dropout.
        :param q: [batch, num_heads, q_seq, dim]
        :param k: [batch, num_heads, k_seq, dim]
        :param v: [batch, num_heads, k_seq, dim]
        :param attn_mask: [q_seq, k_seq]
        :param dropout_p: float
        :param is_causal:是否因果注意力

        :return:  output, attn_weight
        """
        batch_size, num_heads, q_seq, dim = q.shape
        _, _, k_seq, _ = k.shape
        # q_seq, k_seq = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(dim)
        attn_bias = torch.zeros(q_seq, k_seq, dtype=q.dtype, device=q.device)

        # 是否使用注意力掩码
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(q_seq, k_seq, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)  # 返回dropout之前的值，如果需要
        # attn_weight_drop = torch.dropout(attn_weight, p=self.dropout_p, train=self.training)  # dropout
        output = attn_weight @ v

        return output, attn_weight

    def _select(self, k, v, comp_attn, tok_k, attn_mask):
        # Select important blocks based on attention scores
        B, H, L, D = k.shape

        k_blocks = k.view(B, H, -1, self.comp_blk_size, D)
        v_blocks = v.view(B, H, -1, self.comp_blk_size, D)

        # comp_attn: batch, num_heads, q_seq, k_seq
        block_scores = comp_attn.sum(dim=2)  # [batch, num_heads, q_seq, k_seq] -> [batch, num_heads, k_seq]
        # [batch, num_heads, k_seq] -> [batch, num_heads, top_k]
        top_k_scores, top_k_indices = torch.topk(block_scores, k=tok_k, dim=-1)  # top_k_indices: [batch, num_heads, top_k]

        # 构建用于selected的mask
        # attn_mask: [q_seq, k_seq] -> [batch, num_heads, q_seq, top_k]
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        comp_tok_indices = top_k_indices.unsqueeze(-2).expand(B, H, self.seq_length, tok_k)
        selected_block_mask = torch.gather(
            attn_mask, dim=-1, index=comp_tok_indices)
        selected_mask = selected_block_mask.unsqueeze(-1).expand(
            B, H, self.seq_length, tok_k, self.comp_blk_size)
        selected_mask = selected_mask.reshape(B, H, self.seq_length, tok_k * self.comp_blk_size)

        # 构建用于selected的q和k
        k_selected = torch.gather(
            k_blocks, dim=2, index=top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, k_blocks.shape[-2], k_blocks.shape[-1])).view(B, H, -1, D)
        v_selected = torch.gather(
            v_blocks, dim=2, index=top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, v_blocks.shape[-2], v_blocks.shape[-1])).view(B, H, -1, D)

        return k_selected, v_selected, selected_mask

    def forward(self, q, k, v):

        B, H, L, _ = k.shape

        q1, q2 = torch.chunk(q, 2, dim=-1)
        k1, k2 = torch.chunk(k, 2, dim=-1)
        v1, v2 = torch.chunk(v, 2, dim=-1)

        # 先进行Compression Attention
        k_blocks = k1.view(B, H, -1, self.comp_blk_size, self.lora_dim).flatten(3)
        v_blocks = v1.view(B, H, -1, self.comp_blk_size, self.lora_dim).flatten(3)

        k_compressed = self.k_comp_mlp(k_blocks)
        v_compressed = self.v_comp_mlp(v_blocks)
        comp_attn_mask = self.make_comp_causal_mask(
            self.seq_length, k_compressed.shape[-2], dtype=k.dtype, device=k.device)

        comp_out, comp_score = self.MultiHeadAttn(q1, k_compressed, v_compressed, comp_attn_mask)

        k_slc, v_slc, slc_attn_mask = self._select(k2, v2, comp_score, self.top_k, comp_attn_mask)

        slc_out = F.scaled_dot_product_attention(q2, k_slc, v_slc, attn_mask=slc_attn_mask)

        out = torch.cat([comp_out, slc_out], dim=-1)

        return out