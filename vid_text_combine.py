import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class LatentFusionAttention(nn.Module):

    def __init__(self, q_dim, k_dim, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.lora_dim = 128
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        # 定义线性投影层
        self.q_proj = nn.Linear(self.q_dim, self.embed_dim, bias=False)
        self.kv_proj = nn.Linear(self.k_dim, self.embed_dim * 2, bias=False)
        self.q_lora_proj = nn.Linear(self.q_dim, self.lora_dim, bias=False)
        self.kv_lora_proj = nn.Linear(self.k_dim, self.lora_dim * 2, bias=False)

        self.fuse_proj = nn.Linear(self.embed_dim + self.lora_dim, self.q_dim, bias=False)

        self.split_heads = Rearrange('b l (h d) -> b h l d', h=self.num_heads)
        self.merge_heads = Rearrange('b h l d -> b l (h d)')

        # 添加LayerNorm
        self.A_norm = nn.LayerNorm(self.q_dim)
        self.B_norm = nn.LayerNorm(self.k_dim)
        self.out_norm = nn.LayerNorm(self.q_dim)
        self.dropout = 0.1

    def forward(self, A, B):
        """交叉注意力计算"""
        A = self.A_norm(A)
        B = self.B_norm(B)

        # q
        q = self.q_proj(A)
        q_lora = self.q_lora_proj(A)

        # 分kv
        k, v = self.kv_proj(B).chunk(2, dim=-1)
        k_lora, v_lora = self.kv_lora_proj(B).chunk(2, dim=-1)

        # 分多头
        q, k, v = map(self.split_heads, (q, k, v))
        q_lora, k_lora, v_lora = map(self.split_heads, (q_lora, k_lora, v_lora))

        # 计算注意力
        out1 = F.scaled_dot_product_attention(q, k, v)
        out_lora = F.scaled_dot_product_attention(q_lora, k_lora, v_lora)

        out = torch.cat([out1, out_lora], dim=-1)

        out = self.merge_heads(out)

        out = self.fuse_proj(out)

        self.out_norm(out)

        return out


class VideoTextFusion(nn.Module):
    """
    视频文本聚合模块
    """
    def __init__(self, vid_in_dim=512, txt_in_dim=512, embed_dim=512, num_cls=14, num_heads=8):
        super().__init__()
        # 视频解耦
        self.vid_in_dim = vid_in_dim
        self.txt_in_dim = txt_in_dim
        self.embed_dim = embed_dim
        self.num_cls = num_cls

        # 交叉注意力层
        self.vid_cross_attn = LatentFusionAttention(self.vid_in_dim, self.txt_in_dim, self.embed_dim)  # 视频对文本的注意力
        self.txt_cross_attn = LatentFusionAttention(self.txt_in_dim, self.vid_in_dim, self.embed_dim)  # 文本对视频的注意力

        # 添加LayerNorm
        self.norm_vid = nn.LayerNorm(self.vid_in_dim)
        self.norm_txt = nn.LayerNorm(self.txt_in_dim)

    def forward(self, vid_feat, txt_feat):

        b, seq, _ = vid_feat.shape  # 读取batch, 时间步长
        _, n_c, _ = txt_feat.shape  # 读取text的类别数量

        vid_feat = self.norm_vid(vid_feat)
        txt_feat = self.norm_txt(txt_feat)

        vid_feat = vid_feat + self.vid_cross_attn(vid_feat, txt_feat)
        txt_feat = txt_feat + self.txt_cross_attn(txt_feat, vid_feat)

        return vid_feat, txt_feat



