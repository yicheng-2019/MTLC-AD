from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from multi_scale_compress_attn import MultiScaleCompressedAttention
from vid_text_combine import VideoTextFusion
# from prompt_enhance import PromptEnhanceModule
# from prompt_enhance_llm import PromptEnhanceModule
from prompt_enhance_llmv2 import PromptEnhanceModule


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def build_attention_mask(attn_window, visual_length, ):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    # 不是必要的
    mask = torch.empty(visual_length, visual_length)
    mask.fill_(float('-inf'))
    for i in range(int(visual_length / attn_window)):
        if (i + 1) * attn_window < visual_length:
            mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
        else:
            mask[i * attn_window: visual_length, i * attn_window: visual_length] = 0

    return mask


class MTCAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 seq_length: int,  # # layers: int,
                 num_heads: int,
                 attn_mask: torch.Tensor = None,
                 expansion_factor: int = 4,):
        super().__init__()
        """
        Multi-scale Temporal Compression Attention Block
        :param embed_dim: 输入的维度
        :param num_heads: attention的头数
        :param attn_mask: 注意力掩码，如果需要的话
        :param expansion_factor: mlp的中间层的倍率
        """
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.attn_mask = attn_mask

        self.expansion_factor = expansion_factor  # mlp的中间层的倍率
        self.attn = MultiScaleCompressedAttention(
            embed_dim=self.embed_dim, seq_length=self.seq_length, num_heads=num_heads)  # mtc attention
        self.ffn = nn.Sequential(OrderedDict([  # ffn
            ("c_fc", nn.Linear(self.embed_dim, self.embed_dim * self.expansion_factor)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.embed_dim * 4, self.embed_dim))
        ]))
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class VADModel(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length  # 时序维度Seq
        self.embed_dim = embed_dim  # 特征维度Dim
        self.attn_window = attn_window  # 不清楚
        self.learnable_prompt_len = prompt_prefix  # prompt增强维度
        self.dropout_p = 0.1
        self.device = device
        self.temperature = 0.07

        # 视频特征时序维度注意力（用于时序特征处理的注意力模块）
        self.temporal_attn1 = MTCAttentionBlock(self.embed_dim, self.visual_length, 8)
        self.temporal_attn2 = MTCAttentionBlock(self.embed_dim, self.visual_length, 8)
        self.temporal_attn3 = MTCAttentionBlock(self.embed_dim, self.visual_length, 8)

        # 视频-文本融合模块
        self.video_text_fusion1 = VideoTextFusion(vid_in_dim=512, txt_in_dim=512, num_cls=self.num_class)

        # 初始化基于多模态大模型的文本prompt增强模块
        self.prompt_enhance_module = PromptEnhanceModule(
            hidden_dim=self.embed_dim,
            learnable_prompt_len=self.learnable_prompt_len,
            vid_mllm_ckpt='/export/space0/qiu-y/model/deberta-v3-base',
            device=self.device
        )

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        self.mlp1 = nn.Sequential(
            nn.Linear(visual_width, visual_width * 4),
            QuickGELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(visual_width * 4, visual_width)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            QuickGELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )

        # 分类器，异常分数
        self.classifier = nn.Linear(self.embed_dim, 1)

        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def forward(self, visual, text):

        batch_size, Seq, Dim = visual.shape
        # 如果是i3d特征，就需要为visual加上位置编码
        if Dim == 512 or self.embed_dim == 512:
            position_ids = torch.arange(self.visual_length, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
            visual = visual.permute(1, 0, 2) + frame_position_embeddings
            visual = visual.permute(1, 0, 2)

        # 原始增强文本特征[num_cls, Dim]
        text_feat_ori = self.prompt_enhance_module(text)

        # 扩展文本特征到[batch, num_cls, Dim]
        text_feat = text_feat_ori.unsqueeze(0)
        text_feat = text_feat.expand(batch_size, text_feat.shape[1], text_feat.shape[2])

        # 传播
        visual_feat = self.temporal_attn1(visual)

        visual_feat = self.temporal_attn2(visual_feat)
        # visual_feat, text_feat = self.video_text_fusion2(visual_feat, text_feat)
        visual_feat = self.temporal_attn3(visual_feat)

        visual_feat, text_feat = self.video_text_fusion1(visual_feat, text_feat)

        # 构建logits1 [B, Seq, 1]
        logits1 = self.classifier(visual_feat + self.mlp1(visual_feat))

        # 构建logits2 [B, Seq，num_class]
        # 先进行text_feat与logits1的基于attention的关联
        visual_attn = logits1.permute(0, 2, 1) @ visual_feat
        visual_attn = visual_attn / (visual_attn.norm(dim=-1, keepdim=True) + 1e-8)
        visual_attn = visual_attn.expand(batch_size, self.num_class, Dim)
        text_feat = text_feat + visual_attn
        text_feat = text_feat + self.mlp2(text_feat)

        visual_feat_norm = visual_feat / (visual_feat.norm(dim=-1, keepdim=True) + 1e-8)  # L2归一化
        text_feat_norm = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)  # L2归一化
        text_feat_norm = text_feat_norm.permute(0, 2, 1)
        logits2 = visual_feat_norm @ text_feat_norm / self.temperature

        return text_feat_ori, logits1, logits2