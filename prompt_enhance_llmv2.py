import torch
import torch.nn as nn
from transformers import (
    DebertaV2Tokenizer, DebertaV2Model,
    AutoTokenizer, AutoModel
)

class PromptEnhanceModule(nn.Module):
    """
    构建模块主程序
    """
    def __init__(self,
                 hidden_dim=512,
                 vid_mllm_ckpt='/aidata/qiuyc/weights/LanguageBind/LanguageBind_Video_Huge_V1.5_FT',
                 device="cuda:0",
                 learnable_prompt_len=10,):
        super().__init__()
        self.device = device  # 模型运行设备GPU

        # 加载多模态大模型语言部分
        self.mllm = AutoModel.from_pretrained(vid_mllm_ckpt).to(self.device)
        # self.mllm_encoder = self.mllm
        self.tokenizer = AutoTokenizer.from_pretrained(vid_mllm_ckpt)
        # 冻结参数，不训练大模型
        for param in self.mllm.parameters():
            param.requires_grad = False
        self.mllm.eval()

        self.prompt_hidden_dim = self.mllm.config.hidden_size  # 构建模型
        self.hidden_dim = hidden_dim  # 构建特征传播的隐藏层维度，512或1024
        self.seq_max = 77  # 多模态中文本token最大长度,一般为77

        # 构建可学习prompt的长度
        self.learnable_prompt_len = learnable_prompt_len

        # === 关键：把可学习前后缀注册为"参数"（在__init__，而不是forward里）===
        init_std = getattr(self.mllm.config, "initializer_range", 0.02)
        # __init__
        self.prefix = nn.Parameter(torch.randn(self.learnable_prompt_len, self.prompt_hidden_dim) * init_std).to(self.device)
        self.suffix = nn.Parameter(torch.randn(self.learnable_prompt_len, self.prompt_hidden_dim) * init_std).to(self.device)

        self.out_proj = nn.Linear(self.prompt_hidden_dim, self.hidden_dim, bias=False).to(self.device)

        # 参数初始化
        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.learnable_embeddings.weight, std=0.01)
        nn.init.normal_(self.out_proj.weight)  # 初始化降维层的权重，减少信息丢失

    def encode_prompt_enhance_text(self, texts):
        """
        Text_Prompt增强方法, 用于在普通文本中嵌入构建可训练Prompt
        """
        
        # 计算可学习prompt后，实际文本的最大长度
        L_budget = self.seq_max - 2 * self.learnable_prompt_len
        if L_budget <= 0:
            raise ValueError(f"learnable_prompt_len {self.learnable_prompt_len} is too long for seq_max {self.seq_max}.")
        
        encoded = self.tokenizer(
            texts,
            max_length=int(L_budget),
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        num_cls = len(texts)

        word_tokens = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        token_embed = self.mllm.get_input_embeddings()
        word_embedding = token_embed(word_tokens)

        prefix = self.prefix.unsqueeze(0).expand(num_cls, -1, -1)  # (1, L, D) -> (num_cls, L, D)
        suffix = self.suffix.unsqueeze(0).expand(num_cls, -1, -1)  # (1, L, D) -> (num_cls, L, D)

        # 与可学习prompt拼接
        prompt_enhance_embeddings = torch.cat([prefix, word_embedding, suffix], dim=1)
        ones = torch.ones(num_cls, self.learnable_prompt_len, device=self.device, dtype=attention_mask.dtype)
        # mask也拼接
        attention_mask = torch.cat([ones, attention_mask, ones], dim=1)  

        # 传播提取特征，self.mllm本身不可训练
        encoder_outputs = self.mllm(
            inputs_embeds=prompt_enhance_embeddings,
            attention_mask=attention_mask,)

        last_hidden_state = encoder_outputs.last_hidden_state
        
        # 使用CLS池化，即取第一个token的输出作为句子/文本的表示
        pooled_output = last_hidden_state[:, 0, :]  # [num_cls, hidden_size]

        return pooled_output

    def forward(self, texts):

        # 基于mllm大模型读取的特征
        text_feat = self.encode_prompt_enhance_text(texts)

        # 这部分可训练
        text_feat = self.out_proj(text_feat)  # [T, 768] -> [T, 512]

        outputs = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)  # 归一化

        return outputs
    

if __name__ == '__main__':
    # 测试运行和
    device = "cuda:0"
    vid_mllm_ckpt = "/export/space0/qiu-y/model/deberta-v3-base"
    cls_names = [
        'normal', 'abuse', 'arrest', 'arson', 'assault',
        'burglary', 'explosion', 'fighting', 'roadAccidents',
        'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism']

    module = PromptEnhanceModule(vid_mllm_ckpt=vid_mllm_ckpt, device=device)
    result = module(
        texts=cls_names)
    print(f'result.shape: {result.shape}')
    print(result.device)