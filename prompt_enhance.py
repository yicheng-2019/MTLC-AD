import torch
import torch.nn as nn
from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindImageProcessor


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None
                 # Optional[int] = None
                 ):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class PromptEnhanceModule(nn.Module):
    """
    构建模型主程序
    """
    def __init__(self,
                 hidden_dim=512,
                 # vid_mllm_ckpt='/media/qiuyc/data_disk/weights/LanguageBind/LanguageBind_Video_Huge_V1.5_FT',
                 vid_mllm_ckpt='/aidata/qiuyc/weights/LanguageBind/LanguageBind_Video_Huge_V1.5_FT',
                 device="cuda:0",
                 learnable_prompt_len=10,):
        super().__init__()
        # self.vid_mllm_ckpt = vid_mllm_ckpt
        self.device = device  # 模型运行设备GPU

        # 加载多模态大模型语言部分
        self.mllm = LanguageBindVideo.from_pretrained(vid_mllm_ckpt).to(self.device)
        self.mllm_encoder = self.mllm.text_model
        self.mllm_proj = self.mllm.text_projection
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(vid_mllm_ckpt)
        # 冻结参数，不训练大模型
        for param in self.mllm.parameters():
            param.requires_grad = False

        self.prompt_hidden_dim = self.mllm_encoder.config.hidden_size  # 构建模型
        self.hidden_dim = hidden_dim  # 构建特征传播的隐藏层维度，512或1024
        self.seq_max = 77  # 多模态中文本token最大长度,一般为77
        # self.seq_max = self.text_model_config.max_position_embeddings

        # 构建可学习prompt的长度
        self.learnable_prompt_len = learnable_prompt_len

        # 构建用于Prompt Enhance 的embedding层
        self.learnable_embeddings = nn.Embedding(self.seq_max, self.prompt_hidden_dim).to(self.device)

        self.out_proj = nn.Linear(self.prompt_hidden_dim, self.hidden_dim, bias=False).to(self.device)

        # 参数初始化
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.learnable_embeddings.weight, std=0.01)
        nn.init.normal_(self.out_proj.weight)  # 初始化降维层的权重，减少信息丢失
        # nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def encode_prompt_enhance_text(self, texts):
        # Text_Prompt增强方法，用于在普通文本中嵌入构建可训练Prompt
        word_tokens, attention_mask = self.tokenizer(
            texts, max_length=self.seq_max, padding='max_length', truncation=True, return_tensors='pt').values()
        word_embedding = self.mllm_encoder.embeddings(word_tokens.to(self.device))
        prompt_enhance_embeddings = self.learnable_embeddings(
            torch.arange(self.seq_max).to(self.device)).unsqueeze(0).repeat(
            [len(texts), 1, 1])
        prompt_enhance_tokens = torch.zeros(len(texts), 77).to(self.device)

        for i in range(len(texts)):
            ind = torch.argmax(word_tokens[i], -1)
            prompt_enhance_embeddings[i, 0] = word_embedding[i, 0]
            prompt_enhance_embeddings[i, self.learnable_prompt_len + 1: self.learnable_prompt_len + ind] = word_embedding[i, 1: ind]
            prompt_enhance_embeddings[i, self.learnable_prompt_len + ind + self.learnable_prompt_len] = word_embedding[i, ind]
            attention_mask[i, : self.learnable_prompt_len * 2 + ind + 1] = 1
            prompt_enhance_tokens[i, self.learnable_prompt_len * 2 + ind] = word_tokens[i, ind]

        causal_attention_mask = _make_causal_mask(
            word_tokens.size(), prompt_enhance_embeddings.dtype, device=prompt_enhance_embeddings.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask.to(self.device), prompt_enhance_embeddings.dtype)
        # attention_mask =
        encoder_outputs = self.mllm_encoder.encoder(
            inputs_embeds=prompt_enhance_embeddings,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.mllm_encoder.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            prompt_enhance_tokens.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        return pooled_output

    def forward(self, texts):

        # 基于mllm大模型读取的特征
        text_feat = self.encode_prompt_enhance_text(texts)
        text_feat = self.mllm_proj(text_feat)
        text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)  # 预训练读取的模型

        # 如果图像的维度也为1024就不加了
        outputs = self.out_proj(text_feat)   # [T, 1024] -> [T, 512]

        text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

        return outputs
    

if __name__ == '__main__':

    # 测试运行和
    device = "cuda:0"
    vid_mllm_ckpt = '/media/qiuyc/data_disk/weights/LanguageBind/LanguageBind_Video_Huge_V1.5_FT'
    cls_names = [
        'normal', 'abuse', 'arrest', 'arson', 'assault',
        'burglary', 'explosion', 'fighting', 'roadAccidents',
        'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism']
    vid_features = torch.randn((8, 256, 512)).to(device)

    module = PromptEnhanceModule(vid_mllm_ckpt=vid_mllm_ckpt, device=device)
    # PE_model.to(device)
    result = module(videos=vid_features, texts=cls_names)