import torch
import torch.nn as nn
from transformers import (
    DebertaV2Tokenizer, DebertaV2Model,
    AutoTokenizer, AutoModel
)

class PromptEnhanceModule(nn.Module):
    """
    构建模型主程序
    """
    def __init__(self,
                 hidden_dim=512,
                 vid_mllm_ckpt='/aidata/qiuyc/weights/LanguageBind/LanguageBind_Video_Huge_V1.5_FT',
                 device="cuda:0",
                 learnable_prompt_len=10,):
        super().__init__()
        # self.vid_mllm_ckpt = vid_mllm_ckpt
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

        # 构建用于Prompt Enhance 的embedding层
        self.learnable_embeddings = nn.Embedding(self.seq_max, self.prompt_hidden_dim).to(self.device)


        self.out_proj = nn.Linear(self.prompt_hidden_dim, self.hidden_dim, bias=False).to(self.device)

        # 参数初始化
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.learnable_embeddings.weight, std=0.01)
        nn.init.normal_(self.out_proj.weight)  # 初始化降维层的权重，减少信息丢失

    def encode_prompt_enhance_text(self, texts):
        # Text_Prompt增强方法，用于在普通文本中嵌入构建可训练Prompt
        encoded = self.tokenizer(
            texts,
            max_length=self.seq_max,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        word_tokens = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].to(self.device)

        # word_embedding = self.mllm.embeddings(word_tokens.to(self.device))
        token_embed = self.mllm.get_input_embeddings()
        word_embedding = token_embed(word_tokens.to(self.device))


        prompt_enhance_embeddings = self.learnable_embeddings(
            torch.arange(self.seq_max).to(self.device)).unsqueeze(0).repeat(
            [len(texts), 1, 1])

        for i in range(len(texts)):
            # 获取当前文本的sep token所在的位置
            ind = (word_tokens[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
            
            # 将prompt增强的embedding的第一个位置设置为CLS token的embedding
            prompt_enhance_embeddings[i, 0] = word_embedding[i, 0]
            # 加入前缀可学习prompt的embedding
            prompt_enhance_embeddings[i, self.learnable_prompt_len + 1: self.learnable_prompt_len + ind] = word_embedding[i, 1: ind]
            # 加入后缀可学习prompt的embedding
            prompt_enhance_embeddings[i, self.learnable_prompt_len + ind + self.learnable_prompt_len] = word_embedding[i, ind]
            # 更新attention_mask
            attention_mask[i, : self.learnable_prompt_len * 2 + ind + 1] = 1

        '''
        encoder_outputs = self.mllm.encoder(
            hidden_states=prompt_enhance_embeddings,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        '''
        encoder_outputs = self.mllm(
            inputs_embeds=prompt_enhance_embeddings,
            attention_mask=attention_mask,)

        last_hidden_state = encoder_outputs.last_hidden_state
        
        # 使用CLS池化
        pooled_output = last_hidden_state[:, 0, :]

        return pooled_output

    def forward(self, texts):

        # 基于mllm大模型读取的特征
        text_feat = self.encode_prompt_enhance_text(texts)
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