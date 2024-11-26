import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrainablePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_sequence_length, d_model)
        nn.init.constant_(self.embedding.weight, 0.)  # 初始化为 0

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.embedding.weight.device).unsqueeze(0)
        return self.embedding(positions)  # (1, seq_len, d_model)


class PriorNetwork(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=1, num_heads=8, ff_dim=256, dropout=0, max_seq_len=10):
        super(PriorNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # 替换为可学习的位置编码
        self.position_embeddings = TrainablePositionEncoding(max_seq_len, embedding_dim)

        # Transformer Encoder Layer
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # 输出投影层
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

        self._initialize_model()
    
    def _initialize_model(self):                
        # 初始化 Transformer Encoder 为单位映射
        for layer in self.transformer_encoder.layers:
            # 自注意力部分初始化为仅关注自身
            nn.init.eye_(layer.self_attn.in_proj_weight[:self.embedding_dim])  # 初始化 Query 权重
            nn.init.eye_(layer.self_attn.in_proj_weight[self.embedding_dim:2*self.embedding_dim])  # 初始化 Key 权重
            nn.init.eye_(layer.self_attn.in_proj_weight[2*self.embedding_dim:])  # 初始化 Value 权重
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.eye_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            
            # 前向全连接层初始化为恒等变换
            nn.init.eye_(layer.linear1.weight)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.eye_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
        
        # 初始化输出层为恒等映射
        nn.init.eye_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, text_embedding):
        # 扩展输入维度以适配 Transformer 的输入 (batch, seq_len, embed_dim)
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # Repeat到max_seq_len长度
        
        # 添加可学习的位置编码
        seq_len = text_embedding.size(1)
        print("seq_len:" , seq_len  )
        position_encoding = self.position_embeddings(seq_len)
        text_embedding = text_embedding + position_encoding  # (batch, seq_len, embed_dim)

        # 调整维度以适配 Transformer 的输入 (seq_len, batch, embed_dim)
        text_embedding = text_embedding.permute(1, 0, 2)
        
        # 全局注意力通过 Transformer 编码器
        transformer_output = self.transformer_encoder(text_embedding)
        
        # 调整维度回到 (batch, seq_len, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # 使用全序列的注意力，将所有位置的表示进行平均
        motion_embedding = transformer_output.mean(dim=1)  # 全局平均池化
        motion_embedding = self.output_layer(motion_embedding)  # 投影至目标嵌入空间
        
        return motion_embedding


if __name__ == "__main__":
    # 假设 text_embedding 是 (batch, embedding_dim) 维度
    text_embedding = torch.randn(32, 512)  # 示例文本嵌入

    prior_network = PriorNetwork()
    # 获取 motion embedding
    motion_embedding = prior_network(text_embedding)

    def cosine_similarity(a, b):
        # Normalize the embeddings along the feature dimension
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        # Compute the cosine similarity
        cos_sim = torch.sum(a_norm * b_norm, dim=-1)
        return cos_sim

    # 计算相似度
    cos_sim = cosine_similarity(text_embedding, motion_embedding)

    # 输出相似度和 motion embedding 的形状
    print("Motion Embedding Shape:", motion_embedding.shape)
    print("Cosine Similarity:", cos_sim)
    print("motion embedding:",motion_embedding)
    print("text embedding:",text_embedding)
