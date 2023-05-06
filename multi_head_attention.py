from attention import Attention
import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.wk = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.wv = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.dense = nn.Linear(num_heads * d_v, d_model, bias=False)

        self.attention = Attention()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        batch_size, len_q = q.size(0), q.size(1)
        residual = q

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # scaled_attention.shape = (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)

        #scaled_attention.shape = (batch_size, seq_len_q, num_heads, depth)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        # output.shape = (batch_size, seq_len_q, d_model)
        output = self.dropout(self.dense(concat_attention))
        output += residual

        output = self.layer_norm(output)

        #return output, attention_weights
        return output, attention_weights
  
    def split_heads(self, x):
        """
        Split the last dimension from d_model into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        batch_size = x.size(0)
        len_sentence = x.size(1)
        x = x.view(batch_size, len_sentence, self.num_heads, -1)
        return x.transpose(1, 2)
    
if __name__ == "__main__":
    multiHead = MultiHeadAttention(512, 8, 64, 64)
    q = torch.rand(2, 4, 512)
    k = torch.rand(2, 4, 512)
    v = torch.rand(2, 4, 512)
    print(q)
    output, attn = multiHead(q, k, v)
    print("------")
    print(output)