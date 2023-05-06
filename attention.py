import torch
import torch.nn as nn

__author__ = "Gustavo Barros"

class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        '''
        :param q e k: (batch_size, n_heads, seq_len, d_k)
        :param v: (batch_size, n_heads, seq_len, d_v)
        :param mask: (batch_size, 1, 1, seq_len)
        :return: (batch_size, n_heads, seq_len, d_v), (batch_size, n_heads, seq_len, seq_len)
        '''
        assert q.size(-1) == k.size(-1)
        
        attn = torch.matmul(q, k.transpose(-1, -2))
        # scale
        d_k = k.size(-1)
        
        attn = attn / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        # softmax
        attn = torch.softmax(attn, dim=-1)
        
        output = torch.matmul(attn, v)
        return output, attn
        

if __name__ == "__main__":
    # create tensor q: (batch_size, n_heads, seq_len, d_k)
    q = torch.rand(2, 3, 4, 5)
    print(q)
    # create tensor k: (batch_size, n_heads, seq_len, d_k)
    k = torch.rand(2, 3, 4, 5)
    # create tensor v: (batch_size, n_heads, seq_len, d_v)
    v = torch.rand(2, 3, 4, 6)
    # create tensor mask: (batch_size, 1, 1, seq_len)
    mask = torch.zeros(2, 1, 1, 4)
    # create attention object
    attention = ScaledDotProductAttention()
    # forward
    output, attn = attention(q, k, v, mask)
