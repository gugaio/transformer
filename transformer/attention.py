#!/usr/bin/env python

import torch
import torch.nn as nn
import logging


class Attention(nn.Module):
    '''
    Implementation of the scaled dot-product attention
    :param q: (batch_size, n_heads, seq_len, d_k)
    :param k: (batch_size, n_heads, seq_len, d_k)
    :param v: (batch_size, n_heads, seq_len, d_v)
    :param mask: (batch_size, 1, 1, seq_len)
    :Output is a tuple of two tensors: 
        - output: (batch_size, n_heads, seq_len, d_v)
        - attn: (batch_size, n_heads, seq_len, seq_len)
    '''

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('Attention')

    def forward(self, q, k, v, mask=None):
        
        assert q.size(-1) == k.size(-1)
        self.logger.debug(f'Forward q.shape, k.shape and v.shape: {q.shape} {k.shape} {v.shape}')
        
        attn = torch.matmul(q, k.transpose(-1, -2))
        self.logger.debug(f'Attention shape: {attn.shape}')

        # scale
        d_k = k.size(-1)        
        attn = attn / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # mask
        if mask is not None:
            self.logger.debug(f'Mask: {mask.shape} and Attention {attn.shape}')
            attn = attn.masked_fill(mask == 0, -1e9)
            
        # softmax
        attn = torch.softmax(attn, dim=-1)
        
        output = torch.matmul(attn, v)
        self.logger.debug(f'Output shape: {output.shape}')
        return output, attn
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # create tensor q: (batch_size, n_heads, seq_len, d_k)
    q = torch.rand(2, 3, 4, 5)
    k = torch.rand(2, 3, 4, 5)
    v = torch.rand(2, 3, 4, 5)

    # create tensor q: (batch_size, 1, 1, seq_len)
    mask = torch.zeros(2, 1, 1, 4)
    # create attention object
    attention = Attention()

    # forward
    output, attn = attention(q, k, v, mask)
