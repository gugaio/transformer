#!/usr/bin/env python

from attention import Attention
import torch
import torch.nn as nn
import logging
from model_info import ModelInfo

class MultiHeadAttention(nn.Module):
    '''
    Implementation of a Multi Head Dot Scaled Attention
    :param d_model: (int) the number of expected features in the input
    :param num_heads: (int) the number of heads in the multiheadattention models
    :param d_k: (int) the number of features in the key
    :param d_v: (int) the number of features in the value
    :param dropout_rate: (float) the dropout value
    :Output is a tuple of two tensors:
        - output: (batch_size, seq_len_q, d_model)
        - attn: (batch_size, n_heads, seq_len_q, seq_len_k)
    '''
    def __init__(self, d_model, num_heads, d_k, d_v, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.logger = logging.getLogger('MultiHeadAttention')
        self.logger.info(f'Initializing MultiHeadAttention with d_model={d_model}, num_heads={num_heads}, d_k={d_k}, d_v={d_v}, dropout_rate={dropout_rate}')

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.wk = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.wv = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.attn_to_output_dense = nn.Linear(num_heads * d_v, d_model, bias=False)

        self.attention = Attention()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        assert q.size(-1) == k.size(-1)
        self.logger.debug(f'Forward q.shape {q.shape}, k.shape {k.shape} and v.shape {v.shape}')
        self.logger.debug(f'Batch size: {q.shape[0]} Sentence lenght:{q.shape[1]} d_model: {q.shape[2]}')

        batch_size, len_q = q.size(0), q.size(1)
        residual = q

        self.logger.debug("Creating multi-heads weights")

        q = self.wq(q)
        self.logger.debug( f'After linear projection q.shape {q.shape}')
        k = self.wk(k)
        self.logger.debug( f'After linear projection q.shape {k.shape}')
        v = self.wv(v)
        self.logger.debug( f'After linear projection q.shape {v.shape}')

        self.logger.debug(f'After multi-heads weights q.shape {q.shape}, k.shape {k.shape} and v.shape {v.shape}')
        # (seq idx, head idx, word idx, depth´ idx)  <- (seq idx, word idx, depth idx)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        self.logger.debug(f'After split heads q.shape {q.shape}, k.shape {k.shape} and v.shape {v.shape}')
        self.logger.debug(f'Batch size: {q.shape[0]} Heads:{q.shape[1]} Sentence lenght:{q.shape[2]} d_model: {q.shape[3]}')

        if mask is not None:
            self.logger.debug(f'Forward with mask.shape: {mask.shape}')
            # As we add 1 dimension because head, we need to add 1 dimension to mask
            mask = mask.unsqueeze(1)
            self.logger.debug(f'Mask unsqueeze shape: {mask.shape}')

        # scaled_attention.shape = (seq idx, head idx, word idx, depth)
        # attention_weights.shape = (seq idx, head idx, word q idx,, word k idx)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)
        
        #(batch_size, seq_len_q, depth)  <- (batch_size, seq_len_q, num_heads, depth)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        self.logger.debug(f'Concat Attention shape: batch_size {concat_attention.shape[0]} seq_len_q:{concat_attention.shape[1]} depth: {concat_attention.shape[2]}')

        # output.shape = (batch_size, seq_len_q, d_model)
        output = self.dropout(self.attn_to_output_dense(concat_attention))
        self.logger.debug(f'Feed forward result shape {output.shape}')

        output += residual
        output = self.layer_norm(output)

        self.logger.debug(f'Result shape {output.shape}')
        return output, attention_weights
    
  
    def split_into_heads(self, x):
        """
        Split the last dimension from d_model into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        self.logger.debug(f'Split into heads x.shape {x.shape}')
        batch_size = x.size(0)
        len_sentence = x.size(1)
        x = x.view(batch_size, len_sentence, self.num_heads, -1)
        return x.transpose(1, 2)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    d_model = 512
    num_heads = 8
    d_k = 64
    d_v = 64

    multiHead = MultiHeadAttention(d_model, num_heads, d_k, d_v)

    q = torch.rand(2, 4, 512)
    k = torch.rand(2, 4, 512)
    v = torch.rand(2, 4, 512)
   
    mask = torch.ones(2, 1, 4)

    output, attn = multiHead(q, k, v, mask)

    ModelInfo.print(multiHead, multiHead.logger)
    
        