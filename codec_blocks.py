import torch.nn as nn
import torch
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_iiner, n_head, d_k, d_v, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.feed_forward = PositionWiseFeedForward(d_model, d_iiner, dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        # encoded_input.shape = (batch_size, seq_len, d_model)
        # mask.shape = (batch_size, seq_len, seq_len)
        # output.shape = (batch_size, seq_len, d_model)
        x, attention_weights = self.self_attention(q=x, k=x, v=x, mask=mask)
        x = self.feed_forward(x)
        return x, attention_weights

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_iiner, n_head, d_k, d_v, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.encode_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.feed_forward = PositionWiseFeedForward(d_model, d_iiner, dropout_rate=dropout_rate)

    def forward(self, decode_input, encode_output, self_attention_mask=None, decode_encode_attention_mask=None):
        # encoded_input.shape = (batch_size, seq_len, d_model)
        # decoded_input.shape = (batch_size, seq_len, d_model)
        # encoded_mask.shape = (batch_size, seq_len, seq_len)
        # decoded_mask.shape = (batch_size, seq_len, seq_len)
        # output.shape = (batch_size, seq_len, d_model)
        decode_output, decode_self_attention_weights = self.self_attention(decode_input, decode_input, decode_input, self_attention_mask)
        decode_output, decode_encode_attention_weights = self.encode_attention(decode_output, encode_output, encode_output, mask=decode_encode_attention_mask)
        decode_output = self.feed_forward(decode_output)
        return decode_output, decode_self_attention_weights, decode_encode_attention_weights

if __name__ == '__main__':
    encoder = EncoderBlock(512, 2048, 8, 64, 64)
    x = torch.randn(10, 20, 512)
    out, attention_weights = encoder(x)
    print(out.shape)

    decoder = DecoderBlock(512, 2048, 8, 64, 64)
    decode_input = torch.randn(10, 20, 512)
    encode_output = torch.randn(10, 20, 512)
    out, self_attention_weights, encode_attention_weights = decoder(decode_input, encode_output)
    print(out.shape)
