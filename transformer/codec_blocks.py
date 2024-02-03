import torch.nn as nn
import torch
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionWiseFeedForward
import logging

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_iiner, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.logger = logging.getLogger('EncoderBlock')   
        self.d_k = d_k 
        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.feed_forward = PositionWiseFeedForward(d_model, d_iiner, dropout_rate=dropout_rate)

    def forward(self, x, mask=None):
        # X and mask = (batch_size, seq_len, d_model)
        self.logger.debug('Running Self Attention')
        self.logger.debug('X.shape = {}'.format(x.shape))

        if mask is not None:
            #Last dimension of mask should be equal to the size of the sequence
            assert mask.size(-1) == x.size(1)
            assert x.size(0) == mask.size(0)
            self.logger.debug('mask.shape {}'.format(mask.shape))

        x, attention_weights = self.self_attention(x, mask=mask)
        
        self.logger.debug('Running Position Wise Feed Forward')
        x = self.feed_forward(x)

        self.logger.debug('EncodeBlock finished. Returning x and attention_weights')
        self.logger.debug('x.shape = {}'.format(x.shape))
        return x, attention_weights
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_iiner, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.logger = logging.getLogger('DecoderBlock')   
        self.logger.debug('Initializing DecoderBlock')

        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.encode_decode_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout_rate=dropout_rate)
        self.feed_forward = PositionWiseFeedForward(d_model, d_iiner, dropout_rate=dropout_rate)

    def forward(self, decode_input, encode_output, self_attention_mask=None, decode_encode_attention_mask=None):
        # Encodes and mask = (batch_size, seq_len, d_model)
        self.logger.debug('decode_input.shape = {}'.format(decode_input.shape))
        self.logger.debug('encode_output.shape = {}'.format(encode_output.shape))

        if self_attention_mask is not None:
            #Last dimension of mask should be equal to the size of the sequence
            assert self_attention_mask.size(-1) == decode_input.size(1)
            self.logger.debug('self_attention_mask.shape {}'.format(self_attention_mask.shape))

        if decode_encode_attention_mask is not None:
            #Last dimension of mask should be equal to the size of the sequence
            assert decode_encode_attention_mask.size(-1) == encode_output.size(1)
            self.logger.debug('decode_encode_attention_mask.shape {}'.format(decode_encode_attention_mask.shape))

        self.logger.debug('Running Self Attention')
        decode_output, decode_self_attention_weights = self.self_attention(decode_input, decode_input, decode_input, self_attention_mask)
        self.logger.debug('Running Encode Decode Attention')
        decode_output, decode_encode_attention_weights = self.encode_decode_attention(decode_output, encode_output, encode_output, mask=decode_encode_attention_mask)
        self.logger.debug('Running Position Wise Feed Forward')
        decode_output = self.feed_forward(decode_output)

        return decode_output, decode_self_attention_weights, decode_encode_attention_weights

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    d_model = 512
    n_head = 8
    d_k = 64
    d_v = 64
    d_iiner = 2048

    batch_size = 10
    seq_len = 20

    print("\n\n**** ==> Init EncoderBlock \n\n")

    encoder = EncoderBlock(d_model, n_head, d_k, d_v, d_iiner)    
    x = torch.randn(batch_size, seq_len, d_model)
    encode_mask = torch.ones(batch_size, 1, seq_len)
    encode_output, attention_weights = encoder(x, encode_mask)

    print("\n\n**** ==> Init DecoderBlock\n\n")

    decoder = DecoderBlock(d_model, n_head, d_k, d_v, d_iiner)
    decode_input = torch.randn(batch_size, seq_len, d_model)    
    self_attention_mask = torch.ones(batch_size, 1, seq_len)
    decode_encode_mask = torch.ones(batch_size, 1, seq_len)
    out, self_attention_weights, encode_attention_weights = decoder(decode_input, encode_output, self_attention_mask, decode_encode_mask)
