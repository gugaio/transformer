import torch
import torch.nn as nn
import numpy as np
from codec_blocks import EncoderBlock, DecoderBlock
from position_encoding import PositionalEncoding
import logging

class Encoder(nn.Module):
      
  def __init__(self, n_src_vocab, n_layers, d_model, d_iiner, n_head, d_k, d_v,  dropout_rate=0.1, n_position=200, scale_emb=False):
    super(Encoder, self).__init__()
    self.logger = logging.getLogger('Encoder')
    self.scale_emb = scale_emb
    self.src_embedding = nn.Embedding(n_src_vocab, d_model)
    self.position_embedding = PositionalEncoding(d_model, n_position=n_position)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, d_iiner, n_head, d_k, d_v, dropout_rate) for _ in range(n_layers)])    
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.d_model = d_model
      
  def forward(self, X, src_mask, return_attns=False):
    # X.shape = (batch_size, seq_len)
    # src_mask.shape = (batch_size, seq_len, seq_len)

    self.logger.info('Forward X')
    self.logger.debug('X.shape = {}'.format(X.shape))   
    if src_mask is not None:
        self.logger.debug('Src_mask.shape ={}'.format(src_mask.shape))   

    list_attention_weights = []

    X = self.src_embedding(X)
    if self.scale_emb:
        self.logger.info('Scaling X by sqrt(d_model)')
        X *= np.sqrt(self.d_model)

    self.logger.info('Position embeded, dropout and normalizing')
    X = self.dropout(self.position_embedding(X))
    X = self.layer_norm(X)

    self.logger.info('Calling encoder blocks')
    for encoder_block in self.encoder_blocks:
        X, attention_weights = encoder_block(X, mask=src_mask)
        list_attention_weights += [attention_weights] if return_attns else []

    self.logger.info('Encoding finished')
    self.logger.debug('Returning X.shape = {}'.format(X.shape))

    if return_attns:
        return X, list_attention_weights    
    return X
      

class Decoder(nn.Module):
    
    def __init__(self, d_model, d_iiner, n_head, d_k, d_v, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, d_iiner, n_head, d_k, d_v, dropout_rate) for _ in range(6)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, decode_input, encode_output, self_attention_mask=None, decode_encode_attention_mask=None):
        # decode_input.shape = (batch_size, seq_len, d_model)
        # encode_output.shape = (batch_size, seq_len, d_model)
        # self_attention_mask.shape = (batch_size, seq_len, seq_len)
        # decode_encode_attention_mask.shape = (batch_size, seq_len, seq_len)
        # output.shape = (batch_size, seq_len, d_model)
        decode_output = decode_input
        for decoder_block in self.decoder_blocks:
            decode_output, decode_self_attention_weights, decode_encode_attention_weights = decoder_block(decode_output, encode_output, self_attention_mask, decode_encode_attention_mask)
        decode_output = self.layer_norm(decode_output)
        return decode_output, decode_self_attention_weights, decode_encode_attention_weights
    
def main():

    logging.basicConfig(level=logging.DEBUG)

    src_seq = torch.randint(1, 501, (64, 50))

    src_mask = torch.zeros((64, 1, 50)).long()
    encoder = Encoder(10000, 6, 512, 2048, 8, 64, 64, 0.1, 200, True)
    #decoder = Decoder(512, 2048, 8, 64, 64, 0.1)

    #tgt_seq = torch.zeros((64, 50)).long()
    #tgt_mask = torch.zeros((64, 1, 1, 50)).long()
    encode_output = encoder(src_seq, src_mask)

if __name__ == "__main__":
    main()