import torch
import torch.nn as nn
import numpy as np
from transformer.codec_blocks import EncoderBlock, DecoderBlock
from transformer.position_encoding import PositionalEncoding
import logging

class Encoder(nn.Module):
      
  def __init__(self, n_src_vocab, n_layers, d_model, d_inner, n_head, d_k, d_v,padding_idx,  dropout_rate=0.1, n_position=200, scale_emb=False):
    super(Encoder, self).__init__()
    self.logger = logging.getLogger('Encoder')
    self.logger.info('Encoder initializing')
    self.logger.debug('n_src_vocab = {}'.format(n_src_vocab))
    self.logger.debug('n_layers = {}'.format(n_layers))
    self.logger.debug('d_model = {}'.format(d_model))

    self.logger.debug('d_inner = {}'.format(d_inner))
    self.logger.debug('n_head = {}'.format(n_head))

    self.logger.debug('d_k = {}'.format(d_k))
    self.logger.debug('d_v = {}'.format(d_v))
    self.logger.debug('n_position = {}'.format(n_position))

    self.scale_emb = scale_emb
    self.src_embedding = nn.Embedding(n_src_vocab, d_model, padding_idx=padding_idx)
    self.position_embedding = PositionalEncoding(d_model, n_position=n_position)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_head, d_k, d_v, d_inner, dropout_rate) for _ in range(n_layers)])    
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.d_model = d_model
      
  def forward(self, X, src_mask, return_attns=False):
    # X.shape = (batch_size, seq_len)
    #src_mask.shape = (batch_size, seq_len)

    self.logger.debug('Embedding, position encoding, dropout and normalizing')
    self.logger.debug('Encoding X.shape = {}'.format(X.shape))   

    X = self.src_embedding(X)
    self.logger.debug("X.shape after embedding = {}".format(X.shape))

    if self.scale_emb:
        self.logger.debug('Scaling X by sqrt(d_model)')
        X *= np.sqrt(self.d_model)

    X = self.dropout(self.position_embedding(X))
    X = self.layer_norm(X)

    list_attention_weights = []
    self.logger.debug("Calling total of {} encoder blocks".format(len(self.encoder_blocks)))
    for encoder_block in self.encoder_blocks:
        self.logger.debug("Calling encoder block")
        X, attention_weights = encoder_block(X, mask=src_mask)
        self.logger.debug("X.shape after encoder block = {}".format(X.shape))
        list_attention_weights += [attention_weights] if return_attns else []

    self.logger.debug('Encoding finished')

    if return_attns:
        return X, list_attention_weights    
    return X
      

class Decoder(nn.Module):
    
    def __init__(self, n_trg_vocab, d_model, n_layers, n_head, d_k, d_v, padding_idx, d_inner, n_position=200, dropout_rate=0.1, scale_emb=False):
        super(Decoder, self).__init__()
        self.logger = logging.getLogger('Decoder')
        self.logger.info('Decoder initializing')
        self.logger.debug('n_trg_vocab = {}'.format(n_trg_vocab))
        self.logger.debug('d_model = {}'.format(d_model))
        self.logger.debug('d_inner = {}'.format(d_inner))
        self.logger.debug('n_head = {}'.format(n_head))
        self.logger.debug('d_k = {}'.format(d_k))
        self.logger.debug('d_v = {}'.format(d_v))
        self.logger.debug('dropout_rate = {}'.format(dropout_rate))

        self.trg_embedding = nn.Embedding(n_trg_vocab, d_model, padding_idx=padding_idx)
        self.position_encoding = PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_head, d_k, d_v, d_inner, dropout_rate) for _ in range(n_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        
    def forward(self,  decode_input, encode_output, self_attention_mask=None, decode_encode_attention_mask=None, return_attns=False):
        # decode_input.shape = (batch_size, seq_len, d_model)
        self.logger.debug('Decoding started')
        self.logger.debug('decode_input.shape = {}'.format(decode_input.shape))
        self.logger.debug('encode_output.shape = {}'.format(encode_output.shape))

        decode_embedded_input = self.trg_embedding(decode_input)
        self.logger.debug("decode_embedded_input.shape after embedding = {}".format(decode_embedded_input.shape))

        if self.scale_emb:
            self.logger.debug('Scaling decode_embedded_input by sqrt(d_model)')
            decode_embedded_input *= np.sqrt(self.d_model)

        decode_embedded_input = self.dropout(self.position_encoding(decode_embedded_input))
        self.logger.debug("decode_embedded_input.shape after position encoding = {}".format(decode_embedded_input.shape))

        decode_embedded_input = self.layer_norm(decode_embedded_input)
        decode_output = decode_embedded_input

        self.logger.debug('Calling total of {} decoder blocks'.format(len(self.decoder_blocks)))
        for decoder_block in self.decoder_blocks:
            self.logger.debug("Calling decoder block")
            decode_output, decode_self_attention_weights, decode_encode_attention_weights = decoder_block(decode_output, encode_output, self_attention_mask, decode_encode_attention_mask)
            self.logger.debug("decode_output.shape after decoder block = {}".format(decode_output.shape))
            
        if return_attns:
            return decode_output, decode_self_attention_weights, decode_encode_attention_weights
        return decode_output
    
def main():

    logging.basicConfig(level=logging.DEBUG)

    n_src_vocab = 10000
    n_layers = 6
    d_model = 512
    d_iiner = 2048
    n_head = 8

    d_k = d_v = d_model // n_head
    dropout_rate = 0.1
    n_position = 200
    scale_emb = False

    batch_size = 64
    lower_word_index = 1
    upper_word_index = 9999
    seq_len= 50

    padding_idx = 0

    src_seq = torch.randint(lower_word_index, upper_word_index, (batch_size, seq_len))
    src_mask = torch.ones((batch_size, 1, seq_len))

    print("\n\n**** ==> Init Encoder \n\n")

    encoder = Encoder(n_src_vocab, n_layers, d_model, d_iiner, n_head, d_k, d_v,padding_idx, dropout_rate, n_position, scale_emb)
    encode_output = encoder(src_seq, src_mask)

    n_trg_vocab = 10000

    print("\n\n**** ==> Init Encoder \n\n")
    
    decoder = Decoder(n_trg_vocab, d_model, n_layers, n_head, d_k, d_v, padding_idx, d_iiner, n_position, dropout_rate, scale_emb)

    tgt_seq = torch.randint(lower_word_index, upper_word_index, (batch_size, seq_len))
    tgt_mask = torch.ones((batch_size, seq_len, seq_len))

    decode_output = decoder(tgt_seq, encode_output, tgt_mask, src_mask)

    print("End")

if __name__ == "__main__":
    main()