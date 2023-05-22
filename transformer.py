import torch
import torch.nn as nn
import numpy as np
from codec import Encoder, Decoder
import logging

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Transformer(nn.Module):
    ''' Attention is all you need model '''

    def __init__(self, tokenizer,
            d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):
        super().__init__()
        self.logger = logging.getLogger('Transformer')
        self.logger.info("Transformer initializing")

        n_src_vocab = len(tokenizer.SRC_VOCAB)
        n_trg_vocab = len(tokenizer.TRG_VOCAB)
        src_pad_idx = tokenizer.MAX_SRC_SEQ_LEN
        trg_pad_idx = tokenizer.MAX_TRG_SEQ_LEN

        self.logger.debug("n_src_vocab = {}".format(n_src_vocab))
        self.logger.debug("n_trg_vocab = {}".format(n_trg_vocab))
        self.logger.debug("src_pad_idx = {}".format(src_pad_idx))
        self.logger.debug("trg_pad_idx = {}".format(trg_pad_idx))
        self.logger.debug("d_model = {}".format(d_model))
        self.logger.debug("d_inner = {}".format(d_inner))
        self.logger.debug("n_layers = {}".format(n_layers))
        self.logger.debug("n_head = {}".format(n_head))
        self.logger.debug("d_k = {}".format(d_k))
        self.logger.debug("d_v = {}".format(d_v))
        self.logger.debug("dropout = {}".format(dropout))
        self.logger.debug("n_position = {}".format(n_position))
        self.logger.debug("trg_emb_prj_weight_sharing = {}".format(trg_emb_prj_weight_sharing))
        self.logger.debug("emb_src_trg_weight_sharing = {}".format(emb_src_trg_weight_sharing))
        self.logger.debug("scale_emb_or_prj = {}".format(scale_emb_or_prj))

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = True if scale_emb_or_prj == 'emb' else False
        self.scale_prj = True if scale_emb_or_prj == 'prj' else False

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.d_model, self.d_inner, self.n_layers = d_model, d_inner, n_layers

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_layers=n_layers, d_model=d_model,
            d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v,
            padding_idx=src_pad_idx, dropout_rate=dropout, n_position=n_position,
            scale_emb=scale_emb)
        
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_layers=n_layers, d_model=d_model,
            d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v,
            padding_idx=trg_pad_idx, dropout_rate=dropout, n_position=n_position,
            scale_emb=scale_emb)
        
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_k * n_head and d_model == d_v * n_head

        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_embedding.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_embedding.weight = self.decoder.trg_embedding.weight

    def forward(self, src_seq, trg_seq):
        self.logger.debug("src_seq.shape = {}".format(src_seq.shape))
        self.logger.debug("trg_seq.shape = {}".format(trg_seq.shape))
        
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        self.logger.debug("src_mask.shape = {}".format(src_mask.shape))
        self.logger.debug("trg_mask.shape = {}".format(trg_mask.shape))

        enc_output = self.encoder(X=src_seq, src_mask=src_mask)
        self.logger.debug("Encoder result shape = {}".format(enc_output.shape))

        dec_output = self.decoder(decode_input=trg_seq, encode_output=enc_output, self_attention_mask=trg_mask, decode_encode_attention_mask=src_mask)
        self.logger.debug("Decoder result shape = {}".format(dec_output.shape))

        seq_logit = self.trg_word_prj(dec_output)
        self.logger.debug("Seq Logit shape = {}".format(seq_logit.shape))

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
        

  
def main():
    logging.basicConfig(level=logging.DEBUG)

    n_src_vocab = 10000
    n_trg_vocab = 10000
    src_pad_idx = 0
    trg_pad_idx = 0
    d_model = 512
    d_inner = 2048
    n_layers = 6
    n_head = 8
    d_k = 64
    d_v = 64
    dropout = 0.1
    n_position = 200
    trg_emb_prj_weight_sharing = True
    emb_src_trg_weight_sharing = True
    scale_emb_or_prj = 'none'


    transformer = Transformer(
        n_src_vocab=n_src_vocab, n_trg_vocab=n_trg_vocab,
        src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
        d_model=d_model, d_inner=d_inner,
        n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        dropout=dropout, n_position=n_position,
        trg_emb_prj_weight_sharing=trg_emb_prj_weight_sharing,
        emb_src_trg_weight_sharing=emb_src_trg_weight_sharing,
        scale_emb_or_prj=scale_emb_or_prj)
    
    lower_word_index = 0

    batch_size = 64
    seq_len = 20

    src_seq = torch.LongTensor(np.random.randint(lower_word_index, n_src_vocab, size=(batch_size, seq_len)))
    trg_seq = torch.LongTensor(np.random.randint(0, n_trg_vocab, size=(batch_size, seq_len + 5)))

    output = transformer(src_seq, trg_seq)
    print(output.size())
    

if __name__ == "__main__":
    main()
