import torch
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import spacy
import logging

LANG_DICT ={
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
}

spacy_en = spacy.load(LANG_DICT["en"])
spacy_de = spacy.load(LANG_DICT["de"])

class Tokenizer:
     
  def __init__(self, BATCH_SIZE=100) -> None:
    self.logger = logging.getLogger('Tokenizer')
    self.BATCH_SIZE = BATCH_SIZE
    self.MAX_SRC_SEQ_LEN = 0
    self.MAX_TRG_SEQ_LEN = 0
  
  def log(self, msg):
      self.logger.debug(msg)

  def build(self):
      self.log(f'Loading spacy model for {LANG_DICT["en"]} and {LANG_DICT["de"]}')
      self.dataset_train_interable = Multi30k(split='train', language_pair=('en', 'de'))
      self.build_vocabulary(self.dataset_train_interable)
      self.dataset_train_map = to_map_style_dataset(self.dataset_train_interable)
      self.train_dataloader = torch.utils.data.DataLoader(self.dataset_train_map, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=self.data_process)
      return self

  def get_train_dataloader(self):
      return self.train_dataloader

  def build_vocabulary(self, datataset_iterable):
      self.SRC_VOCAB = build_vocab_from_iterator(self.yield_dataset_tokens(datataset_iterable, "en", spacy_en))
      self.TRG_VOCAB = build_vocab_from_iterator(self.yield_dataset_tokens(datataset_iterable, "de", spacy_de))      

  def tokenize(self, text: str, spacy_model) -> List[str]:
      return [tok.text for tok in spacy_model.tokenizer(text)]

  def yield_dataset_tokens(self, dataset_interable: Iterable, language: str, space_model) -> List[str]:
      yield ["<pad>", "<unk>", "<bos>", "<eos>"]
      if language == "en":          
          for src_sample, trg_sample in dataset_interable:
              yield self.tokenize(src_sample.rstrip("\n"), space_model)
      elif language == "de":
          for src_sample, trg_sample in dataset_interable:
              yield self.tokenize(trg_sample.rstrip("\n"), space_model)

  def data_process(self, batch: List[str]):
      src_batch, trg_batch = [], []
      for src_sample, trg_sample in batch:
        self.log("Source " + src_sample)
        src_tensor_ = torch.tensor([self.SRC_VOCAB[token] for token in self.tokenize(src_sample.rstrip("\n"), spacy_en)],
                                dtype=torch.long)
        src_seq_len = src_tensor_.shape[0]
        if src_seq_len > self.MAX_SRC_SEQ_LEN:
            self.MAX_SRC_SEQ_LEN = src_seq_len
          
        self.log("Target " + trg_sample)
        trg_tensor_ = torch.tensor([self.TRG_VOCAB[token] for token in self.tokenize(trg_sample.rstrip("\n"), spacy_de)],
                                dtype=torch.long)
        
        trg_seq_len = trg_tensor_.shape[0]
        if trg_seq_len > self.MAX_TRG_SEQ_LEN:
            self.MAX_TRG_SEQ_LEN = trg_seq_len

        src_batch.append(src_tensor_)
        trg_batch.append(trg_tensor_)

      src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.SRC_VOCAB["<pad>"]).T
      trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=self.TRG_VOCAB["<pad>"]).T
      return src_batch, trg_batch



def main():    
    tokenizer = Tokenizer()
    tokenizer.build()
    dataloader = tokenizer.get_train_dataloader()
    print(type(dataloader))

    for batch_1_src, batch_1_target in dataloader:
        print(batch_1_src.shape)
        continue

    # batch_1_src, batch_1_target = next(iter(dataloader))
    # print(batch_1_src.shape)
    # print(batch_1_target.shape)
    # print("MAX_SRC_SEQ_LEN =", tokenizer.MAX_SRC_SEQ_LEN)
    # print("MAX_TRG_SEQ_LEN =",tokenizer.MAX_TRG_SEQ_LEN)


if __name__ == "__main__":
    main()