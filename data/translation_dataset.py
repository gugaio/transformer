import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import re

from data.textor import Textor

class TranslationDataset(Dataset):
    
    def __init__(self, textor):
        self.textor = textor
        self.MAX_TRG_SEQ_LEN = 0
        self.MAX_SRC_SEQ_LEN = 0

    def __len__(self):
        return len(self.textor.src_lines)

    def __getitem__(self, idx):
        return self.textor.src_lines[idx], self.textor.trg_lines[idx]
    
    def loader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.data_process)
    
    def data_process(self, batch: List[str]):
      src_batch, trg_batch = [], []
      for src_sample, trg_sample in batch:
        src_tensor_ = torch.tensor([self.textor.SRC_VOCAB[token] for token in re.findall(r'\w+', src_sample.lower())],
                                dtype=torch.long)
        src_seq_len = src_tensor_.shape[0]
        if src_seq_len > self.MAX_SRC_SEQ_LEN:
            self.MAX_SRC_SEQ_LEN = src_seq_len
          
        trg_tensor_ = torch.tensor([self.textor.TRG_VOCAB[token] for token in re.findall(r'\w+', trg_sample.lower())],
                                dtype=torch.long)
        
        trg_seq_len = trg_tensor_.shape[0]
        if trg_seq_len > self.MAX_TRG_SEQ_LEN:
            self.MAX_TRG_SEQ_LEN = trg_seq_len

        src_batch.append(src_tensor_)
        trg_batch.append(trg_tensor_)

      src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.textor.SRC_VOCAB["<pad>"]).T
      trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=self.textor.TRG_VOCAB["<pad>"]).T

      self.textor.MAX_SRC_SEQ_LEN = self.MAX_SRC_SEQ_LEN
      self.textor.MAX_TRG_SEQ_LEN = self.MAX_TRG_SEQ_LEN

      return src_batch, trg_batch
    
if __name__ == "__main__":
    textor = Textor("data/train.txt")
    textor.build()
    dataset = TranslationDataset(textor)
    dataloader = dataset.loader(batch_size=5)
    for src_lines, trg_lines in dataloader:
        print(src_lines)
        print(trg_lines)
        break
    