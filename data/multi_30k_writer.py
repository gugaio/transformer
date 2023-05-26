from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import Multi30k
import logging

class Multi30kWriter:
    
    def __init__(self) -> None:
      self.logger = logging.getLogger('Tokenizer')
      self.MAX_SRC_SEQ_LEN = 0
      self.MAX_TRG_SEQ_LEN = 0

    def build(self):
       self.dataset_train_interable = Multi30k(split='train', language_pair=('en', 'de'))

    def write(self, file_path:str):
        with open(file_path, 'w') as f:
            for src_sample, trg_sample in self.dataset_train_interable:
                if len(src_sample) > 50:
                    continue
                f.write(src_sample.rstrip("\n") + "\n")
                f.write(trg_sample.rstrip("\n") + "\n")
        self.logger.info(f'Wrote {file_path}')
        return self
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    writer = Multi30kWriter()
    writer.build()
    writer.write("data/train.txt")