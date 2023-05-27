from collections import Counter
import re

class Textor:
    
    def __init__(self, file_path):
        self.text = file_path
        self.src_lines = []
        self.trg_lines = []
        self.MAX_TRG_SEQ_LEN = 0
        self.MAX_SRC_SEQ_LEN = 0

    def build(self):
        self.read()
        self.SRC_VOCAB = self.build_vocabulary(self.src_lines)
        self.TRG_VOCAB = self.build_vocabulary(self.trg_lines)
        self.n_src_tokens = len(self.SRC_VOCAB)
        self.n_trg_tokens = len(self.TRG_VOCAB)
        return self

    def read(self):
        with open(self.text, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i % 2 == 0:
                    self.src_lines.append(line)
                else:
                    self.trg_lines.append(line)
        
    def build_vocabulary(self, lines):
        token_counts = Counter()
        token_counts.update(["<pad>", "<unk>", "<bos>", "<eos>"])
        for line in lines:
            tokens = re.findall(r'\w+', line.lower())
            token_counts.update(tokens)
        vocabulary = {}
        for token in token_counts.keys():
            vocabulary[token] = len(vocabulary)
        return vocabulary
    
    def reverse_vocabulary(self, counter):
        return {v: k for k, v in counter.items()}

if __name__ == "__main__":
    textor = Textor("data/train.txt")
    textor.build()
    print(textor.n_src_tokens)
    print(textor.n_trg_tokens)
    print(textor.SRC_VOCAB["<pad>"])
    print(textor.SRC_VOCAB["<unk>"])
 