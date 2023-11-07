import re
from typing import List

class Tokenizer():

    def tokenize(self, text) -> List[str]:
        return [token for token in re.findall(r'\w+', text.lower())]
    
    def detokenize(self, tokens) -> str:
        return " ".join(tokens)
    
    def pad(self, tokens, max_seq_len, pad_token="<pad>") -> List[str]:
        return tokens + [pad_token] * (max_seq_len - len(tokens))
    

if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Hello, World!")
    print(tokens)
    print(tokenizer.detokenize(tokens))
    print(tokenizer.pad(tokens, 10))
    print(tokenizer.pad(tokens, 10, pad_token="<unk>"))
    print(tokenizer.pad(tokens, 10, pad_token="<bos>"))
    print(tokenizer.pad(tokens, 10, pad_token="<eos>"))
    print(tokenizer.pad(tokens, 10, pad_token="<pad>"))