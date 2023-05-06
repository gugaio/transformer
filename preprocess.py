import torch
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import spacy
import h5py

spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_en(text: str) -> List[str]:
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_de(text: str) -> List[str]:
    return [tok.text for tok in spacy_de.tokenizer(text)]

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    if language == "en":
        for src_sample, trg_sample in data_iter:
            yield tokenize_en(src_sample.rstrip("\n"))
    elif language == "de":
        for src_sample, trg_sample in data_iter:
            yield tokenize_de(src_sample.rstrip("\n"))

def data_process(batch: List[str]):
    src_batch, trg_batch = [], []

    for src_sample, trg_sample in batch:
        src_tensor_ = torch.tensor([SRC_VOCAB[token] for token in tokenize_en(src_sample.rstrip("\n"))],
                                   dtype=torch.long)
        trg_tensor_ = torch.tensor([TRG_VOCAB[token] for token in tokenize_de(trg_sample.rstrip("\n"))],
                                   dtype=torch.long)
        src_batch.append(src_tensor_)
        trg_batch.append(trg_tensor_)

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=SRC_VOCAB["<pad>"])
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=TRG_VOCAB["<pad>"])

    return src_batch, trg_batch

# Dataloader
BATCH_SIZE = 128

train_iter = Multi30k(split='train', language_pair=('en', 'de'))
SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_iter, "en"))
TRG_VOCAB = build_vocab_from_iterator(yield_tokens(train_iter, "de"))
train_data = to_map_style_dataset(train_iter)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_process)

data = {
    "train_dataloader": train_dataloader,
    "SRC_VOCAB": SRC_VOCAB,
    "TRG_VOCAB": TRG_VOCAB
}

save_data = "data.pkl"
print('[Info] Dumping the processed data to bin file', save_data)
with h5py.File("train_data.hdf5", "w") as f:
    for i, (src, trg) in enumerate(train_data):
        grp = f.create_group(str(i))
        grp.attrs["src"] = src
        grp.attrs["trg"] = trg

