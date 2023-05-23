import math
from tokenizer import Tokenizer
from transformer import Transformer
import logging
import torch
import torch.optim as optim
from optimizer import ScheduledOptim
import torch.nn.functional as F
from trainer import Trainer

logger = logging.getLogger(__name__)

def optimizer_for_model(model):
    lr_mul = 2
    n_warmup_steps = 4000
    return ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul, model.d_model, n_warmup_steps)

def main():
    tokenizer = Tokenizer()
    dataloader = tokenizer.build().get_train_dataloader()
    transformer = Transformer(tokenizer)
    optimizer = optimizer_for_model(transformer) 
    #train(transformer, dataloader, tokenizer, optimizer, device="cpu", epochs=10, log_interval=1)
    trainer = Trainer(tokenizer, dataloader, transformer, optimizer, device="cpu")
    trainer.train(epochs=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()