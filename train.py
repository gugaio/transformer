import math
from tokenizer import Tokenizer
from transformer import Transformer
import logging
import torch
import torch.optim as optim
from optimizer import ScheduledOptim
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def calculate_perfomance(pred, gold, trg_pad_idx):
    ''' Apply label smoothing if needed '''

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word


def calculate_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    logger.debug("gold continoues shape = {}".format(gold.shape))
    logger.debug("pred {}".format(pred[0][0:10]))
    logger.debug("gold {}".format(gold[0]))

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss



def log_batch_result(batch_idx,log_interval, loss, n_correct, n_word):
    if is_time_to_log_batch_result(batch_idx, log_interval):
        logger.info("batch index = {}".format(batch_idx))
        logger.info("loss = {}".format(loss))
        logger.info("n_correct = {}".format(n_correct))
        logger.info("n_word = {}".format(n_word))
        logger.info("accuracy = {}".format(n_correct/n_word))

def is_time_to_log_batch_result(batch_idx, log_interval):
    return batch_idx % log_interval == 0

def learn(loss, optimizer):
    loss.backward()
    optimizer.step_and_update_lr()

def train_batch(model, src_batch, trg_batch, optimizer, device, smoothing, trg_pad_idx):
    src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
    optimizer.zero_grad()
    prediction = model(src_batch, trg_batch)
    n_correct, n_word = calculate_perfomance(prediction, trg_batch, trg_pad_idx) 
    loss = calculate_loss(prediction, trg_batch, trg_pad_idx, smoothing=smoothing)
    learn(loss,optimizer)
    return loss, n_correct, n_word

def train_epoch(model, dataloader, optimizer, log_interval, device, smoothing, trg_pad_idx):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 
    for batch_idx, (src_batch, trg_batch) in enumerate(dataloader):
        logger.info("\n\n batch_idx = {}".format(batch_idx))
        loss, n_correct, n_word = train_batch(model, src_batch, trg_batch, optimizer, device, smoothing, trg_pad_idx)
        log_batch_result(batch_idx, log_interval, loss, n_correct, n_word)
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
    loss_per_word = total_loss/n_word_total
    logger.info("\n\n Epoch loss per word = {}".format(loss_per_word))
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def perplexity(loss):
    return math.exp(min(loss, 100))

def train(model, dataloader, tokenizer, optimizer, device, epochs, log_interval):
    for epoch in range(epochs):
        logger.info("\n\n\nEpoch {}".format(epoch))
        train_loss, accuracy = train_epoch(model, dataloader, optimizer, log_interval, device, smoothing=False, trg_pad_idx=tokenizer.TRG_VOCAB["<pad>"])        
        #train_perplexity = perplexity(train_loss)

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
    train(transformer, dataloader, tokenizer, optimizer, device="cpu", epochs=10, log_interval=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()