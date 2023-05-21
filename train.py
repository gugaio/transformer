from tokenizer import Tokenizer
from transformer import Transformer
import logging
import torch
import torch.optim as optim
from optimizer import ScheduledOptim
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    logger.info("gold continoues shape = {}".format(gold.shape))
    logger.info("pred {}".format(pred[0][0:10]))
    logger.info("gold {}".format(gold[0]))

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

def train_batch(model, src_batch, trg_batch, optimizer, device, smoothing, trg_pad_idx):
    src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
    optimizer.zero_grad()
    logger.info("src_batch.shape = {}".format(src_batch.shape))
    prediction = model(src_batch, trg_batch)
    loss, n_correct, n_word = cal_performance(prediction, trg_batch, trg_pad_idx, smoothing=smoothing) 
    loss.backward()
    optimizer.step_and_update_lr()
    return loss, n_correct, n_word


def train(model, dataloader, tokenizer, optimizer, device, epochs, log_interval):
    model.train()
    for epoch in range(epochs):
        logger.info("\n\n\nEpoch {}".format(epoch))
        for batch_idx, (src_batch, trg_batch) in enumerate(dataloader):
            loss, n_correct, n_word = train_batch(model, src_batch, trg_batch, optimizer, device, smoothing=False, trg_pad_idx=tokenizer.TRG_VOCAB["<pad>"])
            if batch_idx % log_interval == 0:
                logger.info("batch index = {}".format(batch_idx))
                logger.info("loss = {}".format(loss))
                logger.info("n_correct = {}".format(n_correct))
                logger.info("n_word = {}".format(n_word))
                logger.info("accuracy = {}".format(n_correct/n_word))

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
    train(transformer, dataloader, tokenizer, optimizer, device="cpu", epochs=10, log_interval=100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()