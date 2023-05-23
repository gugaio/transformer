import math
from tokenizer import Tokenizer
from transformer import Transformer
import logging
import torch
import torch.optim as optim
from optimizer import ScheduledOptim
import torch.nn.functional as F

class Trainer:
    
    def __init__(self, tokenizer, dataloader, transformer, optimizer, device):
        self.logger = logging.getLogger('Trainer')
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.model = transformer
        self.optimizer = optimizer
        self.device = device
        self.log_interval = 10

    def train(self, epochs):
      for epoch in range(epochs):
          self.logger.info("\n\n\nEpoch {}".format(epoch))
          train_loss, accuracy = self.train_epoch(smoothing=False, trg_pad_idx=self.tokenizer.TRG_VOCAB["<pad>"])
        
    def train_epoch(self, smoothing, trg_pad_idx):
      self.model.train()
      total_loss, n_word_total, n_word_correct = 0, 0, 0 
      for batch_idx, (src_batch, trg_batch) in enumerate(self.dataloader):
          self.logger.info("\n\n batch_idx = {}".format(batch_idx))
          loss, n_correct, n_word = self.train_batch(src_batch, trg_batch, smoothing, trg_pad_idx)
          self.log_batch_result(batch_idx, loss, n_correct, n_word)
          n_word_total += n_word
          n_word_correct += n_correct
          total_loss += loss.item()
      loss_per_word = total_loss/n_word_total
      self.logger.info("\n\n Epoch loss per word = {}".format(loss_per_word))
      accuracy = n_word_correct/n_word_total
      return loss_per_word, accuracy
    
    def train_batch(self, src_batch, trg_batch, smoothing, trg_pad_idx):
      src_batch, trg_batch = src_batch.to(self.device), trg_batch.to(self.device)
      self.optimizer.zero_grad()
      prediction = self.model(src_batch, trg_batch)
      n_correct, n_word = self.calculate_perfomance(prediction, trg_batch, trg_pad_idx) 
      loss = self.calculate_loss(prediction, trg_batch, trg_pad_idx, smoothing=smoothing)
      self.learn(loss,self.optimizer)
      return loss, n_correct, n_word
    
    def log_batch_result(self, batch_idx,loss, n_correct, n_word):
      if self.is_time_to_log_batch_result(batch_idx, self.log_interval):
          self.logger.info("batch index = {}".format(batch_idx))
          self.logger.info("loss = {}".format(loss))
          self.logger.info("n_correct = {}".format(n_correct))
          self.logger.info("n_word = {}".format(n_word))
          self.logger.info("accuracy = {}".format(n_correct/n_word))

    def is_time_to_log_batch_result(self, batch_idx, log_interval):
      return batch_idx % log_interval == 0
    
    def learn(self, loss, optimizer):
      loss.backward()
      optimizer.step_and_update_lr()

    def calculate_perfomance(self, pred, gold, trg_pad_idx):
      ''' Apply label smoothing if needed '''

      pred = pred.max(1)[1]
      gold = gold.contiguous().view(-1)
      non_pad_mask = gold.ne(trg_pad_idx)
      n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
      n_word = non_pad_mask.sum().item()

      return n_correct, n_word


    def calculate_loss(self, pred, gold, trg_pad_idx, smoothing=False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        self.logger.debug("gold continoues shape = {}".format(gold.shape))
        self.logger.debug("pred {}".format(pred[0][0:10]))
        self.logger.debug("gold {}".format(gold[0]))

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