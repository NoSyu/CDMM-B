import torch
import torch.nn as nn
import models
import os
import re
from utils import time_desc_decorator, TensorboardWriter
from transformers import AdamW


class Solver(object):
    def __init__(self, config, train_data_loader, eval_data_loader, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.config.model)(self.config)

            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)

                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            if self.config.optimizer is None:
                # AdamW
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': 0.01},
                    {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0}
                ]
                self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
            else:
                self.optimizer = self.config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                       lr=self.config.learning_rate)

    def save_model(self, epoch):
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        train_acc = getattr(self, 'train_acc', None)
        if train_acc is not None:
            self.writer.update_loss(
                loss=train_acc,
                step_i=epoch_i + 1,
                name='train_acc')

        validation_acc = getattr(self, 'validation_acc', None)
        if validation_acc is not None:
            self.writer.update_loss(
                loss=validation_acc,
                step_i=epoch_i + 1,
                name='validation_acc')

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def test(self, is_print=True):
        raise NotImplementedError

    def _calc_accuracy(self, x, y):
        max_vals, max_indices = torch.max(x, 1)
        train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]

        return train_acc
