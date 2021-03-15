from .solver import Solver
from utils import time_desc_decorator
from tqdm import tqdm
from utils import to_var
import torch
import torch.nn as nn
from math import isnan
from transformers import get_linear_schedule_with_warmup
import codecs
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class SolverCDMMB(Solver):
    def __init__(self, config, train_data_loader, eval_data_loader, is_train=True, model=None):
        super(SolverCDMMB, self).__init__(config, train_data_loader, eval_data_loader, is_train, model)
        self.loss_fn = nn.CrossEntropyLoss()

    @time_desc_decorator('Training Start!')
    def train(self):
        highest_validation_acc = 0.0
        t_total = len(self.train_data_loader) * self.config.n_epoch
        warmup_step = int(t_total * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                    num_training_steps=t_total)

        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            self.model.train()
            train_acc = 0.0

            for batch_i, (decisions, users, conv_length, convs) in enumerate(tqdm(self.train_data_loader, ncols=80)):
                token_ids = list()
                valid_length = list()
                segment_ids = list()
                for one_conv_token_ids, one_conv_valid_length, one_conv_segment_ids in convs:
                    token_ids += one_conv_token_ids
                    valid_length += one_conv_valid_length
                    segment_ids += one_conv_segment_ids

                token_ids = to_var(torch.LongTensor(token_ids))
                segment_ids = to_var(torch.LongTensor(segment_ids))
                valid_length = valid_length
                decisions = to_var(torch.LongTensor(decisions))
                users = to_var(torch.LongTensor(users))

                self.optimizer.zero_grad()

                out = self.model(token_ids, valid_length, segment_ids, users, conv_length)
                batch_loss = self.loss_fn(out, decisions)
                assert not isnan(batch_loss.item())

                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                self.optimizer.step()
                scheduler.step()

                train_acc += self._calc_accuracy(out, decisions)

            print('\n<Validation>...')
            self.validation_acc = self.evaluate()
            self.train_acc = train_acc
            print("epoch {} train acc {} validation acc {}".format(epoch_i + 1,
                                                                   self.train_acc / (batch_i + 1),
                                                                   self.validation_acc))

            if self.validation_acc > highest_validation_acc:
                self.save_model(epoch_i + 1)
                highest_validation_acc = self.validation_acc

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

        return None

    def evaluate(self):
        self.model.eval()
        output_decisions = list()
        output_outs = list()

        for batch_i, (decisions, users, conv_length, convs) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            with torch.no_grad():
                token_ids = list()
                valid_length = list()
                segment_ids = list()
                for one_conv_token_ids, one_conv_valid_length, one_conv_segment_ids in convs:
                    token_ids += one_conv_token_ids
                    valid_length += one_conv_valid_length
                    segment_ids += one_conv_segment_ids

                token_ids = to_var(torch.LongTensor(token_ids))
                segment_ids = to_var(torch.LongTensor(segment_ids))
                valid_length = valid_length
                decisions = to_var(torch.LongTensor(decisions))
                users = to_var(torch.LongTensor(users))

            out = self.model(token_ids, valid_length, segment_ids, users, conv_length)

            max_vals, max_indices = torch.max(out, 1)
            max_indices = max_indices.data.cpu().numpy().tolist()
            decisions = decisions.data.cpu().numpy().tolist()

            output_outs.append(max_indices)
            output_decisions.append(decisions)

        output_outs = [one_ele for sub_list in output_outs for one_ele in sub_list]
        output_decisions = [one_ele for sub_list in output_decisions for one_ele in sub_list]

        validation_acc = accuracy_score(output_decisions, output_outs)

        return validation_acc

    def test(self, is_print=True):
        self.model.eval()
        output_decisions = list()
        output_outs = list()

        for batch_i, (decisions, users, conv_length, convs) in enumerate(tqdm(self.eval_data_loader, ncols=80)):
            with torch.no_grad():
                token_ids = list()
                valid_length = list()
                segment_ids = list()
                for one_conv_token_ids, one_conv_valid_length, one_conv_segment_ids in convs:
                    token_ids += one_conv_token_ids
                    valid_length += one_conv_valid_length
                    segment_ids += one_conv_segment_ids

                token_ids = to_var(torch.LongTensor(token_ids))
                segment_ids = to_var(torch.LongTensor(segment_ids))
                valid_length = valid_length
                decisions = to_var(torch.LongTensor(decisions))
                users = to_var(torch.LongTensor(users))

            out = self.model(token_ids, valid_length, segment_ids, users, conv_length)
            max_vals, max_indices = torch.max(out, 1)
            max_indices = max_indices.data.cpu().numpy().tolist()
            decisions = decisions.data.cpu().numpy().tolist()
            output_outs.append(max_indices)
            output_decisions.append(decisions)

        output_outs = [one_ele for sub_list in output_outs for one_ele in sub_list]
        output_decisions = [one_ele for sub_list in output_decisions for one_ele in sub_list]

        target_file_name = 'outputs_{}.csv'.format(self.epoch_i)
        with codecs.open(os.path.join(self.config.save_path, target_file_name), 'w', "utf-8") as output_f:
            for one_out, one_dec in zip(output_outs, output_decisions):
                print("{},{}".format(one_out, one_dec), file=output_f)

        if is_print:
            print(accuracy_score(output_decisions, output_outs))
            print(precision_recall_fscore_support(output_decisions, output_outs, average='macro'))
            print(precision_recall_fscore_support(output_decisions, output_outs, average='micro'))
            print(precision_recall_fscore_support(output_decisions, output_outs, average='weighted'))

        return output_decisions, output_outs
