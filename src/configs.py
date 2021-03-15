import os
import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import codecs
import json

project_dir = Path(__file__).resolve().parent
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam, 'AdamW': None}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
username = Path.home().name
save_dir = project_dir.joinpath("results")
os.makedirs(save_dir, exist_ok=True)


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_json(json_path):
    with codecs.open(json_path, "r", "utf-8") as json_f:
        load_data = json.load(json_f)
    return load_data


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                setattr(self, key, value)

        self.dataset_dir = project_dir.joinpath('ajd_data')
        self.user_dict_path = self.dataset_dir.joinpath("user_dict.csv")

        self.data_dir = self.dataset_dir.joinpath(self.mode)

        self.all_path = self.data_dir.joinpath('convs_decisions_users.json')
        self.convs_path = self.data_dir.joinpath('convs.json')
        self.decisions_path = self.data_dir.joinpath('decisions.json')
        self.users_path = self.data_dir.joinpath('users_new.json')

        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = save_dir.joinpath(self.data, self.model, time_now)
            self.logdir = str(self.save_path)
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = str(self.save_path)

        self.user_dict = dict()
        self.user_map_dict = dict()
        with codecs.open(self.user_dict_path, "r", "utf-8") as csv_f:
            for line in csv_f:
                line_arr = line.strip().split("\t")
                name = line_arr[0]
                idx = int(line_arr[1])
                self.user_dict[name] = idx
                self.user_map_dict[str(idx)] = idx
        self.user_size = len(self.user_dict)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

    def to_json(self, json_f):
        json.dump(pprint.pformat(self.__dict__), fp=json_f, sort_keys=True, indent=4)


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test')

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--model', type=str, default='CDMMTN')
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--rnn_hidden_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--dr_rate', type=float, default=0.2)
    parser.add_argument('--context_size', type=int, default=1000)
    parser.add_argument('--max_users', type=int, default=10)

    parser.add_argument('--plot_every_epoch', type=int, default=1)

    parser.add_argument('--data', type=str, default='ajd_data')
    parser.add_argument('--max_len', type=int, default=100)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
