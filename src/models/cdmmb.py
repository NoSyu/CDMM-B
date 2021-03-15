import torch
import torch.nn as nn
import math
from kobert.pytorch_kobert import get_pytorch_kobert_model


def _gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.float()


class CDMMB(nn.Module):
    def __init__(self, config):
        super(CDMMB, self).__init__()

        self.config = config

        bertmodel, vocab = get_pytorch_kobert_model()
        self.bert = bertmodel
        self.vocab = vocab

        self.top_rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.rnn_hidden_size,
                              dropout=0, bidirectional=False, batch_first=True)

        self.user_rnn = nn.GRU(input_size=config.embedding_size, hidden_size=config.rnn_hidden_size,
                               dropout=0, bidirectional=False, batch_first=True)

        self.classifier = nn.Linear(config.rnn_hidden_size * 4, config.num_classes)
        self.dropout = nn.Dropout(p=config.dr_rate)

        self.user_embedding = nn.Embedding(config.user_size+1, config.embedding_size, padding_idx=0)
        self.user_embedding.weight.requires_grad = True

    def _attention_net(self, rnn_output, final_hidden_state):
        scale = 1. / math.sqrt(self.config.rnn_hidden_size)
        query = final_hidden_state.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = rnn_output.permute(0, 2, 1)  # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = nn.functional.softmax(energy.mul_(scale), dim=2)  # scale, normalize

        values = rnn_output # [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]

        return linear_combination

    def _user(self, users, conv_length):
        embedded_users = self.user_embedding(users)
        users_input = nn.utils.rnn.pack_padded_sequence(embedded_users, conv_length,
                                                        batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.user_rnn(users_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        attn_output = self._attention_net(rnn_output, hidden[-1])

        return attn_output

    def _conv(self, token_ids, valid_length, segment_ids, conv_length, batch_size):
        attention_mask = _gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))

        output_pooler = pooler.view(batch_size, -1, self.config.hidden_size)

        convs_input = nn.utils.rnn.pack_padded_sequence(output_pooler, conv_length,
                                                        batch_first=True, enforce_sorted=False)

        packed_output, hidden = self.top_rnn(convs_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        avg_pool = nn.functional.adaptive_avg_pool1d(rnn_output.permute(0, 2, 1), 1).view(batch_size, -1)
        max_pool = nn.functional.adaptive_max_pool1d(rnn_output.permute(0, 2, 1), 1).view(batch_size, -1)

        return hidden[-1], avg_pool, max_pool

    def forward(self, token_ids, valid_length, segment_ids, users, conv_length):
        batch_size = len(conv_length)

        user_hidden = self._user(users, conv_length)

        conv_hidden, conv_avg_pool, conv_max_pool = self._conv(token_ids, valid_length, segment_ids,
                                                               conv_length, batch_size)
        merged_output = [user_hidden, conv_hidden, conv_avg_pool, conv_max_pool]
        merged_output = torch.cat(merged_output, dim=1)
        out = self.classifier(merged_output)

        return out

