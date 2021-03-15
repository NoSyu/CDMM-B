from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import kobert.utils
from kobert.pytorch_kobert import get_pytorch_kobert_model


class CDMMBDataset(Dataset):
    def __init__(self, convs, decisions, users, bert_tokenizer, max_len, pad, pair, user_map_dict, max_users):
        max_user_id = len(user_map_dict)
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.convs = list()
        self.conv_length = list()

        one_utter_output = transform(("test sentence",))
        padding_utter_tokens = np.zeros_like(one_utter_output[0])
        padding_utter_valid_length = np.zeros_like(one_utter_output[1])
        padding_utter_segment_ids = np.zeros_like(one_utter_output[2])

        for one_conv in convs:
            convs_tokens = list()
            convs_valid_length = list()
            convs_segment_ids = list()

            one_conv_list = one_conv.split("<utter/>")
            if len(one_conv_list) < max_users:
                for each_utter in one_conv_list:
                    one_utter_output = transform((each_utter,))
                    convs_tokens.append(one_utter_output[0])
                    convs_valid_length.append(one_utter_output[1])
                    convs_segment_ids.append(one_utter_output[2])
                for _ in range(max_users - len(one_conv_list)):
                    convs_tokens.append(padding_utter_tokens)
                    convs_valid_length.append(padding_utter_valid_length)
                    convs_segment_ids.append(padding_utter_segment_ids)
                self.conv_length.append(len(one_conv_list))
            else:
                for each_utter in one_conv_list[:max_users]:
                    one_utter_output = transform((each_utter,))
                    convs_tokens.append(one_utter_output[0])
                    convs_valid_length.append(one_utter_output[1])
                    convs_segment_ids.append(one_utter_output[2])
                self.conv_length.append(max_users)

            self.convs.append((convs_tokens, convs_valid_length, convs_segment_ids))

        self.decisions = [np.int32(one_deci) for one_deci in decisions]
        self.users = list()
        for one_users in users:
            one_users_arr = [np.int32(user_map_dict[x]) for x in one_users.split(",")]
            if len(one_users_arr) < max_users:
                one_users_output = np.pad(one_users_arr, (0, max_users-len(one_users_arr)),
                                          'constant', constant_values=max_user_id)
            else:
                one_users_output = one_users_arr[:max_users]
            self.users.append(one_users_output)

    def __getitem__(self, idx):
        one_conv = self.convs[idx]
        one_deci = self.decisions[idx]
        one_user = self.users[idx]
        one_conv_length = self.conv_length[idx]

        return one_deci, one_user, one_conv_length, one_conv

    def __len__(self):
        return len(self.decisions)


def get_loader(raw_data, max_len, batch_size=100, shuffle=False, user_map_dict=None, max_users=10):
    def collate_fn(data):
        return zip(*data)

    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = kobert.utils.get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    dataset = CDMMBDataset(raw_data[0], raw_data[1], raw_data[2], tok, max_len, True, False, user_map_dict, max_users)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader
