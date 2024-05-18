# le classi

import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

class TokensAndLabels(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.tokens = []
        self.labels = []
        self.attentions = []
        self.unk = unk
        
        for x in dataset:
            self.tokens.append(x['tokens'])
            self.labels.append(x['labels'])
            self.attentions.append(x['attention'])

        self.labels_ids = self.mapping_seq(self.labels, lang.labels2id)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = torch.Tensor(self.tokens[idx])
        labs = torch.Tensor(self.labels_ids[idx])
        attention = torch.Tensor(self.attentions[idx])
        sample = {'tokens': tok, 'labels': labs, 'attention': attention}
        return sample
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

class Lang():
    def __init__(self, intents, PAD_TOKEN, cutoff=0):
        self.PAD_TOKEN = PAD_TOKEN
        self.labels2id = self.lab2id(intents, pad=False)
        self.id2lables = {v:k for k, v in self.labels2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
class BertFineTune(nn.Module):
    def __init__(self,model, label_len, dropout_prob=0.1, device="cpu"):
        super().__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = model.to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, label_len).to(device)  # Sposta il classificatore di slot sul dispositivo

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        slot_logits = self.slot_classifier(sequence_output)
        slot_logits = slot_logits.permute(0, 2, 1)

        return slot_logits