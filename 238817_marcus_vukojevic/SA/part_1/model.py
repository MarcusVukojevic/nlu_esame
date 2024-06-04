
import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

from evals import evaluate, evaluate_ote, evaluate_ts
from utils_eval import *


class BertFineTune(nn.Module):
    def __init__(self,model,  ote_label_len, ts_label_len, dropout_prob=0.1, device="cpu"):
        super().__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = model.to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(0.4)
        self.ote_classifier = nn.Linear(self.bert.config.hidden_size, ote_label_len).to(device)  
        self.ts_classifier = nn.Linear(self.bert.config.hidden_size, ts_label_len).to(device) 

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        sequence_output_2 = self.dropout_2(sequence_output)

        ote_logits = self.ote_classifier(sequence_output)
        ote_logits = ote_logits.permute(0, 2, 1)

        ts_logits = self.ts_classifier(sequence_output_2)
        ts_logits = ts_logits.permute(0, 2, 1)

        return ote_logits, ts_logits