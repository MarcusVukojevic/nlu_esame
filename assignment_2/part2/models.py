# le classi

import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

from utils import allineo_slots

class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

class IntentsAndSlotsNew(data.Dataset):
    def __init__(self, dataset, tokenizer, slots2id, intents2id, max_lunghezza):
        
        self.tokenizer = tokenizer
        self.slots2id = slots2id
        self.intents2id = intents2id
        self.max_lunghezza = max_lunghezza

        self.utterances = []
        self.intents = []
        self.slots = []
        self.attentions = []

        for i in dataset:
            tmp = allineo_slots(i, self.tokenizer, self.slots2id, self.intents2id, self.max_lunghezza) 
            self.utterances.append(tmp[0])
            self.intents.append(tmp[3])
            self.slots.append(tmp[1])
            self.attentions.append(tmp[2])
    
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        sample = {
            'utterance': self.utterances[idx], 
            'slots': self.slots[idx], 
            'attention':  self.attentions[idx],
            'intent': self.intents[idx],
            }
        return sample


class Lang():
    def __init__(self, slots2id, intents2ids):
        self.slot2id = slots2id
        self.intent2id = intents2ids
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    


class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout_rate=0.1):
        super(ModelIAS, self).__init__()
        # Initialization parameters
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Adding bidirectionality to the LSTM and including dropout in the LSTM if n_layer > 1
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        
        # Output layer dimensions are doubled because LSTM is bidirectional
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
        # Dropout applied to the outputs of the LSTM before the final prediction layers
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, utterance, seq_lengths):
        # Embedding the input
        utt_emb = self.embedding(utterance)
        
        # Packing the sequence for LSTM processing
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        
        # Unpacking the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Applying dropout to the output of the LSTM
        utt_encoded = self.dropout(utt_encoded)
        
        # Handling the bidirectional output for intent classification by concatenating the last hidden states
        if self.utt_encoder.bidirectional:
            # Concatenating the last hidden states from the forward and backward directions
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]
        
        # Computing slot and intent logits
        slots = self.slot_out(utt_encoded)  # batch_size, seq_len, num_slots
        intent = self.intent_out(last_hidden)  # batch_size, num_intents
        
        # Permuting slots to match expected dimension order for loss computation
        slots = slots.permute(0, 2, 1)  # batch_size, num_slots, seq_len
        
        return slots, intent
    
from transformers import BertTokenizer, BertModel

class BertFineTune(nn.Module):
    def __init__(self, intent_num_labels, slot_num_labels, model_name="bert-base-uncased", dropout_prob=0.1, device="cpu"):
        super(BertFineTune, self).__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = BertModel.from_pretrained(model_name).to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_num_labels).to(device)  # Sposta il classificatore di intenti sul dispositivo
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_num_labels).to(device)  # Sposta il classificatore di slot sul dispositivo

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)
        
        intent_logits = self.intent_classifier(pooled_output)
        #print(intent_logits.shape)
        slot_logits = slot_logits.permute(0, 2, 1)
        return slot_logits, intent_logits
