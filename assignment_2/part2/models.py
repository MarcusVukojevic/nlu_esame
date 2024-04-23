# le classi

import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

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

class Lang():
    def __init__(self, words, intents, slots, PAD_TOKEN, cutoff=0):
        self.PAD_TOKEN = PAD_TOKEN
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': self.PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    


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
