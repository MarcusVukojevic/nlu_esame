# le classi

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

# This class is the extension of the old class, where i added: bidirectionality and the dropout layer

class ModelIAS_BI(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout_rate=0.1):
        super(ModelIAS_BI, self).__init__()
        # Initialization parameters
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # aggiungo bidirectional = True 
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        
        # visto che ho messo bidirectional = True quello che viene fuori dall'LSTM è raddoppiato
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
        # The dropout layer
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, utterance, seq_lengths):
        # Embedding the input
        utt_emb = self.embedding(utterance)
        
        # Packing the sequence for LSTM processing
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        
        # Unpacking the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply the dropout layer
        utt_encoded = self.dropout(utt_encoded)
        
        # Concatenating the last hidden states from the forward and backward directions
        # This is to make the bidirectionality work
        last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        last_hidden = self.dropout(last_hidden)
        
        # Computing slot and intent logits
        slots = self.slot_out(utt_encoded)  # batch_size, seq_len, num_slots
        intent = self.intent_out(last_hidden)  # batch_size, num_intents
        
        # Permuting slots to match expected dimension order for loss computation
        slots = slots.permute(0, 2, 1)  # batch_size, num_slots, seq_len
        
        return slots, intent
