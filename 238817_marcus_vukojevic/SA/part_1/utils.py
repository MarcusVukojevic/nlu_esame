# Add functions or classes used for data loading and preprocessing

import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

from evals import evaluate, evaluate_ote, evaluate_ts
from utils_eval import *


PAD_TOKEN = 0
device = "cuda:0"


def convert_tags_to_bieos(data):
    finale = []
    
    for idx, item in enumerate(data):
        original_ote_tags = item["ote_tags"]
        original_ts_tags = item["ts_tags"]
        
        converted_ote_tags = ot2bieos_ote(original_ote_tags)
        converted_ts_tags = ot2bieos_ts(original_ts_tags)
        
        finale.append({
            "sentence": item["sentence"],
            "words": item["words"],
            "ote_tags": converted_ote_tags,
            "ts_tags": converted_ts_tags,
        })
    
    return finale


def allineo_slots_2(item, tokenizer):
    nuovo_testo = []
    #print(item)
    #print(item["words"])
    assert "words" in item, print("dio")
    for i, testo in enumerate(item["words"]):
        testo_token = tokenizer.tokenize(testo)
        nuovo_testo.append(testo_token[0])
    tmp = tokenizer.encode_plus(" ".join(nuovo_testo),add_special_tokens=True, return_attention_mask=True,  return_tensors='pt')
    item["tokens"] = tmp["input_ids"][0]
    item["attention"] = tmp["attention_mask"][0]
    item["ote_tags"] = ["O"] + item["ote_tags"] + ["O"]
    item["ts_tags"] = ["O"] + item["ts_tags"] + ["O"]

    return item


class Lang():
    def __init__(self, ote_label, ts_labels, PAD_TOKEN, cutoff=0):
        self.PAD_TOKEN = PAD_TOKEN

        self.ote_2_id = self.lab2id(ote_label, pad=False)
        self.id_2_ote = {v:k for k, v in self.ote_2_id.items()}

        self.ts_2_id = self.lab2id(ts_labels, pad=False)
        self.id_2_ts = {v:k for k, v in self.ts_2_id.items()}

    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab



class TokensAndLabels(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.tokens = []
        self.ote_labels = []
        self.ts_labels = []
        self.attentions = []
        self.unk = unk
        
        for x in dataset:
            self.tokens.append(x['tokens'])
            self.ote_labels.append(x['ote_tags'])
            self.ts_labels.append(x['ts_tags'])
            self.attentions.append(x['attention'])

        self.ote_labels_ids = self.mapping_seq(self.ote_labels, lang.ote_2_id)
        self.ts_labels_ids = self.mapping_seq(self.ts_labels, lang.ts_2_id)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = torch.Tensor(self.tokens[idx])
        otes = torch.Tensor(self.ote_labels_ids[idx])
        tss = torch.Tensor(self.ts_labels_ids[idx])
        attention = torch.Tensor(self.attentions[idx])
        sample = {'tokens': tok, 'ote_tags': otes, "ts_tags": tss, 'attention': attention}
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
    

def collate_fn(data):
    def merge(sequences, PAD_TOKEN):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['tokens']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_tokens, _ = merge(new_item['tokens'], PAD_TOKEN)
    y_ote_labels, y_ote_lengths =  merge(new_item["ote_tags"], PAD_TOKEN)
    y_ts_labels, y_ts_lengths =  merge(new_item["ts_tags"], PAD_TOKEN)
    src_att, _ = merge(new_item["attention"], PAD_TOKEN)
    src_tokens = src_tokens.to(device) # We load the Tensor on our selected device
    
    y_ote_labels = y_ote_labels.to(device)
    y_ote_lengths = torch.LongTensor(y_ote_lengths).to(device)

    y_ts_labels = y_ts_labels.to(device)
    y_ts_lengths = torch.LongTensor(y_ts_lengths).to(device)

    new_item = {}
    new_item["tokens"] = src_tokens
    new_item["y_ote_labels"] = y_ote_labels
    new_item["y_ote_lengths"] = y_ote_lengths

    new_item["y_ts_labels"] = y_ts_labels
    new_item["y_ts_lengths"] = y_ts_lengths

    new_item["attention"] = src_att

    return new_item