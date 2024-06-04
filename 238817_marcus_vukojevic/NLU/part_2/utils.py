# Add functions or classes used for data loading and preprocessing

import torch
import torch.utils.data as data
from collections import Counter
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.utils.data as data

#  required functions to iumplement the exercise
import torch.nn as nn
import torch
from conll import evaluate
from sklearn.metrics import classification_report
import json

PAD_TOKEN = 0
device = "cuda:0"

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


def allineo_slots(item, tokenizer):
    #slot_finale = []
    text = item["utterance"]
    #labels = item["slots"].split()
    new_testo = []
    for i, testo in enumerate(text.split()):
        testo_token = tokenizer.tokenize(testo)
        new_testo.append(testo_token[0])
        #if len(testo_token) != 1:
        #    for j in range(len(testo_token)):
        #        if j != 0 and labels[i] != "O":
        #            slot_finale.append(labels[i].replace('B-', 'I-'))
        #        else:
        #            slot_finale.append(labels[i])
        #else:   
        #    slot_finale.append(labels[i])
    #item["slots"] = " ".join(slot_finale)
    item["utterance"] = new_testo
    if len(item["utterance"]) == len(item["slots"].split()):
        pass
    else:
        print("gugu")
    return item
    
class Lang():
    def __init__(self,  intents, slots, PAD_TOKEN, cutoff=0):
        self.PAD_TOKEN = PAD_TOKEN
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer,unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.attentions = []
        self.unk = unk
        self.tokenizer = tokenizer
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])     

        self.tmp = [tokenizer.encode_plus(i, add_special_tokens=True, return_attention_mask=True,  return_tensors='pt') for i in self.utterances]
        self.utt_ids = [i["input_ids"][0] for i in self.tmp]
        self.attentions = [i["attention_mask"][0]for i in self.tmp]
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        attention = torch.Tensor(self.attentions[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent, 'attention': attention}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            seq = "O " + seq + " O"
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
def collate_fn(data):
    def merge(sequences):
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
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    slots, y_lenghts = merge(new_item["slots"])
    src_att, _ = merge(new_item["attention"])

    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    slots = slots.to(device)
    intent = intent.to(device)
    src_att = src_att.to(device)
    y_lengths = torch.LongTensor(y_lenghts).to(device)
    
    new_item["utterance"] = src_utt
    new_item["intent"] = intent
    new_item["slots"] = slots
    new_item["attention"] = src_att
    new_item["slots_len"] = y_lengths
    
    return new_item