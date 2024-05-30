
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

class BertFineTune(nn.Module):
    def __init__(self,model,  ote_label_len, ts_label_len, dropout_prob=0.1, device="cpu"):
        super().__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = model.to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.ote_classifier = nn.Linear(self.bert.config.hidden_size, ote_label_len).to(device)  
        self.ts_classifier = nn.Linear(self.bert.config.hidden_size, ts_label_len).to(device) 

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        ote_logits = self.ote_classifier(sequence_output)
        ote_logits = ote_logits.permute(0, 2, 1)

        ts_logits = self.ts_classifier(sequence_output)
        ts_logits = ts_logits.permute(0, 2, 1)

        return ote_logits, ts_logits
    


def train_loop(data, optimizer, criterion_ote_labels, criterion_ts_labels, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        ote_labels, ts_labels = model(sample['tokens'], sample['attention'])
        loss_ote = criterion_ote_labels(ote_labels, sample['y_ote_labels'])
        loss_ts = criterion_ts_labels(ts_labels, sample['y_ts_labels'])

        loss = loss_ote + loss_ts
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array



def eval_loop(data, criterion_ote_labels, criterion_ts_labels, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_slots_ote = []
    hyp_slots_ote = []


    ref_slots_ts = []
    hyp_slots_ts = []

    with torch.no_grad():  # Evita la creazione del grafico computazionale
        for sample in data:
        
            
            ote_labels, ts_labels = model(sample['tokens'], sample['attention'])
            loss_ote = criterion_ote_labels(ote_labels, sample['y_ote_labels'])
            loss_ts = criterion_ts_labels(ts_labels, sample['y_ts_labels'])

            loss = loss_ote + loss_ts
            loss_array.append(loss.item())
                
            # Inferenza delle etichette predette
            ote_output_labels = torch.argmax(ote_labels, dim=1) 
            ts_output_labels = torch.argmax(ts_labels, dim=1)  
            #print(output_labels)

            # PARTE OTE
            for id_seq in range(len(ote_output_labels)):
                length = sample['y_ote_lengths'][id_seq].tolist()  # Lunghezza reale della sequenza
                #utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_ote_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id_2_ote[elem] for elem in gt_ids]
                #utterance = [tokenizer.convert_ids_to_tokens(tok) for tok in utt_ids]
                to_decode = ote_output_labels[id_seq][:length].tolist()
                #print(to_decode, [lang.id2lables[elem] for elem in to_decode])
                ref_slots_ote.append(gt_slots)
                hyp_slots_ote.append([lang.id_2_ote[elem] for elem in to_decode])
            
            # PARTE TS
            for id_seq in range(len(ts_output_labels)):
                length = sample['y_ts_lengths'][id_seq].tolist()  # Lunghezza reale della sequenza
                #utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_ts_labels'][id_seq][:length].tolist()
                gt_slots = [lang.id_2_ts[elem] for elem in gt_ids]
                #utterance = [tokenizer.convert_ids_to_tokens(tok) for tok in utt_ids]
                to_decode = ts_output_labels[id_seq][:length].tolist()
                #print(to_decode, [lang.id2lables[elem] for elem in to_decode])
                ref_slots_ts.append(gt_slots)
                hyp_slots_ts.append([lang.id_2_ts[elem] for elem in to_decode])
            
    try:
       
        ot_2_bieos_hyp_ote = [ot2bieos_ote(i) for i in hyp_slots_ote]
        ot_2_bieos_ref_ote = [ot2bieos_ote(i) for i in ref_slots_ote]


        ot_2_bieos_hyp_ts = [ot2bieos_ts(i) for i in hyp_slots_ts]
        ot_2_bieos_ref_ts = [ot2bieos_ts(i) for i in ref_slots_ts]

        # di ote mi ritorna (precision, recall, f1) e di ts (ts_macro_f1, ts_micro_p, ts_micro_r, ts_micro_f1)
        results = evaluate(ot_2_bieos_ref_ote, ot_2_bieos_ref_ts, ot_2_bieos_hyp_ote, ot_2_bieos_hyp_ts)
        
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots_ote])
        hyp_s = set([x[1] for x in hyp_slots_ote])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    
    return results, loss_array