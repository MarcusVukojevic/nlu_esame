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

PAD_TOKEN = 0
device = "mps:0"

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
    
class BertFineTune(nn.Module):
    def __init__(self,model, intent_len, slot_len, dropout_prob=0.1, device="cpu"):
        super().__init__()
        self.device = device  # Memorizza il dispositivo come attributo della classe
        self.bert = model.to(device)  # Sposta il modello BERT sul dispositivo specificato
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_len).to(device)  # Sposta il classificatore di intenti sul dispositivo
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, slot_len).to(device)  # Sposta il classificatore di slot sul dispositivo

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
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


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterance'], sample['attention'])
        
        loss_intent = criterion_intents(intent, sample['intent'])
        loss_slot = criterion_slots(slots, sample['slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            #print(sample.keys())
            #print(sample["utterance"].shape)
            #exit()
            #print(sample["utterance"].shape)
            #print(sample["attention"].shape)
            slots, intents = model(sample['utterance'], sample['attention'])
            #print(slots.shape)
            #print(intents.shape)
            #print(model(sample['utterance'], sample['attention']))
            loss_intent = criterion_intents(intents, sample['intent'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intent'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                #utterance = [lang.id2word[elem] for elem in utt_ids]
                utterance = [elem for elem in tokenizer.convert_ids_to_tokens(utt_ids)]
                #print("Utterance")
                #print(utterance)
                #print(tokenizer.decode(utt_ids))
                #exit()
                to_decode = seq[:length].tolist()
                #print("partorito dal modello, to dec")
                #print(to_decode)
                #print("ground truth")
                #print(gt_slots)
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                #print("Quello che ho partorito:")
                #print([i[1] for i in tmp_seq])
                #print("\n")
                hyp_slots.append(tmp_seq)
            
                
                
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array