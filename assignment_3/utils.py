# tutte le funzioni per  caricare il dataset

import torch


PAD_TOKEN = 0
device = "cpu:0"


import re

def load_data(path):
    '''
        input: path/to/data.txt
        output: list of dicts with tokens and labels
    '''
    dataset = []
    # Definisci un insieme di etichette valide
    valid_labels = {'O', 'T-POS', 'T-NEG', 'T-NEU'}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if "####" in line:  # Assicurati che la linea contenga il separatore
                sentence, tagged_tokens = line.split("####")
                tokens = []
                labels = []
                tagged_tokens = tagged_tokens.strip().split()
                for tagged_token in tagged_tokens:
                    # Utilizza una regex per estrarre token e label
                    match = re.match(r"(.+)=(\S+)", tagged_token)
                    if match:
                        token, label = match.groups()
                        if label in valid_labels:
                            tokens.append(token)
                            labels.append(label)
                        else:
                            print(f"Label '{label}' not recognized in token: {tagged_token}")
                    else:
                        print(f"Skipping malformed token: {tagged_token}")
                # Aggiungi la frase e le etichette corrispondenti al dataset
                dataset.append({'sentence': sentence.strip(), 'tokens': tokens, 'labels': labels})
    return dataset

def allineo_slots(item, tokenizer):
    slot_finale = []
    labels = item["labels"]

    for i, testo in enumerate(item["tokens"]):
        testo_token = tokenizer.tokenize(testo)
        if len(testo_token) != 1:
            print(testo_token)
            print(item["tokens"])
            print(item["labels"])
            exit()
            for j in range(len(testo_token)):
                if j != 0 and labels[i] != "O":
                    slot_finale.append(labels[i].replace('B-', 'I-'))
                else:
                    slot_finale.append(labels[i])
        else:   
            slot_finale.append(labels[i])
    item["labels"] = " ".join(slot_finale)
    return item
    

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
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'], PAD_TOKEN)
    y_slots, y_lengths = merge(new_item["slots"], PAD_TOKEN)
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item