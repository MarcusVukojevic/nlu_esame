# tutte le funzioni per  caricare il dataset

import torch


PAD_TOKEN = 0
device = "mps:0"


import re

def load_data(path, valid_labels):
    '''
        input: path/to/data.txt
        output: list of dicts with tokens and labels
    '''
    dataset = []
    # Definisci un insieme di etichette valide
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
    nuovo_testo = []
    for i, testo in enumerate(item["tokens"]):
        testo_token = tokenizer.tokenize(testo)
        nuovo_testo.append(testo_token[0])
    tmp = tokenizer.encode_plus(" ".join(nuovo_testo),add_special_tokens=True, return_attention_mask=True,  return_tensors='pt')
    item["tokens"] = tmp["input_ids"][0]
    item["attention"] = tmp["attention_mask"][0]
    item["labels"] = ["O"] + item["labels"] + ["O"]
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
    data.sort(key=lambda x: len(x['tokens']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_tokens, _ = merge(new_item['tokens'], PAD_TOKEN)
    y_lables, y_lengths = merge(new_item["labels"], PAD_TOKEN)
    src_att, _ = merge(new_item["attention"], PAD_TOKEN)
    src_tokens = src_tokens.to(device) # We load the Tensor on our selected device
    y_lables = y_lables.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item = {}
    new_item["tokens"] = src_tokens
    new_item["y_labels"] = y_lables
    new_item["attention"] = src_att
    new_item["labels_len"] = y_lengths
    return new_item