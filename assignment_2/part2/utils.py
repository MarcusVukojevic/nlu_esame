# tutte le funzioni per  caricare il dataset

import json
import torch


PAD_TOKEN = 0
device = "mps:0"

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def collate_fn(data):
    # Crea dizionari per il batch raccolto
    batch = {
        'utterances': [],
        'slots': [],
        'attention_masks': [],
        'intents': [],
        'slots_len': []
    }

    # Appendi ciascun elemento del dataset a una lista nel batch
    for item in data:
        batch['utterances'].append(item['utterance'])
        batch['slots'].append(item['slots'])
        batch['attention_masks'].append(item['attention'])
        batch['intents'].append(item['intent'])  # questo è presumibilmente un intero
        batch["slots_len"].append(len(item['slots'][:(list(item['slots'][1:]).index(0)) + 2]))
    
    # Stack di ciascun elemento della lista in un unico tensore per il batch
    batch['utterances'] = torch.stack(batch['utterances']).to(device)
    batch['slots'] = torch.stack(batch['slots']).to(device)
    batch['attention_masks'] = torch.stack(batch['attention_masks']).to(device)
    batch['intents'] = torch.tensor(batch['intents']).to(device)  # Converti la lista di int in tensore
    batch["slots_len"] = torch.tensor(batch['slots_len']).to(device)
    #print(batch)
    #print(batch['utterances'].shape)      
    #print(batch['slots'].shape)
    #print(batch['attention_masks'].shape)
    #print(batch['intents'].shape)
    #print(batch["slots_len"].shape)
    return batch

def allineo_slots(item, tokenizer, slots2id, intents2id, max_lunghezza):
    slot_finale = []
    text = item["utterance"]
    labels = item["slots"].split()
    for i, testo in enumerate(text.split()):
        testo_token = tokenizer.tokenize(testo)
        if len(testo_token) != 1:
            for j in range(len(testo_token)):
                if j != 0 and labels[i] != "O":
                    slot_finale.append(labels[i].replace('B-', 'I-'))
                else:
                    slot_finale.append(labels[i])
        else:   
            slot_finale.append(labels[i])

    slot_finale_tok = [slots2id[f] for f in slot_finale]
    slot_finale_tok = [0] + slot_finale_tok + [0]*(max_lunghezza-len(slot_finale_tok) - 1)
    inten_finale_tok = intents2id[item["intent"]]

    testo_tokenizzato = tokenizer.encode_plus(text, add_special_tokens=True, return_attention_mask=True,  return_tensors='pt')
    # Padding del tensor degli input_ids
    if testo_tokenizzato['input_ids'].size(1) < max_lunghezza:
        padding_length = max_lunghezza - testo_tokenizzato['input_ids'].size(1)
        # Padding
        padded_input_ids = torch.cat([
            testo_tokenizzato['input_ids'], 
            torch.zeros((1, padding_length), dtype=torch.long)
        ], dim=1)

        padded_mask = torch.cat([
            testo_tokenizzato['attention_mask'], 
            torch.zeros((1, padding_length), dtype=torch.long)
        ], dim=1)
    else:
        padded_input_ids = testo_tokenizzato['input_ids']
        padded_mask  = testo_tokenizzato['attention_mask']
    
    return padded_input_ids[0], torch.tensor(slot_finale_tok), padded_mask[0], inten_finale_tok