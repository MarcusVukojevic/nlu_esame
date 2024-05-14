
# Global variables
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter


from utils import load_data, collate_fn, allineo_slots
from models import Lang, IntentsAndSlots, ModelIAS, IntentsAndSlotsNew, BertFineTune
from functions import init_weights, train_loop, eval_loop


# BERT model script from: huggingface.co
from transformers import BertTokenizer, BertModel
from pprint import pprint


device = 'mps:0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
PAD_TOKEN = 0

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])
# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

w2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
slot2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
intent2id = {}



corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, # however this depends on the research purpose

slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

max_lunghezza = 52

#print(max_lunghezza)

slots2id = {'pad':PAD_TOKEN}

for slot in slots:
    if slot not in slots2id:
        slots2id[slot] = len(slots2id)
    nuovo = slot.replace('B-', 'I-')
    if nuovo not in slots2id:
        slots2id[nuovo] = len(slots2id)


intents2id = {'pad': PAD_TOKEN}
for intent in intents:
    if intent not in intents2id:
        intents2id[intent] = len(intents2id)

lang = Lang(slots2id, intents2id)
 

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model


train_dataset = IntentsAndSlotsNew(train_raw, tokenizer, slots2id, intents2id, max_lunghezza)
dev_dataset = IntentsAndSlotsNew(dev_raw, tokenizer, slots2id, intents2id, max_lunghezza)
test_dataset = IntentsAndSlotsNew(test_raw, tokenizer, slots2id, intents2id, max_lunghezza)

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

lr = 0.0001 # learning rate
clip = 5 # Clip the gradient


out_slot = len(slots2id)
out_int = len(intents2id)

n_epochs = 2
runs = 5


model = BertFineTune(out_int, out_slot, device=device).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()

patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
for x in range(1,n_epochs):
    print("N_epoch:", x)
    loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model)
    if x % 5 == 0:
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model,lang, tokenizer)
        losses_dev.append(np.asarray(loss_dev).mean())
        f1 = results_dev['total']['f']

        if f1 > best_f1:
            best_f1 = f1
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patient
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, model, lang, tokenizer)


#plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
#plt.title('Train and Dev Losses')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.plot(sampled_epochs, losses_train, label='Train loss')
#plt.plot(sampled_epochs, losses_dev, label='Dev loss')
#plt.legend()
#plt.show()

slot_f1s = np.asarray(results_test['total']['f'])
intent_acc = np.asarray(intent_test['accuracy'])
print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))