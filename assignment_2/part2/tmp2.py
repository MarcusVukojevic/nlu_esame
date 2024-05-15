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


from utils import load_data
from functions import init_weights, train_loop, eval_loop

# BERT model script from: huggingface.co
from transformers import BertTokenizer, BertModel
from pprint import pprint

from mio_utils import *

device = 'cuda:0'
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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model

new_train_raw = []
for i in train_raw:
    new_train_raw.append(allineo_slots(i,tokenizer))

new_test_raw = []
for i in test_raw:
    new_test_raw.append(allineo_slots(i,tokenizer))

new_dev_raw = []
for i in dev_raw:
    new_dev_raw.append(allineo_slots(i,tokenizer))

corpus = new_train_raw + new_test_raw + new_dev_raw

slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(intents, slots, PAD_TOKEN ,cutoff=0)


# Create our datasets
train_dataset = IntentsAndSlots(new_train_raw, lang, tokenizer)
dev_dataset = IntentsAndSlots(new_dev_raw, lang, tokenizer)
test_dataset = IntentsAndSlots(new_test_raw, lang, tokenizer)

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)


#lr = 0.0001 # learning rate
clip = 5 # Clip the gradient
lr = 5e-5

slot_len = len(lang.slot2id)
intent_len = len(lang.intent2id)

n_epochs = 20
runs = 5

slot_f1s, intent_acc = [], []



modello = BertFineTune(model, intent_len, slot_len, device=device).to(device)
optimizer = optim.Adam(modello.parameters(), lr=lr)

criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
for x in range(1,n_epochs):
    print("N_epoch:", x)
    loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, modello)
    if x % 2 == 0:
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, modello, lang, tokenizer)
        losses_dev.append(np.asarray(loss_dev).mean())
        f1 = results_dev['total']['f']

        if f1 > best_f1:
            best_f1 = f1
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patient
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, modello, lang, tokenizer)
intent_acc.append(intent_test['accuracy'])
slot_f1s.append(results_test['total']['f'])
print("gugu")
print(results_test['total']['f'])
#PATH = os.path.join("bin", "model_1")
#saving_object = {"epoch": x, 
#                "model": model.state_dict(), 
#                "optimizer": optimizer.state_dict(), 
#                "w2id": w2id, 
#                "slot2id": slot2id, 
#                "intent2id": intent2id}
#torch.save(saving_object, PATH)

#plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
#plt.title('Train and Dev Losses')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.plot(sampled_epochs, losses_train, label='Train loss')
#plt.plot(sampled_epochs, losses_dev, label='Dev loss')
#plt.legend()
#plt.show()


# printa il calcolo finale
print(slot_f1s)
slot_f1s = np.asarray(slot_f1s)
intent_acc = np.asarray(intent_acc)
print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))