
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

from utils_eval import *
from functions import  eval_loop, convert_tags_to_bieos
from transformers import BertTokenizer, BertModel

from mio_functions import *


device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

tmp_train_raw = set_labels(read_data(os.path.join('dataset','train.txt')))
test_raw = set_labels(read_data(os.path.join('dataset','test.txt')))

ote__labels = tmp_train_raw[1]
ts_labels = tmp_train_raw[2]

# Converti i tag nel formato BIEOS
convertito_train_tmp = convert_tags_to_bieos(tmp_train_raw[0])
convertito_test_tmp = convert_tags_to_bieos(test_raw[0])

portion = 0.10

# Dividere il dataset di addestramento in set di addestramento e di test
train_data, dev_data = train_test_split(convertito_train_tmp, test_size=portion, random_state=42, shuffle=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model


new_train_raw = []
for i in train_data:
    new_train_raw.append(allineo_slots_2(i,tokenizer))

new_dev_data = []
for i in dev_data:
    new_dev_data.append(allineo_slots_2(i,tokenizer))

new_test_raw = []
for i in convertito_test_tmp:
    new_test_raw.append(allineo_slots_2(i,tokenizer))


w2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1

corpus = new_train_raw + new_test_raw + new_dev_data



ote_labels = set([tag for line in corpus for tag in line['ote_tags']])
ts_labels = set([tag for line in corpus for tag in line['ts_tags']])



lang = Lang(ote_labels, ts_labels, PAD_TOKEN ,cutoff=0)

# Create our datasets
train_dataset = TokensAndLabels(new_train_raw, lang)
dev_dataset = TokensAndLabels(new_dev_data, lang)
test_dataset = TokensAndLabels(new_test_raw, lang)

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

#lr = 0.0001 # learning rate
clip = 5 # Clip the gradient
lr = 5e-5

ote_label_len = len(lang.ote_2_id)
ts_label_len = len(lang.ts_2_id)


n_epochs = 20
runs = 5

slot_f1s = []


for run in range(1):

    modello = BertFineTune(model, ote_label_len, ts_label_len, device=device).to(device)
    optimizer = optim.Adam(modello.parameters(), lr=lr)

    criterion_ote_labels = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_ts_labels = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in range(1,n_epochs):
        print("N_epoch:", x)
        loss = train_loop(train_loader, optimizer, criterion_ote_labels, criterion_ts_labels, modello)
        print("Loss: ", loss)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(dev_loader, criterion_ote_labels, criterion_ts_labels, modello, lang, tokenizer)
            # precision recall f1
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev[0][2] # prendo solo quello di ote??

            if f1 > best_f1:
                best_f1 = f1
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patient
                print("pat")
                break # Not nice but it keeps the code clean

    
    results_test, loss_test = eval_loop(test_loader, criterion_ote_labels, criterion_ts_labels, modello, lang, tokenizer)
    print("gugu")
    print(results_test)
    
    slot_f1s.append(results_test[0][2])
    #print("gugu")
    #print(results_test['total']['f'])
    #PATH = os.path.join("bin", "model_1")
    #if not os.path.exists(os.path.dirname(PATH)):
    #    os.makedirs(os.path.dirname(PATH))
    #saving_object = {"epoch": x, 
    #                "model": model.state_dict(), 
    #                "optimizer": optimizer.state_dict(), 
    #                "slot2id": lang.slot2id, 
    #                "intent2id": lang.intent2id}
    ##torch.save(saving_object, PATH)
    #plt.figure(num = run, figsize=(8, 5)).patch.set_facecolor('white')
    #plt.title('Train and Dev Losses')
    #plt.ylabel('Loss')
    #plt.xlabel('Epochs')
    #plt.plot(sampled_epochs, losses_train, label='Train loss')
    #plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    #plt.legend()
    #plt.show()
    #plt.savefig(f"results_{run}.png")
    exit()


# printa il calcolo finale
print(slot_f1s)
slot_f1s = np.asarray(slot_f1s)
#intent_acc = np.asarray(intent_acc)
print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
#print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
