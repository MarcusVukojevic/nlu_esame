
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
from models import Lang, TokensAndLabels, BertFineTune
from functions import train_loop, eval_loop
from transformers import BertTokenizer, BertModel

device = 'mps:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

valid_labels = {'O', 'T-POS', 'T-NEG', 'T-NEU'}

tmp_train_raw = load_data(os.path.join('dataset','train.txt'), valid_labels)[:300]
test_raw = load_data(os.path.join('dataset','test.txt'), valid_labels)[:300]

portion = 0.10

# Dividere il dataset di addestramento in set di addestramento e di test
train_data, dev_data = train_test_split(tmp_train_raw, test_size=portion, random_state=42, shuffle=True)



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
model = BertModel.from_pretrained("bert-base-uncased") # Download the model

new_train_raw = []
for i in train_data:
    new_train_raw.append(allineo_slots(i,tokenizer))

new_test_raw = []
for i in test_raw:
    new_test_raw.append(allineo_slots(i,tokenizer))

new_dev_data = []
for i in dev_data:
    new_dev_data.append(allineo_slots(i,tokenizer))



w2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
slot2id = {'pad':PAD_TOKEN} # Pad tokens is 0 so the index count should start from 1
intent2id = {}


corpus = new_train_raw + new_test_raw + new_dev_data 
slots = set(sum([line['labels'] for line in corpus],[])) # sarebbe uguale a valid labels


lang = Lang(slots, PAD_TOKEN ,cutoff=0)


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

label_len = len(lang.labels2id)

n_epochs = 20
runs = 5

slot_f1s = []


for run in range(1):
    modello = BertFineTune(model, label_len, device=device).to(device)
    optimizer = optim.Adam(modello.parameters(), lr=lr)

    criterion_labels = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in range(1,n_epochs):
        print("N_epoch:", x)
        loss = train_loop(train_loader, optimizer, criterion_labels, modello)
        #print("Loss: ", loss)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(dev_loader, criterion_labels, modello, lang, tokenizer)
            # precision recall f1
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev[0][2]

            if f1 > best_f1:
                best_f1 = f1
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patient
                break # Not nice but it keeps the code clean

    
    results_test, loss_test = eval_loop(test_loader, criterion_labels, modello, lang, tokenizer)
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
