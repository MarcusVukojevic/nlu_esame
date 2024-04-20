import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import math
import numpy as np
from nn_classes import LM_LSTM_2_COMPLETA, Lang, PennTreeBank, LM_LSTM_NO_DROP, LM_LSTM, LM_LSTM_2_WT
from functools import partial
from torch.utils.data import DataLoader
from utils import read_file, get_vocab, collate_fn, init_weights, train_loop, eval_loop
from tqdm import tqdm
import copy
import os

from ottimizzatore import Nt_AvSGD

DEVICE = 'cuda:0'
cartella_risultati = "risultati"
os.makedirs(f'{cartella_risultati}', exist_ok=True)

train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")#[-1000:]
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")#[-100:]
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")#[-100:]


vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
lang = Lang(train_raw, ["<pad>", "<eos>"])
train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


#! TUTTI CON PPL < 250

# PRIMO ASSIGNMENT:
# primo era: rimpiazzo RNN con LSTM --> LM_LSTM_NO_DROP
# secondo era: metto due dropout layers --> LM_LSTM
# terzo era: sostituisco SGD con ADAMW --> optim.AdamW(model.parameters(), lr=lear_rate)


# SECONDO ASSIGNMENT:
# primo era: weigth tying --> LM_LSTM_2_WT
# secondo era: variational dropout --> LM_LSTM_2_COMPLETA
# terzo era: nt-avsgd

architettura = ["lstm_no_drop", "lstm", 'lstm-wt', 'lstm-completa']
loss_model = ["sgd" , "adam", "nt-avsgd"]

parametri = [300, 300]
learning_rate = 1

clip = 5 # Clip the gradient
vocab_len = len(lang.word2id)

n_epochs = 100
patience = 6

assignments = {
    "1" : [LM_LSTM_NO_DROP(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "RNN->LSTM"],
    "2": [LM_LSTM(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "LSTM+DROP"],
    "3": [LM_LSTM(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None, "LSTM+DROP+ADAMW" ],
    "4": [LM_LSTM_2_WT(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None, "LSTM+WT"],
    "5": [LM_LSTM_2_COMPLETA(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "LSTM+WT+VD"],
    "6": [LM_LSTM_2_COMPLETA(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "LSTM+WT+VD+Nt_AvSGD"],
}

assignments["1"][1] = optim.SGD(assignments["1"][0].parameters(), lr=learning_rate)
assignments["2"][1] = optim.SGD(assignments["2"][0].parameters(), lr=learning_rate)
assignments["3"][1] = optim.AdamW(assignments["3"][0].parameters(), lr=learning_rate)
assignments["4"][1] = optim.SGD(assignments["4"][0].parameters(), lr=learning_rate)
assignments["5"][1] = optim.SGD(assignments["5"][0].parameters(), lr=learning_rate)
assignments["6"][1] = Nt_AvSGD(assignments["6"][0].parameters(), lr=learning_rate, n=5)


for i in range(1,7):
    print("Esperimento: ", assignments["1"][2])
    model = assignments[f"{i}"][0].to(DEVICE)
    model.apply(init_weights)

    optimizer = assignments[f"{i}"][1]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # così mi cambia la lr in modo automatico

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            scheduler.step(loss_dev)
            if i == 6:
                optimizer.check(ppl_dev)
            
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 6
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                print("early stopped")
                break
            
    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)

    
    print(f'Test PPL for {assignments[f"{i}"][2]} --> ', final_ppl)
    # Save the model
    #model_path = f'drop_out_01/TY{arch}2_LR{lear_rate}_E{params[1]}_H{params[0]}_{losss}.pt'
    #torch.save(best_model.state_dict(), model_path)
    # Save the perplexities
    with open(f'{cartella_risultati}/esperimento_{assignments[f"{i}"][2]}.txt', 'w') as f:
        f.write(f'Perplexity finale: {final_ppl}\n')