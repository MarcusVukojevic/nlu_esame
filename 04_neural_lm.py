import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import math
import numpy as np
from nn_classes import LM_LSTM, LM_RNN, Lang, PennTreeBank, LM_LSTM_NO_DROP
from functools import partial
from torch.utils.data import DataLoader
from utils import read_file, get_vocab, collate_fn, init_weights, train_loop, eval_loop
from tqdm import tqdm
import copy
import os


DEVICE = 'cuda:0'

train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
lang = Lang(train_raw, ["<pad>", "<eos>"])
train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))


# Make sure the directories exist
os.makedirs('model_bin', exist_ok=True)
os.makedirs('perplexities', exist_ok=True)

architettura = ["rnn", "lstm_no_drop", "lstm"]
loss_model = ["sgd" , "adam"]


learning_rate = [0.0001, 0.001]
parametri = [(200, 300), (150, 250)]

for arch in architettura:
    if arch == "lstm":
        for losss in loss_model:
            for lear_rate in learning_rate:
                for params in parametri:

                    clip = 5 # Clip the gradient
                    vocab_len = len(lang.word2id)
                    model = LM_LSTM(params[1], params[0], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
                    model.apply(init_weights)
                    if losss == "sgd":
                        optimizer = optim.SGD(model.parameters(), lr=lear_rate)
                    else:
                        optimizer = optim.AdamW(model.parameters(), lr=lear_rate)


                    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
                    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

                    n_epochs = 100
                    patience = 3
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
                            losses_dev.append(np.asarray(loss_dev).mean())
                            pbar.set_description("PPL: %f" % ppl_dev)
                            if  ppl_dev < best_ppl: # the lower, the better
                                best_ppl = ppl_dev
                                best_model = copy.deepcopy(model).to('cpu')
                                patience = 3
                            else:
                                patience -= 1

                            if patience <= 0: # Early stopping with patience
                                break # Not nice but it keeps the code clean
                    best_model.to(DEVICE)
                    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
                    print(f'Test PPL for {arch} with LR={lr}, with LOSS={losss}, with emb={params[1]}, hid={params[0]}: PPL: ', final_ppl)
                    # Save the model
                    model_path = f'model_bin/{arch}_LR{lear_rate}_E{params[1]}_H{params[0]}_{losss}.pt'
                    torch.save(best_model.state_dict(), model_path)
                    # Save the perplexities
                    with open(f'perplexities/{arch}_LR{lear_rate}_E{params[1]}_H{params[0]}_{losss}.txt', 'w') as f:
                        f.write(f'Test PPL: {final_ppl}\n')
                        f.write(f'Test PPL for {arch} with LR={lr}, with LOSS={losss}, with emb={params[1]}, hid={params[0]}: PPL: ', final_ppl)
    else:
        hid_size = 200
        emb_size = 300

        # With SGD try with an higher learning rate (> 1 for instance)
        lr = 0.0001 # This is definitely not good for SGD
        clip = 5 # Clip the gradient

        vocab_len = len(lang.word2id)
        if arch == "rnn":
            model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        else:
            model = LM_LSTM_NO_DROP(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        
        model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        #optimizer = optim.AdamW(model.parameters(), lr=lr)

 
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        n_epochs = 100
        patience = 3
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
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1

                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        
        print(f'Test PPL for {arch} with LR={lr}, with LOSS=SGD, with emb={emb_size}, hid={hid_size}: PPL: ', final_ppl)
        # Save the model
        model_path = f'model_bin/{arch}_LR{lr}_E{emb_size}_H{hid_size}_SGD.pt'
        torch.save(best_model.state_dict(), model_path)
        # Save the perplexities
        with open(f'perplexities/{arch}_LR{lr}_E{emb_size}_H{hid_size}_SGD.txt', 'w') as f:
            f.write(f'Test PPL: {final_ppl}\n')
            f.write(f'Test PPL for {arch} with LR={lr}, with LOSS=SGD, with emb={emb_size}, hid={hid_size}: PPL: ', final_ppl)
