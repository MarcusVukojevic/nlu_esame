import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import math
from tqdm import tqdm
import copy
import os
import numpy as np
from functools import partial

from model import LM_LSTM_NO_DROP, LM_LSTM
from utils import read_file, get_vocab, collate_fn, Lang, PennTreeBank
from functions import init_weights, train_loop, eval_loop


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")


cartella_risultati = "risultati"

os.makedirs(f'{cartella_risultati}', exist_ok=True)
os.makedirs('modelli', exist_ok=True)

os.chdir("risultati")
# Conta i file nella cartella
num_files = len([name for name in os.listdir('.') if os.path.isfile(name)])

# visto che ho 6 file per esperimento, almeno non li sovrascrive
multipli_di_6 = num_files // 6

os.chdir("..")


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


#! TUTTI CON PPL < 250

# PRIMO ASSIGNMENT:
# primo era: rimpiazzo RNN con LSTM --> LM_LSTM_NO_DROP
# secondo era: metto due dropout layers --> LM_LSTM
# terzo era: sostituisco SGD con ADAMW --> optim.AdamW(model.parameters(), lr=lear_rate)



parametri = [600, 600] # hidden e embedding size
learning_rate = 10

clip = 5 # Clip the gradient
vocab_len = len(lang.word2id)

n_epochs = 100
patience = 6

# Dizionario che contiene tutti gli esperimenti che verrano eseguiti in serie
# per ogni oesperimento, dichiaro quale modello uso, quale otimizzatore e il titolo dell'esperimento
# per far in modo di passare gli stessi parametri del modello, nel blocco successivo venogno dichiarate le funzioni di ottimizzazione
assignments = {
    "1" : [LM_LSTM_NO_DROP(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "RNN->LSTM"],
    "2": [LM_LSTM(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None , "LSTM+DROP"],
    "3": [LM_LSTM(parametri[1], parametri[0], vocab_len, pad_index=lang.word2id["<pad>"]), None, "LSTM+DROP+ADAMW" ],
}

assignments["1"][1] = optim.SGD(assignments["1"][0].parameters(), lr=learning_rate)
assignments["2"][1] = optim.SGD(assignments["2"][0].parameters(), lr=learning_rate)
assignments["3"][1] = optim.AdamW(assignments["3"][0].parameters(), lr=0.001) # dopo esperimento 1: abbasso learning rate per Adam perché performance non buona



for i in range(1,4):

    # lista dei learning rates usati durante l'allenamento
    learning_rates = []

    print("Esperimento: ", assignments[f"{i}"][2])

    # chiamo il mio modello e ottimizzatore
    model = assignments[f"{i}"][0].to(DEVICE)
    model.apply(init_weights)

    optimizer = assignments[f"{i}"][1]

    # Dichiaro lo scheduler che mi andrà ad adattare la learning rate durante il traning, alcuni sono stati provati
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # così mi cambia la lr in modo automatico
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


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

            # tengo traccia delle learning rates
            for param_group in optimizer.param_groups:
                learning_rates.append(f"Epoch {epoch+1}, Current LR: {param_group['lr']}")
    
            if i != 3:
                scheduler.step() #chiamo lo step dello scheduler per tutti tranne che per AdamW
            
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

    # Salvo in un file i miei risultati e il modello risultante
    print(f'Test PPL for {assignments[f"{i}"][2]} --> ', final_ppl)
    # Save the model
    model_path = f'bin/esperimento_{multipli_di_6}_{assignments[f"{i}"][2]}.pt'
    torch.save(best_model.state_dict(), model_path)
    
    # Save the perplexities in a txt file
    with open(f'{cartella_risultati}/esperimento_{multipli_di_6}_{assignments[f"{i}"][2]}.txt', 'w') as f:
        f.write(f'Perplexity finale: {final_ppl}, learning_rates: {learning_rates}\n')