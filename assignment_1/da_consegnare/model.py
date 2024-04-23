import torch.nn as nn
import torch
import torch.utils.data as data
import torch 
import torch.nn as nn 
from torch.autograd import Variable 


class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

class LM_LSTM_NO_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_NO_DROP, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=3):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Embedding dropout layer
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # Output dropout layer before the last linear layer
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Apply embedding dropout
        emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        # Apply dropout before the last linear layer
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

 
 
class LockedDropout(nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x, dropout=0.1): 
        if not self.training or not dropout: 
            return x 
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout) 
        mask = Variable(m, requires_grad=False) / (1 - dropout) 
        mask = mask.expand_as(x) 
        return mask * x 


class LM_LSTM_2_WT(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=3): 
        super(LM_LSTM_2_WT, self).__init__() 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) 
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
        


class LM_LSTM_2_COMPLETA(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=3): 
        super(LM_LSTM_2_COMPLETA, self).__init__() 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.embedding_dropout = LockedDropout()  #nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) 
        # Intermediate layer to map from hidden_size to emb_size
        self.out_dropout = LockedDropout()
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # Apply embedding dropout
        emb = self.embedding_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        # Apply dropout before the last linear layer
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    


# self.param_groups[0] (non so se puoi questa lista aumenta o meno)
# --> 'params' lista di tensori --> required_grad = True
# --> tutto quello che è in --> defaults = dict(lr=lr, n=n, weight_decay=weight_decay, fine_tuning=fine_tuning, t0=t0, t=0, logs=[])

import torch
from torch.optim import Optimizer

class Nt_AvSGD(Optimizer):
    def __init__(self, params, lr=1, n=5, L=2):
        self.T_inizio_averagin = 0
        self.t_validation_checks = 0
        self.validation_logs = [] # lista dei logs
        self.k = 0 #numero totale di iterazioni
        self.n_checks = n # numero di epoche che se superate fanno iniziare l'average aka non monotone interval
        self.L = L # loggin interval
        defaults = dict(lr=lr)
        super(Nt_AvSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue  # skip if gradient is zero

                state = self.state[p]
                # Inizializzo lo stato
                if len(state) == 0:
                    state['mu'] = 1
                    state['ax'] = p.data.clone()

                grad = p.grad.data  # prendi il gradiente 

                if self.T_inizio_averagin > 0:
                    # Adesso uso i parametri co n la media
                    ax = state['ax']
                    ax -= group['lr'] * grad 
                    p.data.copy_(ax) 
                else:
                    # Update normale del peso
                    p.data -= group['lr'] * grad
                    # in caso dovrebbe anche andare p.data.add_(-group['lr'], grad)

                # Se inizio a fare averagin, costruisco mano a mano la mia nuova media dei pesi
                if self.T_inizio_averagin > 0:
                    state['mu'] = 1 / max(1, self.k - self.T_inizio_averagin)
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)  # così rimane sincronizzato fino a quando inzia

        return loss

    def check(self, ppl_dev):
        self.k += 1  # incrementazioni totali --> sarebbero come le epoche
        if self.k % self.L == 0:
            if self.T_inizio_averagin == 0:
                range_valido = max(0, len(self.validation_logs) - self.n_checks)
                min_ppl = min(self.validation_logs[:range_valido]) if range_valido > 0 else float('inf')
                if self.t_validation_checks > self.n_checks and ppl_dev < min_ppl:
                    self.T_inizio_averagin = self.k 
                    print("gugu")
                self.validation_logs.append(ppl_dev)
                self.t_validation_checks += 1
