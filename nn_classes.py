import torch.nn as nn
import torch
import torch.utils.data as data

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

class RNN_cell(nn.Module):
    def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
        super(RNN_cell, self).__init__()

        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_hidden, word):
        input_emb = self.W(word)
        prev_hidden_rep = self.U(prev_hidden)
        # ht = σ(Wx + Uht-1 + b)
        hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
        # yt = σ(Vht + b)
        output = self.output(hidden_state)
        return hidden_state, output

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
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
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
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
    

import torch 
import torch.nn as nn 
from torch.autograd import Variable 
 
"""
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
"""
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or dropout == 0:
            return x
        # Generate the dropout mask that's consistent across the time steps
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - dropout) / (1 - dropout)
        mask = mask.expand_as(x)  # Ensure the mask is the same size as the input
        return mask * x

'''
class LM_LSTM_2(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, 
                 emb_dropout=0.1, n_layers=1): 
        super(LM_LSTM_2, self).__init__() 
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index) 
        #do variational dropout 
        self.embedding_dropout = nn.Dropout(emb_dropout) #LockedDropout() 
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False) 
        self.pad_token = pad_index 
        #do variational dropout 
        self.output_dropout = nn.Dropout(emb_dropout) #LockedDropout() 
        self.output = nn.Linear(hidden_size, output_size) #CAMBIARE CON SOTMAX??? 
        #Use the same weights for both embedding and output layer -> WEIGHT TYING 
        self.output.weight  = self.embedding.weight 
 
    def forward(self, input_sequence): 
        emb = self.embedding(input_sequence) 
        lstm_out, _  = self.lstm(emb) 
        output = self.output(lstm_out).permute(0,2,1) 
        return output
''' 

import torch
import torch.optim as optim

class NTASGD(optim.Optimizer):
    def __init__(self, params, lr=1, n=5, weight_decay=0, fine_tuning=False):
        t0 = 0 if fine_tuning else 10e7
        defaults = dict(lr=lr, n=n, weight_decay=weight_decay, fine_tuning=fine_tuning, t0=t0, t=0, logs=[])
        super(NTASGD, self).__init__(params, defaults)

    def check(self, v):
        #print("gugu")
        for group in self.param_groups:
            #Training
            if (not group['fine_tuning'] and group['t0'] == 10e7) or (group['fine_tuning']):
                if group['t'] > group['n'] and v > min(group['logs'][:-group['n']]):
                    group['t0'] = self.state[next(iter(group['params']))]['step']
                    print("Non-monotonic condition is triggered!")
                    return True
                group['logs'].append(v)
                group['t'] += 1

    def lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
                               
    def step(self):
        #print(self.param_groups[0])
        
        for group in self.param_groups:
            print("GROUP:" )
            for p in group['params']:
                #print("p: ", p)
                grad = p.grad.data
                #print("grad:" , grad)
                state = self.state[p]
                #print("state:" , state)
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)
                state['step'] += 1
                # update parameter
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                #print("pensio sia il gradient step: ")
                p.data.add_(-group['lr'], grad)
                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)
                # update mu
                state['mu'] = 1 / max(1, state['step'] - group['t0'])

class LM_LSTM_2_WT(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1): 
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
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1): 
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
        