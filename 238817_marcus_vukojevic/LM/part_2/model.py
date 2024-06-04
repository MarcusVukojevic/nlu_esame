import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from torch.optim import Optimizer
 
 

# In this class i've took the same class as in part1 and added weight tying 

class LM_LSTM_2_WT(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=3): 
        super(LM_LSTM_2_WT, self).__init__() 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) 
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight #wt

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
        

# La classe LockedDropout applica il dropout al tensore di input in modo coerente lungo la prima dimensione,
# mascherando interamente alcune unità lungo questa dimensione per ogni passo temporale o feature, 
# invece di applicare un mascheramento indipendente per ogni elemento.



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


# In this class i've implemented the weight tying regularization tech. and applyied the LockedDropout
# instead of the normal Dropout

class LM_LSTM_2_COMPLETA(nn.Module): 
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=3): 
        super(LM_LSTM_2_COMPLETA, self).__init__() 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # we lock the dropout
        self.embedding_dropout = LockedDropout() 
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) 
        # we lock the dropout
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


# The Nt_AvSGD class is a custom optimizer that extends the standard SGD optimizer by incorporating a 
# non-monotone interval averaging technique. This means it performs standard SGD updates initially but starts 
# averaging the parameters after a certain number of epochs if validation performance improves.
# It maintains consistency by tracking the number of iterations and validation checks, and it uses these to determine when to switch from standard updates to parameter averaging.

class Nt_AvSGD(Optimizer):
    def __init__(self, params, lr=1, n=5, L=2):
        self.T_inizio_averagin = 0
        self.t_validation_checks = 0
        self.validation_logs = [] # list of logs
        self.k = 0 # total number of iterations
        self.n_checks = n # number of epochs that if we get higher than this number we start averaging (aka non monotone interval)
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

                grad = p.grad.data  # take the gradient

                if self.T_inizio_averagin > 0:
                    # Now i use the parameters with the mean
                    ax = state['ax']
                    ax -= group['lr'] * grad 
                    p.data.copy_(ax) 
                else:
                    # Normal update
                    p.data -= group['lr'] * grad
                
                # If i start averagin, i build by hand my new average of weights
                if self.T_inizio_averagin > 0:
                    state['mu'] = 1 / max(1, self.k - self.T_inizio_averagin)
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)  # i keep track 

        return loss

    # This method is used to monitor the performance on the validation set. It increments the iteration count and 
    # logs the validation perplexity (ppl_dev). If the validation performance improves after a certain number of checks, 
    # it starts averaging the parameters.
    def check(self, ppl_dev):
        self.k += 1  # number of increments (they are equal to the number of epochs)
        if self.k % self.L == 0:
            if self.T_inizio_averagin == 0: # if i didn't start averaging
                range_valido = max(0, len(self.validation_logs) - self.n_checks)
                min_ppl = min(self.validation_logs[:range_valido]) if range_valido > 0 else float('inf')
                
                if self.t_validation_checks > self.n_checks and ppl_dev < min_ppl:
                    self.T_inizio_averagin = self.k 
                self.validation_logs.append(ppl_dev)
                self.t_validation_checks += 1
