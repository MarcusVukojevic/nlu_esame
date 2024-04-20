
import torch
import torch.optim as optim


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
