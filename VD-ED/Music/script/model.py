import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class VAE(nn.Module):
    def __init__(self, 
                 roll_dims,  
                 hidden_dims, 
                 infor_dims, 
                 n_step,  
                 condition_dims, 
                 k=700):
        super(VAE, self).__init__()
        self.grucell_1 = nn.GRUCell(roll_dims + infor_dims, hidden_dims) 
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_1 = nn.Linear(infor_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)

        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1 
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])
    def _sampling(self, x): 
        idx = x.max(1)[1] 
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def final_decoder(self, infor,length):  
        silence = torch.zeros(self.roll_dims)
        silence[-1] = -1
        out = torch.zeros((infor.size(0), self.roll_dims)) 
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(infor)) 
        hx[0] = t  
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step): 
            out = torch.cat([out, infor], 1) 
            hx[0] = self.grucell_1(out, hx[0])  
            if i == 0: 
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:  
                    out = self.sample[:, i, :]
                else:  
                    out = self._sampling(out)
                self.eps = self.k / (self.k + torch.exp(self.iteration / self.k)) 
            else:
                out = self._sampling(out)
        x = torch.stack(x, 1)

        
        for j in range(x.shape[0]):
            if length[j] != self.n_step :
                x[j,length[j]-self.n_step:] = silence
        
        return x

    def decoder(self, pitch, rhythm, condition=None):
        infor = torch.cat((pitch , rhythm)).unsqueeze(0)
        length = torch.sum(rhythm).int().unsqueeze(0)
        return self.final_decoder(infor, length)

    def forward(self, infor,x,length):
        if self.training:
            self.sample = x
            self.iteration += 1
        recon = self.final_decoder(infor,length) 
        return recon