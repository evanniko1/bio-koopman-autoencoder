from torch import nn
import torch

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, hidden=2, ALPHA=1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()


        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.N, 16*ALPHA))
        for _ in range(hidden):
            self.layers.append(nn.Linear(16*ALPHA, 16*ALPHA))
        self.layers.append(nn.Linear(16*ALPHA, b))

        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        
        for layer in self.layers:
            x = self.tanh(layer(x))


        
        return x

class decoderNet(nn.Module):
    def __init__(self, m, n, b, hidden=2, ALPHA=1):
        super(decoderNet, self).__init__()
        self.m = m
        self.n = n
        self.b = b
        self.tanh = nn.Tanh()
        

        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(b, 16*ALPHA))
        for _ in range(hidden):
            self.layers.append(nn.Linear(16*ALPHA, 16*ALPHA))
        self.layers.append(nn.Linear(16*ALPHA, m*n))
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        for layer in self.layers[:-1]:
            x = self.tanh(layer(x))  
        x = self.layers[-1](x)
        x = x.view(-1, 1, self.m, self.n)

        return x


class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x




class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back,hidden = 2,alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b,hidden=hidden, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b,hidden=hidden,ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
