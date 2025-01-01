import torch
import torch.nn.functional as F
import torch.optim as optim

from Models.model import *
from utils.tools import *
from Models.efficient_kan import *
from Models.polyKAN import *

class koopmanAE_KAN(nn.Module):
    def __init__(self, m, n, b, steps, steps_back,hidden = 2,alpha = 1, init_scale = 1,spline_knots = 5):
        super(koopmanAE_KAN, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.encoder = encoder_pyKAN(m, n, b,hidden=hidden, ALPHA=alpha, spline_knots=spline_knots)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoder_pyKAN(m, n, b,hidden=hidden, ALPHA=alpha, spline_knots=spline_knots)

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

class encoder_pyKAN(nn.Module):
    def __init__(self, m, n, b, hidden=2, ALPHA=1, spline_knots=5):
        super(encoder_pyKAN, self).__init__()
        self.N = m * n
        

        layers = [self.N]  # Input layer
        for _ in range(hidden):
            layers.append(16*ALPHA)  # Hidden layers
        layers.append(b)  # Output layer
        
        self.encoder = KAN(layers,grid_size=spline_knots)
        
    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.encoder(x)
        return x
        
class decoder_pyKAN(nn.Module):
    def __init__(self, m, n, b, hidden=2, ALPHA=1, spline_knots=5):
        super(decoder_pyKAN, self).__init__()
        self.m = m
        self.n = n 
        self.b = b
        
        layers = [b]  # Input layer
        for _ in range(hidden):
            layers.append(16*ALPHA)  # Hidden layers
        layers.append(m*n)  # Output layer
        
        self.decoder = KAN(layers,grid_size=spline_knots)

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.decoder(x)
        x = x.view(-1, 1, self.m, self.n)
        return x


class koopmanAE_polyKAN(nn.Module):
    def __init__(self, m, n, b, 
                 steps, 
                 steps_back, hidden = 2,
                 alpha = 1, init_scale = 1,
                 basis_function = 'chebyshev',
                 degree = 4):
        super(koopmanAE_polyKAN, self).__init__()
        self.steps = steps
        self.steps_back = steps_back

        self.encoder = encoder_polyKAN(m, n, b, basis_function, degree,hidden=hidden, ALPHA=alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoder_polyKAN(m, n, b, basis_function, degree,hidden=hidden, ALPHA=alpha)

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
        
class encoder_polyKAN(nn.Module):
    def __init__(self, m, n, b, basis_function, degree_poly,hidden = 2, ALPHA = 1):
        super(encoder_polyKAN, self).__init__()
        self.N = m * n 
        
        layers = [self.N]  # Input layer
        for _ in range(hidden):
            layers.append(16*ALPHA)
        layers.append(b)  # Output layer
        self.encoder = polynet(layers, poly_type=basis_function, degree=degree_poly)
        
    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.encoder(x)

        return x
        
class decoder_polyKAN(nn.Module):
    def __init__(self, m, n, b, basis_function, degree_poly,hidden = 2, ALPHA = 1):
        super(decoder_polyKAN, self).__init__()

        self.m = m
        self.n = n 
        self.b = b

        layers = [b]  # Input layer
        for _ in range(hidden):
            layers.append(16*ALPHA)
        layers.append(m*n)  # Output layer

        self.decoder = polynet(layers, poly_type=basis_function, degree=degree_poly)

    def forward(self, x):

        x = x.view(-1, 1, self.b)
        x = self.decoder(x)
        x = x.view(-1, 1, self.m, self.n)

        return x