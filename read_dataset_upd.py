import os
import random
import math
import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import ellipj, ellipk
from scipy.integrate import odeint

import torch

#******************************************************************************
# Read in data util functions
#******************************************************************************
def data_from_name(name, combi_n = 100, combi_n_samples = 50, time_points = 50, time_intervals = 10, noise = 0.0, theta=2.4, orthogonal_project=False):
    if name == 'pendulum_lin':
        return pendulum_lin(noise, orthog_project=orthogonal_project)      
    elif name == 'pendulum':
        return pendulum_lin(noise, theta, lin=False, orthog_project=orthogonal_project)    
    elif name == 'discrete_spectrum':
        x1range = [-3.1, 3.1]
        x2range = [-2,2]
        tSpan = np.linspace(0, 1, 51)
        mu = -0.05
        lamda = -1
        numICs = 5000
        seed = 42
        return DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lamda, seed)
    elif name == "isolated_repressilator":
        return isolated_repressilator_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals)
    elif name == "duffing_oscillator":
        return duffing_oscillator_fn(n=combi_n, num_samples=combi_n_samples, time_points=time_points, time_intervals=time_intervals)
    else:
        raise ValueError('dataset {} not recognized'.format(name))

def rescale(Xsmall, Xsmall_test):
    # rescale data
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test

def rotate_scale(samples_array, dims = (64,2)):

    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    # rotate
    samples_array = samples_array.T.dot(Q.T)        
    
    # scale 
    samples_array = 2 * (samples_array - np.min(samples_array)) / np.ptp(samples_array) - 1

    return samples_array

def train_test(samples_array, percent = 0.5):

    # simple split in two data partitions
    cutoff = int(samples_array.shape[0] * percent)
    train, test = samples_array[:cutoff], samples_array[cutoff:]
    
    return train, test

#******************************************************************************
# Dynamical Systems functions -- the actual generators
#******************************************************************************

def pendulum_lin(noise, theta=0.8, lin=True, orthog_project=False):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    if lin:
        X = sol(anal_ts, 0.8)
    else:
        X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    if orthog_project:
        # Rotate to high-dimensional space and scale
        X = rotate_scale(X)
        Xclean = rotate_scale(Xclean)

    return X, Xclean, 64, 1

class DiscreteSpectrum():

    def __init__(self, y0=None):
        super(DiscreteSpectrum, self).__init__()

        # Check initial values
        if y0 is None:

            self._y0 = np.array([1,1])

        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')

    def n_outputs(self):
        """ See :meth:`pints.ForwardModel.n_outputs()`. """
        return 2

    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. """
        return 2

    def _rhs(self, y, t, mu, lamda):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(2)
        dy[0] = mu * y[0]
        dy[1] = lamda*(y[1] - y[0]**2)

        return dy

    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """
        mu, lamda = parameters
        y = odeint(self._rhs, self._y0, times, (mu, lamda))
        return y[:, :]


    def suggested_parameters(self):
        """ See :meth:`pints.toy.ToyModel.suggested_parameters()`. """
        # Toni et al.:
        return np.array([-0.05, -1])


    def suggested_times(self):
        """ See :meth:`pints.toy.ToyModel.suggested_times()`. """
        # Toni et al.:
        return np.linspace(0, 1, 51)

def DiscreteSpectrumExampleFn(x1range, x2range, numICs, tSpan, mu, lamda, seed):

    # try some initial conditions for x1, x2
    random.seed(seed)

    # randomly start from x1range(1) to x1range(2)
    x1 = np.random.uniform(x1range[0], x1range[1], numICs)

    # randomly start from x2range(1) to x2range(2)
    x2 = np.random.uniform(x2range[0], x2range[1], numICs)

    lenT = len(tSpan)

    # make an empty dataframe
    data = pd.DataFrame()

    count = 1
    for j in range(numICs):
        # x1 and x2 are the initial conditions
        y1 = x1[j]
        y2 = x2[j]
        y0 = np.array([y1, y2])
        model = DiscreteSpectrum(y0=y0)
        xhat = model.simulate([mu, lamda], tSpan)
        # make xhat into pandas dataframe without column names
        xhat = pd.DataFrame(xhat)
        # make the data have the index 
        xhat.index = [j]*lenT
        # append the data
        data = pd.concat([data, xhat])

    return data

def discrete_data_format(df,chunk_size=1):
    X = []
    for i in tqdm.tqdm(df.index.unique()):
        x = torch.FloatTensor(df.loc[i].values)
        size = x.shape[0]
        if chunk_size > 1:
            size = int(size/chunk_size)
        x = torch.chunk(x,chunk_size)
        X.extend(x)
    X = torch.stack(X, 0)
    return X

# Repressilator model
class RepressilatorModel():
    """
    The "Repressilator" model describes oscillations in a network of proteins
    that suppress their own creation [1]_, [2]_.

    The formulation used here is taken from [3]_ and analysed in [4]_. It has
    three protein states (:math:`p_i`), each encoded by mRNA (:math:`m_i`).
    Once expressed, they suppress each other:

    .. math::
        \\dot{m_0} = -m_0 + \\frac{\\alpha}{1 + p_2^n} + \\alpha_0

        \\dot{m_1} = -m_1 + \\frac{\\alpha}{1 + p_0^n} + \\alpha_0

        \\dot{m_2} = -m_2 + \\frac{\\alpha}{1 + p_1^n} + \\alpha_0

        \\dot{p_0} = -\\beta (p_0 - m_0)

        \\dot{p_1} = -\\beta (p_1 - m_1)

        \\dot{p_2} = -\\beta (p_2 - m_2)

    With parameters ``alpha_0``, ``alpha``, ``beta``, and ``n``.

    Parameters
    ----------
    y0
        The system's initial state, must have 6 entries all >=0.

    References
    ----------
    .. [1] A Synthetic Oscillatory Network of Transcriptional Regulators.
          Elowitz, Leibler (2000) Nature.
          https://doi.org/10.1038/35002125

    .. [2] https://en.wikipedia.org/wiki/Repressilator

    .. [3] Dynamic models in biology. Ellner, Guckenheimer (2006) Princeton
           University Press

    .. [4] Approximate Bayesian computation scheme for parameter inference and
           model selection in dynamical systems. Toni, Welch, Strelkowa, Ipsen,
           Stumpf (2009) J. R. Soc. Interface.
           https://doi.org/10.1098/rsif.2008.0172
    """

    def __init__(self, y0=None):
        super(RepressilatorModel, self).__init__()

        # Check initial values
        if y0 is None:
            # Toni et al.:
            self._y0 = np.array([0, 0, 0, 2, 1, 3])
            # Figure 42 in book
            #self._y0 = np.array([0.2, 0.1, 0.3, 0.1, 0.4, 0.5], dtype=float)
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 6:
                raise ValueError('Initial value must have size 6.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        return 6

    def n_parameters(self):
        return 4

    def _rhs(self, y, t, alpha_0, alpha, beta, n):
        """
        Calculates the model RHS.
        """
        dy = np.zeros(6)
        dy[0] = -y[0] + alpha / (1 + y[5]**n) + alpha_0
        dy[1] = -y[1] + alpha / (1 + y[3]**n) + alpha_0
        dy[2] = -y[2] + alpha / (1 + y[4]**n) + alpha_0
        dy[3] = -beta * (y[3] - y[0])
        dy[4] = -beta * (y[4] - y[1])
        dy[5] = -beta * (y[5] - y[2])
        return dy

    def simulate(self, parameters, times):
        alpha_0, alpha, beta, n = parameters
        y = odeint(self._rhs, self._y0, times, (alpha_0, alpha, beta, n))
        return y[:, :]

    def suggested_parameters(self):
        # Toni et al.:
        return np.array([1, 1000, 5, 2])

        # Figure 42 in book:
        #return np.array([0, 50, 0.2, 2])

    def suggested_times(self):
        # Toni et al.:
        return np.linspace(0, 40, 400)

        # Figure 42 in book:
        #return np.linspace(0, 300, 600)

# Function to generate parameter combinations
def CombinationGenerator_repr(n):
    # Define the parameter ranges
    alpha_0_range = (0, 5)
    alpha_range = (100, 5000)
    beta_range = (0, 10)
    n_range = (0, 10)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    alpha_0_values = np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations)
    alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], num_combinations)
    beta_values = np.random.uniform(beta_range[0], beta_range[1], num_combinations)
    n_values = np.random.uniform(n_range[0], n_range[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'alpha_0': alpha_0_values,
        'alpha': alpha_values,
        'beta': beta_values,
        'n': n_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'alpha_0': np.random.uniform(alpha_0_range[0], alpha_0_range[1], num_combinations - len(combinations)),
            'alpha': np.random.uniform(alpha_range[0], alpha_range[1], num_combinations - len(combinations)),
            'beta': np.random.uniform(beta_range[0], beta_range[1], num_combinations - len(combinations)),
            'n': np.random.uniform(n_range[0], n_range[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator_repr(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = RepressilatorModel().suggested_times()
    
    # Create the model
    model = RepressilatorModel(y0=y)
    # Simulate the model
    data = model.simulate(combination, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data

# combination & number of samples
def generate_data_repr(n, num_samples):
    # Generate the combinations
    combinations = CombinationGenerator_repr(n)
    # transform combinations dataframe into a list of list
    combinations = combinations.values.tolist()
    # initial conditions
    if num_samples == 1:
        y = np.array([0, 0, 0, 2, 1, 3]).tolist()
    else:
        y = []
        for i in range(num_samples):
            y0 = np.random.uniform(0, 100, 6)
            y0 = y0.tolist()
            y.append(y0)
    
    # for each combination make a tuple of each combination and each member of y
    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))
    
    # shuffle the data
    np.random.shuffle(data)
    return data

def isolated_repressilator_fn(n, num_samples, time_points, time_intervals):
    # generate time points
    times = np.linspace(0, time_intervals, time_points)
    # generate parameter and initial values combinations
    combinations = generate_data_repr(n, num_samples)
    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        # call the ODE system solver
        sol = simulator_repr(combi[0], combi[1], times)
        # pass sol object into a dataframe
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx]*len(times)
        # concatenate existing dataframes
        dataset = pd.concat([dataset, sol_df])

    return dataset

# Duffing oscillator
class ForcedDuffingOscillator:
    """
    The Duffing oscillator is a nonlinear dynamic system characterized by a second-order differential equation
    that includes both linear and nonlinear restoring forces, described in [1] [2].

    \frac{d^2 x[0]}{dt^2} + \delta \frac{dx[0]}{dt} + \beta x[0]^3 + \alpha x[0] = \gamma \cos(\omega t)

    where delta is the damping force, beta is the nonlinear restoring force, alpha is the spring constant, 
    gamma is the amplitude of the driving force, and omega is the angular frequency.

    \dot{y[0]} = y[1]
    \dot{y[1]} = -\delta y[1] - \beta y[0]^3 - \alpha y[0] + \gamma \cos(\omega t)

    Parameters
    ----------
    y0
        The system's initial state, must have 2 entries all >=0.

    delta : float
        The damping force
    beta : float
        The nonlinear restoring force
    alpha : float
        The spring constant
    gamma : float
        The amplitude of the driving force
    omega : float
        The angular frequency

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Duffing_equation
    .. [2] State Identification of Duffing Oscillator Based on Extreme Learning Machine
           https://ieeexplore.ieee.org/abstract/document/8080207
    """

    def __init__(self, y0=None):
        # Check initial values
        if y0 is None:
            self._y0 = np.array([0, 0])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 2:
                raise ValueError('Initial value must have size 2.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """Returns the number of state variables."""
        return 2

    def n_parameters(self):
        """Returns the number of parameters."""
        return 5

    def _rhs(self, y, t, delta, beta, alpha, gamma, omega):
        """
        Calculates the right-hand side of the differential equations.
        
        Parameters
        ----------
        y : array_like
            Current state vector [position, velocity]
        t : float
            Current time
        delta, beta, alpha, gamma, omega : float
            System parameters
            
        Returns
        -------
        array_like
            Derivatives [dy[0]/dt, dy[1]/dt]
        """
        dy = np.zeros(2)
        dy[0] = y[1]
        dy[1] = -delta * y[1] - beta * y[0]**3 - alpha * y[0] + gamma * np.cos(omega * t)
        return dy

    def simulate(self, parameters, times):
        """
        Simulate the system for given parameters and time points.
        
        Parameters
        ----------
        parameters : array_like
            System parameters [delta, beta, alpha, gamma, omega]
        times : array_like
            Time points for simulation
            
        Returns
        -------
        array_like
            Solution array with shape (len(times), 2)
        """
        delta, beta, alpha, gamma, omega = parameters
        y = odeint(self._rhs, self._y0, times, args=(delta, beta, alpha, gamma, omega))
        return y

    def suggested_parameters(self):
        """Returns a set of example parameters that produce interesting behavior."""
        return np.array([0.2, 1.0, -1.0, 0.3, 1.2])

    def suggested_times(self):
        """Returns a suggested time array for simulation."""
        return np.linspace(0, 50, 1000)

# Function to generate parameter combinations
def CombinationGenerator_dfn(n):
    # Define the parameter ranges
    # delta, beta, alpha, gamma, omega
    delta_range = (0, 1)
    beta_range = (0, 1)
    alpha_range = (-1, 1)
    gamma_range = (0, 1)
    omega_range = (0, 2)

    # Set the number of combinations
    num_combinations = n

    # Use numpy to generate the combinations with uniform distribution
    delta_values = np.random.uniform(delta_range[0], delta_range[1], num_combinations)
    beta_values = np.random.uniform(beta_range[0], beta_range[1], num_combinations)
    alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], num_combinations)
    gamma_values = np.random.uniform(gamma_range[0], gamma_range[1], num_combinations)
    omega_values = np.random.uniform(omega_range[0], omega_range[1], num_combinations)

    # Combine the values into a DataFrame
    combinations = pd.DataFrame({
        'delta': delta_values,
        'beta': beta_values,
        'alpha': alpha_values,
        'gamma': gamma_values,
        'omega': omega_values
    })

    # Ensure uniqueness (though with uniform random generation, collisions are highly unlikely)
    combinations = combinations.drop_duplicates()

    # If there are not enough unique combinations, regenerate until we have enough
    while len(combinations) < num_combinations:
        additional_combinations = pd.DataFrame({
            'delta': np.random.uniform(delta_range[0], delta_range[1], num_combinations - len(combinations)),
            'beta': np.random.uniform(beta_range[0], beta_range[1], num_combinations - len(combinations)),
            'alpha': np.random.uniform(alpha_range[0], alpha_range[1], num_combinations - len(combinations)),
            'gamma': np.random.uniform(gamma_range[0], gamma_range[1], num_combinations - len(combinations)),
            'omega': np.random.uniform(omega_range[0], omega_range[1], num_combinations - len(combinations))
        })
        combinations = pd.concat([combinations, additional_combinations]).drop_duplicates()

    # Ensure we have exactly the desired number of unique combinations
    combinations = combinations.head(num_combinations)

    return combinations

# simulator function that takes the combination and the number of samples per combination and the suggested times
def simulator_dfn(combination,y,times=None):
    # If times is not provided, use the suggested times
    if times is None:
        times = ForcedDuffingOscillator().suggested_times()
    
    # Create the model
    model = ForcedDuffingOscillator(y0=y)
    # Simulate the model
    data = model.simulate(combination, times)
    
    # Add noise to the data
    #noise = np.random.normal(0, 0.1, data.shape)
    #data += noise
    # Return the data
    return data

# combination & number of samples
def generate_data_dfn(n, num_samples):
    # Generate the combinations
    combinations = CombinationGenerator_dfn(n)
    # transform combinations dataframe into a list of list
    combinations = combinations.values.tolist()
    # initial conditions
    if num_samples == 1:
        y = np.array([0, 0]).tolist()
    else:
        y = []
        for i in range(num_samples):
            y0 = np.random.uniform(0, 10, 2)
            y0 = y0.tolist()
            y.append(y0)
    
    # for each combination make a tuple of each combination and each member of y
    data = []
    for combination in combinations:
        for y0 in y:
            data.append((combination, y0))
    
    # shuffle the data
    np.random.shuffle(data)
    return data

def duffing_oscillator_fn(n, num_samples, time_points, time_intervals):
    # generate time points
    times = np.linspace(0, time_intervals, time_points)
    # generate parameter and initial values combinations
    combinations = generate_data_dfn(n, num_samples)
    dataset = pd.DataFrame()
    for idx, combi in enumerate(combinations):
        # call the ODE system solver
        sol = simulator_dfn(combi[0], combi[1], times)
        # pass sol object into a dataframe
        sol_df = pd.DataFrame(sol)
        sol_df.index = [idx]*len(times)
        # concatenate existing dataframes
        dataset = pd.concat([dataset, sol_df])

    return dataset