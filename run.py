import argparse
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.init as init

from utils.read_dataset_upd import data_from_name, discrete_data_format, train_test, rescale
from Models.model import *
from Models.deepkan import *
from utils.tools import *
from KoopmanTrainer import *
from utils.Viz import *
from utils.config_args import *

import os
####
import torch.multiprocessing as mp
####
mp.set_start_method('spawn')

torch.use_deterministic_algorithms(False)
#==============================================================================
# Training settings
#==============================================================================
args = get_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)

# device is cuda else cpu
device = get_device()

#******************************************************************************
# Create folder to save results
#******************************************************************************
experiment_folder = 'experiments'
if not os.path.isdir(experiment_folder):
    os.mkdir(experiment_folder)

folder_path = os.path.join(experiment_folder, args.folder)
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

#******************************************************************************
# save arguments in json file
#******************************************************************************
args_dict = vars(args)
with open(os.path.join(folder_path, 'args.json'), 'w') as f:
    json.dump(args_dict, f)
    

#==============================================================================
# Dataset
#==============================================================================
Xtrain, Xtest, Xtrain_clean, Xtest_clean, m, n= data_preprocessing(args)

train_loader, test_loader = create_dataloader(args, Xtrain, Xtest)

#==============================================================================
# Model
#==============================================================================

if args.model == "koopmanAE":
    model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back,args.hidden, args.alpha, args.init_scale)
    print('koopmanAE')
elif args.model == "koopmanAE_KAN":
    model = koopmanAE_KAN(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden,args.alpha, args.init_scale,args.spline_knots)
    print("koopmanAE_KAN")
elif args.model == "koopmanAE_polyKAN":
    model = koopmanAE_polyKAN(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden,args.alpha, args.init_scale, args.basis_function, args.degree)
    print("koopmanAE_polyKAN")
#model = torch.nn.DataParallel(model)
model = model.to(device)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')
print(model)

#==============================================================================
# Load from checkpoint
#==============================================================================
if args.load_from_checkpoint:
    checkpoint_path = os.path.join(folder_path, 'model.pkl')
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Model loaded from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')

#==============================================================================
# Training 
#==============================================================================

Trainer = Trainer(model, args, input_size = (m,n), device = device, train_loader=train_loader, test_loader=test_loader)

if args.policy == 'KoopmanAE':
    Trainer.train_KoopmanAE()
elif args.policy == 'AE':
    Trainer.train_AE()
elif args.policy == 'Koopman':
    Trainer.train_Koopman()
elif args.policy == 'sequential':
    Trainer.train_sequential()
elif args.policy == 'Custom':
    # Custom training policy
    Trainer.train_custom()

#==============================================================================
# Visualization
#==============================================================================

if args.policy == 'AE':
    plot_recon_trajectory(model, Xtest, device,folder=folder_path, num_trajectories=10, traj_steps = args.time_steps)
    violin_plot(model, Xtest, device,folder=folder_path)
else:
    print('Plotting prediction')
    plot_combined_trajectory(model, Xtrain, device, folder=folder_path, num_trajectories=2,prediction_steps = 50,traj_steps = args.time_steps)



