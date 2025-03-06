import json
import os
import optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import argparse

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


torch.use_deterministic_algorithms(False)
#================================================================================


batch_range = [32,64, 128, 256, 512]
lambda_range = [1,1e-1, 1e-2, 1e-3, 1e-4]
steps_range = [1, 2, 4, 8, 16]
steps_back_range = [1, 2, 4, 8, 16]
gradclip_range = [0.01, 0.05, 0.1]

args = get_args()
args.opt_params = args.opt_params.split(',')
args.save = False


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
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

# save configuration in json file
with open(f'{args.folder}/config.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
#==============================================================================
# Dataset
#==============================================================================
Xtrain, Xtest, Xtrain_clean, Xtest_clean, m, n = data_preprocessing(args)

train_loader, test_loader = create_dataloader(args, Xtrain, Xtest)

#==============================================================================
# Model
#==============================================================================

if args.model == "koopmanAE":
    model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back,args.hidden, args.alpha, args.init_scale)
    print('koopmanAE')
elif args.model == "koopmanAE_KAN":
    model = koopmanAE_KAN(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden,args.alpha, args.init_scale)
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


#================================================================================
# Objective function
#================================================================================

def objective(trial):
    # Update parameters based on trial suggestions
    for opt_param in args.opt_params:
        if opt_param == 'lr':
            args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        elif opt_param == 'batch':
            args.batch = trial.suggest_categorical('batch', batch_range)
        elif opt_param == 'steps':
            args.steps = trial.suggest_int('steps', 10, 50)
        elif opt_param == 'steps_back':
            args.steps_back = trial.suggest_categorical('steps_back', steps_back_range)
        elif opt_param == 'gradclip':
            args.gradclip = trial.suggest_categorical('gradclip', gradclip_range)
        elif opt_param == 'degree':
            args.degree = trial.suggest_int('degree', 2, 10)
        elif opt_param == 'alpha':
            # the depth of the network
            args.alpha = trial.suggest_int('alpha', 1, 20)
        elif opt_param == 'hidden':
            # the number of hidden layers
            args.hidden = trial.suggest_int('hidden', 0, 50,5)
        elif opt_param == 'bottleneck':
            # the size of the bottleneck layer
            args.bottleneck = trial.suggest_int('bottleneck', 4, 32)
        elif opt_param == 'lamb':
            args.lamb = trial.suggest_categorical('lamb', lambda_range)
        elif opt_param == 'spline_knots':
            args.spline_knots = trial.suggest_int('spline_knots', 2, 12)
        else:
            raise ValueError(f'Unknown optimization parameter: {opt_param}')
    
    # Create dataloaders
    Xtrain, Xtest, Xtrain_clean, Xtest_clean, m, n = data_preprocessing(args)

    train_loader, test_loader = create_dataloader(args, Xtrain, Xtest)

    # Create model
    model_map = {
        "koopmanAE": lambda: koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden, args.alpha, args.init_scale),
        "koopmanAE_KAN": lambda: koopmanAE_KAN(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden, args.alpha, args.init_scale,args.spline_knots),
        "koopmanAE_polyKAN": lambda: koopmanAE_polyKAN(m, n, args.bottleneck, args.steps, args.steps_back, args.hidden, args.alpha, args.init_scale, args.basis_function, args.degree)
    }
    
    model = model_map.get(args.model)()
    if model is None:
        raise ValueError(f'Unknown model type: {args.model}')
            
    model = model.to(device)
    # Create and run trainer
    trainer = Trainer(model, args, input_size=(m,n), device=device, 
                    train_loader=train_loader, test_loader=test_loader)
    
    trainer.train_KoopmanAE() 
    
    return trainer.evaluate_reconstruction(), trainer.evaluate_prediction()

# make device cpu
device = get_device()

#================================================================================
# Optimize
#================================================================================

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=args.num_trials,n_jobs=5)

print(f'The best parameters are: {study.best_trials}')

pareto_trials = study.best_trials

best_trial_reconstruction = min(pareto_trials, key=lambda trial: trial.values[0])
best_trial_prediction = min(pareto_trials, key=lambda trial: trial.values[1])

print(f'The best trial for reconstruction is: {best_trial_reconstruction}')
print(f'The best trial for prediction is: {best_trial_prediction}')


# save image
fig1_1 = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[0], target_name="Reconstruction")
fig1_2 = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[1], target_name="Prediction")
fig2 = optuna.visualization.plot_slice(study,target=lambda t: t.values[1], target_name="Prediction")
fig3 = optuna.visualization.plot_pareto_front(study,target_names=["Reconstruction","Prediction"])

fig1_1.write_image(f'{args.folder}/param_importances_reconstruction.png')
fig1_2.write_image(f'{args.folder}/param_importances_prediction.png')
fig2.write_image(f'{args.folder}/slice.png')
fig3.write_image(f'{args.folder}/pareto_front.png')

# Save the study
joblib.dump(study, f'{args.folder}/study.pkl')