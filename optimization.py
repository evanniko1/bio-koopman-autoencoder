import optuna

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn.init as init

from read_dataset_upd import data_from_name, discrete_data_format, train_test
from model import *
from deepkan import *
from tools import *
from train import *

import os

torch.use_deterministic_algorithms(False)

#==============================================================================
# Training settings
#==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='koopmanAE', help='model to train (multilayer perceptron or KAN)')
#
parser.add_argument('--train_size', type=float, default=0.5, help='size of the training set')
#
parser.add_argument('--alpha', type=int, default=1,  help='model width')
alpha_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
parser.add_argument('--dataset', type=str, default='flow_noisy', metavar='N', help='dataset')
#
parser.add_argument('--orthogonal_projection', type=bool, default=False, help='orthogonal projection of data to higher dimensions')
#
parser.add_argument('--theta', type=float, default=2.4,  metavar='N', help='angular displacement')
#
parser.add_argument('--noise', type=float, default=0.0,  metavar='N', help='noise level')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
lr_range = [1e-2, 1e-3, 1e-4]
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
batch_range = [32,64, 128, 256]
#
parser.add_argument('--batch_test', type=int, default=200, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
lambda_range = [1,1e-1, 1e-2, 1e-3, 1e-4]
#
parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
#
parser.add_argument('--eta', type=float, default='1e-2',  help='tune consistent loss')
#
parser.add_argument('--steps', type=int, default='8',  help='steps for learning forward dynamics')
steps_range = [1, 2, 4, 8, 16, 32]
#
parser.add_argument('--steps_back', type=int, default='8',  help='steps for learning backwards dynamics')
steps_back_range = [1, 2, 4, 8, 16, 32]
#
parser.add_argument('--bottleneck', type=int, default='6',  help='size of bottleneck layer')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[30, 200, 400, 500], help='decrease learning rate at these epochs')
#
parser.add_argument('--lr_decay', type=float, default='0.2',  help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--backward', type=int, default=0, help='train with backward dynamics')
#
parser.add_argument('--init_scale', type=float, default=0.99, help='init scaling')
#
parser.add_argument('--gradclip', type=float, default=0.05, help='gradient clipping')
gradclip_range = [0.01, 0.05, 0.1]
#
parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
#
parser.add_argument('--basis_function', type=str, default='chebyshev', help='alternatives to b-splines for KANs')
#
parser.add_argument('--degree', type=int, default=4, help='degree for polynomials')
degree_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#
parser.add_argument('--seed', type=int, default='1',  help='seed value')
# add opt_params to parser as a list of parameters you want to optimize
parser.add_argument('--opt_params', type=str, default='lr,lamb,alpha,batch,steps,steps_back,gradclip,degree', help='list of parameters to optimize')

args = parser.parse_args()


args.opt_params = args.opt_params.split(',')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
set_seed(args.seed)
device = get_device()


#******************************************************************************
# Create folder to save results
#******************************************************************************
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

#******************************************************************************
# load data
#******************************************************************************
if args.dataset == "discrete_spectrum":
     data = data_from_name(args.dataset, orthogonal_project=args.orthogonal_projection)
elif args.dataset == "isolated_repressilator":
    data = data_from_name(args.dataset)
elif args.dataset == "duffing_oscillator":
    data = data_from_name(args.dataset)
elif args.dataset == "host_aware_repressilator":
    # load julia df -- ONE FOR NOW
    df = pd.read_csv("./results_perturbation_joint_induction_binding_rate.csv")

    data = pd.DataFrame()
    for idx in range(df.shape[0]):
        sol_df = pd.DataFrame()
        for col in df.columns:
            temp_list_sols = df.iloc[idx][col][1:-1].split(",")
            temp_list = []
            for elm in temp_list_sols:
                temp_list.append(float(elm))
            sol_df[col] = temp_list
            sol_df.index = [idx]*len(temp_list)
        data = pd.concat([data, sol_df])
else:
    X, Xclean, m, n = data_from_name(args.dataset, noise = args.noise, theta = args.theta, orthogonal_project = args.orthogonal_projection)
    Xtrain, Xtest = train_test(X, percent = args.train_size)
    Xtrain_clean, Xtest_clean = Xtrain, Xtest

#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************

# transfer to tensor
if "pendulum" in args.dataset:
    # in case we choose the pendulum dataset
    Xtrain, Xtrain_clean = add_channels(Xtrain), add_channels(Xtrain_clean)
    Xtest, Xtest_clean = add_channels(Xtest),add_channels(Xtest_clean)
    Xtrain, Xtrain_clean = torch.from_numpy(Xtrain).float().contiguous(), torch.from_numpy(Xtrain_clean).float().contiguous()
    Xtest, Xtest_clean = torch.from_numpy(Xtest).float().contiguous(), torch.from_numpy(Xtest_clean).float().contiguous()
else:
    X = discrete_data_format(data)
    X = add_channels(X)
    Xtrain, Xtest = train_test(X, percent = args.train_size)
    Xtrain_clean = Xtrain.clone()
    Xtest_clean = Xtest.clone()
    m, n = X.shape[2], X.shape[3] 

#******************************************************************************
# Create Dataloader objects
#******************************************************************************
trainDat = []
start = 0
for i in np.arange(args.steps,-1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat)
del(trainDat)

train_loader = DataLoader(dataset = train_data,
                          batch_size = args.batch,
                          shuffle = True)

testDat = []
start = 0 
for i in np.arange(args.steps, -1, -1):
    if i == 0:
        testDat.append(Xtest[start:].float())
    else:
        testDat.append(Xtest[start:-i].float())
    start +=  1

test_data = torch.utils.data.TensorDataset(*testDat)
del(testDat)

test_loader = DataLoader(dataset = test_data,
                         batch_size = args.batch,
                         shuffle = False)
#==============================================================================
# Model
#==============================================================================

if args.model == "koopmanAE":
    model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
    print('koopmanAE')
elif args.model == "koopmanAE_KAN":
    model = koopmanAE_KAN(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
    print("koopmanAE_KAN")
elif args.model == "koopmanAE_polyKAN":
    model = koopmanAE_polyKAN(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale, args.basis_function, args.degree)
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


def objective(trial):
    # Recreate the model for each trial
    if args.model == "koopmanAE":
        model = koopmanAE(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
    elif args.model == "koopmanAE_KAN":
        model = koopmanAE_KAN(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale)
    elif args.model == "koopmanAE_polyKAN":
        model = koopmanAE_polyKAN(m, n, args.bottleneck, args.steps, args.steps_back, args.alpha, args.init_scale, args.basis_function, args.degree)
    
    model = model.to(device)

    # Default values for all parameters
    lr = args.lr
    batch = args.batch
    steps = args.steps
    steps_back = args.steps_back
    gradclip = args.gradclip
    degree = args.degree
    alpha = args.alpha
    lamb = args.lamb

    # Update parameters based on trial suggestions
    for opt_param in args.opt_params:
        if opt_param == 'lr':
            lr = trial.suggest_categorical('lr', lr_range)
        elif opt_param == 'batch':
            batch = trial.suggest_categorical('batch', batch_range)
        elif opt_param == 'steps':
            steps = trial.suggest_categorical('steps', steps_range)
        elif opt_param == 'steps_back':
            steps_back = trial.suggest_categorical('steps_back', steps_back_range)
        elif opt_param == 'gradclip':
            gradclip = trial.suggest_categorical('gradclip', gradclip_range)
        elif opt_param == 'degree':
            degree = trial.suggest_categorical('degree', degree_range)
        elif opt_param == 'alpha':
            alpha = trial.suggest_categorical('alpha', alpha_range)
        elif opt_param == 'lambda':
            lamb = trial.suggest_categorical('lamb', lambda_range)
        else:
            print('opt_param not found')
    
    # Train the model with selected hyperparameters
    model, optimizer, epoch_hist = train(model, train_loader,
                                        lr=lr,
                                        weight_decay=args.wd, 
                                        lamb=lamb, 
                                        num_epochs=args.epochs,
                                        learning_rate_change=args.lr_decay, 
                                        epoch_update=args.lr_update,
                                        nu=args.nu, 
                                        eta=args.eta, 
                                        backward=args.backward, 
                                        steps=steps, 
                                        steps_back=steps_back,
                                        gradclip=gradclip
                                        )
    
    # Ensure models are in evaluation mode
    loss_function = nn.MSELoss()
    model.encoder.eval()
    model.decoder.eval()

    # Calculate reconstruction loss
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for data in test_loader:
            # Move data to the appropriate device
            inputs = data[0].to(device)
            
            # Forward pass through encoder and decoder
            latent_representation = model.encoder(inputs)
            reconstructions = model.decoder(latent_representation)
            
            # Calculate reconstruction loss
            loss = loss_function(reconstructions, inputs)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size for total loss

    # Average loss over the dataset
    average_loss = total_loss / len(test_data)
    print(f"Reconstruction Loss: {average_loss:.4f}")

    return average_loss

###################################################################################################
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=10, timeout=300)

print(f'The best parameters are: {study.best_params}')

###################################################################################################
# Save the study
joblib.dump(study, f'{args.folder}/study.pkl')



