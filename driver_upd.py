import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=600, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=200, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='test',  help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='1',  help='balance between reconstruction and prediction loss')
#
parser.add_argument('--nu', type=float, default='1e-1',  help='tune backward loss')
#
parser.add_argument('--eta', type=float, default='1e-2',  help='tune consistent loss')
#
parser.add_argument('--steps', type=int, default='8',  help='steps for learning forward dynamics')
#
parser.add_argument('--steps_back', type=int, default='8',  help='steps for learning backwards dynamics')
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
#
parser.add_argument('--pred_steps', type=int, default='1000',  help='prediction steps')
#
parser.add_argument('--basis_function', type=str, default='chebyshev', help='alternatives to b-splines for KANs')
#
parser.add_argument('--degree', type=int, default=4, help='degree for polynomials')
#
parser.add_argument('--seed', type=int, default='1',  help='seed value')
#
parser.add_argument('--policy', type=str, default='KoopmanAE',  help='training policy')
#
args = parser.parse_args()

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
    print('Loading data...')
    X, Xclean, m, n = data_from_name(args.dataset, noise = args.noise, theta = args.theta, orthogonal_project = args.orthogonal_projection)
    Xtrain, Xtest = train_test(X, percent = args.train_size)
    Xtrain_clean, Xtest_clean = Xtrain, Xtest

#******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
#******************************************************************************

# transfer to tensor
if "pendulum" in args.dataset:
    # in case we choose the pendulum dataset
    print('Pendulum dataset')
    print('the shape of the data is: ', Xtrain.shape)
    Xtrain, Xtrain_clean = add_channels(Xtrain), add_channels(Xtrain_clean)
    Xtest, Xtest_clean = add_channels(Xtest),add_channels(Xtest_clean)
    Xtrain, Xtrain_clean = torch.from_numpy(Xtrain).float().contiguous(), torch.from_numpy(Xtrain_clean).float().contiguous()
    Xtest, Xtest_clean = torch.from_numpy(Xtest).float().contiguous(), torch.from_numpy(Xtest_clean).float().contiguous()
else:
    X = discrete_data_format(data)
    # scalling the data
    #X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    # concatenate dimension 0 and 1 in one dimension 
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    X = add_channels(X)
    Xtrain, Xtest = train_test(X, percent = args.train_size)
    Xtrain_clean = Xtrain.clone()
    Xtest_clean = Xtest.clone()
    m, n = X.shape[2], X.shape[3] 

#******************************************************************************
# Create Dataloader objects
#******************************************************************************

if args.dataset == "pendulum":
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

else:
    trainDat = [torch.empty(0) for _ in range(args.steps + 1)]

    for i in range(int(len(Xtrain)/50)):
        traj = Xtrain[i*50 : (i+1)*50-1].float()
        start = 0
        for j in np.arange(args.steps,-1, -1):
            if j == 0:
                trainDat[0] = torch.cat((trainDat[0], traj[start:].float()), dim=0)
            else:
                trainDat[j] = torch.cat((trainDat[j], traj[start:-j].float()), dim=0)
            start += 1

    train_data = torch.utils.data.TensorDataset(*trainDat)
    del(trainDat)

    train_loader = DataLoader(dataset = train_data,
                            batch_size = args.batch,
                            shuffle = True)

    testDat = [torch.empty(0) for _ in range(args.steps + 1)]

    for i in range(int(len(Xtest)/50)):
        traj = Xtest[i*50 : (i+1)*50-1].float()
        start = 0
        for j in np.arange(args.steps,-1, -1):
            if j == 0:
                testDat[0] = torch.cat((testDat[0], traj[start:].float()), dim=0)
            else:
                testDat[j] = torch.cat((testDat[j], traj[start:-j].float()), dim=0)
            start += 1

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


#==============================================================================
# Start training
#==============================================================================
model, optimizer, epoch_hist = train(model, train_loader,
                                     lr=args.lr,
                                     weight_decay=args.wd, 
                                     lamb=args.lamb, 
                                     num_epochs = args.epochs,
                                     learning_rate_change=args.lr_decay, 
                                     epoch_update=args.lr_update,
                                     nu = args.nu, 
                                     eta = args.eta, 
                                     backward=args.backward, 
                                     steps=args.steps, 
                                     steps_back=args.steps_back,
                                     gradclip=args.gradclip,
                                     policy=args.policy
                                     )


torch.save(model.state_dict(), args.folder + '/model'+'.pkl')

#******************************************************************************
# Plot loss against epoch
#******************************************************************************

if args.policy == 'AE':
    def plot_loss(epoch_hist, folder):
        fig = plt.figure(figsize=(15,12))
        plt.plot(epoch_hist, 'o--', lw=3, label='Training Reconstruction loss', color='#377eb8')
        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
        plt.ylabel('Loss', fontsize=22)
        plt.xlabel('Epoch', fontsize=22)
        plt.grid(False)
        plt.legend(fontsize=22)
        fig.tight_layout()
        plt.savefig(folder +'/000loss' +'.png')
        plt.close()

    plot_loss(epoch_hist, args.folder)
    exit()


#******************************************************************************
# Prediction
#******************************************************************************
Xinput, Xtarget = Xtest[:-1], Xtest[1:]
_, Xtarget = Xtest_clean[:-1], Xtest_clean[1:]

snapshots_pred = []
snapshots_truth = []


error = []
for i in range(30):
            error_temp = []
            init = Xinput[i].float().to(device)
            if i == 0:
                init0 = init
            
            z = model.encoder(init) # embedd data in latent space

            for j in range(args.pred_steps):
                if isinstance(z, tuple):
                    z = model.dynamics(*z) # evolve system in time
                else:
                    z = model.dynamics(z)
                if isinstance(z, tuple):
                    x_pred = model.decoder(z[0])
                else:
                    x_pred = model.decoder(z) # map back to high-dimensional space

                target_temp = Xtarget[i+j].data.cpu().numpy().reshape(m,n)
                error_temp.append(np.linalg.norm(x_pred.data.cpu().numpy().reshape(m,n) - target_temp) / np.linalg.norm(target_temp))
                
                if i == 0:
                    snapshots_pred.append(x_pred.data.cpu().numpy().reshape(m,n))
                    snapshots_truth.append(target_temp)
 
            error.append(np.asarray(error_temp))


error = np.asarray(error)

fig = plt.figure(figsize=(15,12))
plt.plot(error.mean(axis=0), 'o--', lw=3, label='', color='#377eb8')
plt.fill_between(x=range(error.shape[1]),y1=np.quantile(error, .05, axis=0), y2=np.quantile(error, .95, axis=0), color='#377eb8', alpha=0.2)

plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=10)

plt.ylabel('Relative prediction error', fontsize=22)
plt.xlabel('Time step', fontsize=22)
plt.grid(False)
#plt.yscale("log")
plt.ylim([0.0,error.max()*2])
#plt.legend(fontsize=22)
fig.tight_layout()
plt.savefig(args.folder +'/000prediction' +'.png')
#plt.savefig(args.folder +'/000prediction' +'.eps')

plt.close()

np.save(args.folder +'/000_pred.npy', error)

print('Average error of first pred: ', error.mean(axis=0)[0])
print('Average error of last pred: ', error.mean(axis=0)[-1])
print('Average error overall pred: ', np.mean(error.mean(axis=0)))
  
    
import scipy
save_preds = {'pred' : np.asarray(snapshots_pred), 'truth': np.asarray(snapshots_truth), 'init': np.asarray(init0.float().to(device).data.cpu().numpy().reshape(m,n))} 
scipy.io.savemat(args.folder +'/snapshots_pred.mat', dict(save_preds), appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

plt.close('all')
#******************************************************************************
# Eigenvalues
#******************************************************************************
model.eval()

#if hasattr(model.dynamics, 'dynamics'):
A =  model.dynamics.dynamics.weight.cpu().data.numpy()
#A =  model.module.test.data.cpu().data.numpy()
w, v = np.linalg.eig(A)
print(np.abs(w))

fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
plt.scatter(w.real, w.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')

maxeig = 1.4
plt.xlim([-maxeig, maxeig])
plt.ylim([-maxeig, maxeig])
plt.locator_params(axis='x',nbins=4)
plt.locator_params(axis='y',nbins=4)

#plt.xlabel('Real', fontsize=22)
#plt.ylabel('Imaginary', fontsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )

#plt.legend(loc="upper left", fontsize=16)
t = np.linspace(0,np.pi*2,100)
plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
plt.tight_layout()
plt.show()
plt.savefig(args.folder +'/000eigs' +'.png')
plt.savefig(args.folder +'/000eigs' +'.eps')
plt.close()

plt.close('all')

# Define the loss function
loss_function = nn.MSELoss()  # Or use nn.BCELoss() for binary data

# Move models to the appropriate device
model.encoder.to(device)
model.decoder.to(device)

# Ensure models are in evaluation mode
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
