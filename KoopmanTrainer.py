import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

from utils.tools import *
from utils.read_dataset_upd import data_from_name, discrete_data_format, train_test, rescale
from Models.model import *
from Models.deepkan import *
from utils.tools import *
#from train import *

import os


def data_preprocessing(args):
    #******************************************************************************
    # load data
    #******************************************************************************
    if args.dataset == "discrete_spectrum":
        if not os.path.isfile(os.path.join(os.getcwd(), 'data', 'discrete_spectrum.pkl')):
            data = data_from_name(args.dataset, orthogonal_project=args.orthogonal_projection)
        else:
            data = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'discrete_spectrum.pkl'))
    elif args.dataset == "isolated_repressilator":
        if not os.path.isfile(os.path.join(os.getcwd(), 'data', 'duffing_oscillator_{}_{}_{}_{}.pkl'.format(args.num_combinations, args.num_samples, args.time_steps, args.max_time))):
            data = data_from_name(args.dataset,combi_n = args.num_combinations, combi_n_samples = args.num_samples, time_points = args.time_steps, time_intervals = args.max_time)
        else:
            data = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'duffing_oscillator_{}_{}_{}_{}.pkl'.format(args.num_combinations, args.num_samples, args.time_steps, args.max_time)))
    elif args.dataset == "duffing_oscillator":
        if not os.path.isfile(os.path.join(os.getcwd(), 'data', 'duffing_oscillator_{}_{}_{}_{}.pkl'.format(args.num_combinations, args.num_samples, args.time_steps, args.max_time))):
            data = data_from_name(args.dataset,combi_n = args.num_combinations, combi_n_samples = args.num_samples, time_points = args.time_steps, time_intervals = args.max_time)
        else:
            data = pd.read_pickle(os.path.join(os.getcwd(), 'data', 'duffing_oscillator_{}_{}_{}_{}.pkl'.format(args.num_combinations, args.num_samples, args.time_steps, args.max_time)))
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
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        X = add_channels(X)
        Xtrain, Xtest = train_test(X, percent = args.train_size)
        Xtrain_clean = Xtrain.clone()
        Xtest_clean = Xtest.clone()
        # rescalling the data
        Xtrain, Xtest = rescale(Xtrain,Xtest)
        Xtrain_clean, Xtest_clean = rescale(Xtrain_clean,Xtest_clean)
        m, n = X.shape[2], X.shape[3]

    return Xtrain, Xtest, Xtrain_clean, Xtest_clean, m, n


def create_dataloader(args, Xtrain, Xtest):
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

        for i in range(int(len(Xtrain)/args.time_steps)):
            traj = Xtrain[i*args.time_steps : (i+1)*args.time_steps-1].float()
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

        for i in range(int(len(Xtest)/args.time_steps)):
            traj = Xtest[i*args.time_steps : (i+1)*args.time_steps-1].float()
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
    return train_loader, test_loader


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): The current validation loss.
        """
        # Initialize best_loss if it's the first validation
        if self.best_loss is None:
            self.best_loss = val_loss
        # If there is an improvement (greater than min_delta), reset counter
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        # If no improvement, increase the counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
        

class Trainer:
    def __init__(self,model,args,input_size,device,train_loader,test_loader):
        self.args = args
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.device = get_device()
        self.input_size = input_size
        self.m = input_size[0]
        self.n = input_size[1]
        self.device = device
        self.model = model.to(device)
        self.num_epochs = args.epochs
        self.learning_rate_change = args.lr_decay
        self.epoch_update = args.lr_update
        self.backward = args.backward
        self.steps = args.steps
        self.steps_back = args.steps_back
        self.gradclip = args.gradclip
        self.lamb = args.lamb
        self.nu = args.nu
        self.eta = args.eta
        self.folder = args.folder
        self.save = args.save
        self.early_stopping = args.early_stopping
        self.ES_epochs = 0

        # get the training and testing data
        self.train_loader = train_loader
        self.test_loader = test_loader

        # self.optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Loss function
        self.criterion = nn.MSELoss().to(device)

    # scheduler
    def lr_scheduler(self, optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return self.optimizer
                    else:
                        return self.optimizer



    def train_KoopmanAE(self):
        epoch_hist = []
        loss_hist = []
        epoch_loss = []
        forward_loss = []
        recon_loss = []

        for epoch in range(self.num_epochs):
            #print(epoch)
            for batch_idx, data_list in enumerate(self.train_loader):
                self.model.train()
                out, out_back = self.model(data_list[0].to(self.device), mode='forward')


                for k in range(self.steps):
                    if k == 0:
                        loss_fwd = self.criterion(out[k], data_list[k+1].to(self.device))
                    else:
                        loss_fwd += self.criterion(out[k], data_list[k+1].to(self.device))

                
                loss_identity = self.criterion(out[-1], data_list[0].to(self.device)) * self.steps

                loss_bwd = 0.0
                loss_consist = 0.0

                loss_bwd = 0.0
                loss_consist = 0.0

                if self.backward == 1:
                    out, out_back = self.model(data_list[-1].to(self.device), mode='self.backward')
    

                    for k in range(self.steps_back):
                        
                        if k == 0:
                            loss_bwd = self.criterion(out_back[k], data_list[::-1][k+1].to(self.device))
                        else:
                            loss_bwd += self.criterion(out_back[k], data_list[::-1][k+1].to(self.device))
                            
                                
                    A = self.model.dynamics.dynamics.weight
                    B = self.model.backdynamics.dynamics.weight

                    K = A.shape[-1]

                    for k in range(1,K+1):
                        As1 = A[:,:k]
                        Bs1 = B[:k,:]
                        As2 = A[:k,:]
                        Bs2 = B[:,:k]

                        Ik = torch.eye(k).float().to(self.device)

                        if k == 1:
                            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                            torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                        else:
                            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                            torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k) 
                    #Ik = torch.eye(K).float().to(device)
                    #loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
                    #torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
    
                loss = loss_fwd + self.lamb * loss_identity +  self.nu * loss_bwd + self.eta * loss_consist

                # ===================self.backward====================
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip) # gradient clip
                self.optimizer.step()           

            # schedule learning rate decay    
            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)
            loss_hist.append(loss.item())                
            epoch_loss.append(epoch)
            forward_loss.append(loss_fwd.item())
            recon_loss.append(loss_identity.item())
            
            
            if (epoch) % 20 == 0:
                    print('********** Epoche %s **********' %(epoch+1))
                    
                    print("loss identity: ", loss_identity.item())
                    if self.backward == 1:
                        print("loss self.backward: ", loss_bwd.item())
                        print("loss consistent: ", loss_consist.item())
                    print("loss forward: ", loss_fwd.item())
                    print("loss sum: ", loss.item())

                    epoch_hist.append(epoch+1) 

                    if hasattr(self.model.dynamics, 'dynamics'):
                        w, _ = np.linalg.eig(self.model.dynamics.dynamics.weight.data.cpu().numpy())
                        print(np.abs(w))


        if self.backward == 1:
            loss_consist = loss_consist.item()
                
        if self.save:
            # visualising the losses
            # reconstruction loss
            self.visualize(forward_loss, 'Prediction')
            self.visualize(recon_loss, 'Reconstruction')
            self.visualize(loss_hist, 'Total')

            # save the model
            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))

            # save the losses
            np.save(os.path.join('experiments', self.folder, 'loss_hist.npy'), loss_hist)
            np.save(os.path.join('experiments', self.folder, 'recon_loss.npy'), recon_loss)
            np.save(os.path.join('experiments', self.folder, 'forward_loss.npy'), forward_loss)
        ########################
        return self.model, self.optimizer, [epoch_hist, loss_fwd.item(), loss_consist]
        
    def train_AE(self):
        
        epoch_hist = []
        loss_hist = []
        epoch_loss = []
        validation_losses = []

        ES = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(self.num_epochs):
            for batch_idx, data_list in enumerate(self.train_loader):
                self.model.train()
                out = self.model.encoder(data_list[0].to(self.device))
                out = self.model.decoder(out)
                loss = self.criterion(out, data_list[0].to(self.device))

                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # schedule learning rate decay
            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)
            loss_hist.append(loss.item())
            epoch_loss.append(epoch)

            # validation
            val_loss = self.evaluate_reconstruction()
            validation_losses.append(val_loss)

            if (epoch) % 20 == 0:
                    print('********** Epoche %s **********' %(epoch+1))
                    print("Reconstruction Loss: ", loss.item())
                    print("Validation loss: ", val_loss)

                    epoch_hist.append(epoch+1)
            
            # early stopping
            if self.early_stopping:
                ES(val_loss)
                if ES.early_stop:
                    print("Early stopping triggered at epoch ", epoch)
                    self.ES_epochs = epoch
                    break
        
        if self.save:
            # visualising the losses
            self.visualize(loss_hist, 'Reconstruction')
            # visualising the validation loss
            self.visualize(validation_losses, 'Reconstruction_Validation')

            # save the model
            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))

            # save the losses
            np.save(os.path.join('experiments', self.folder, 'loss_hist.npy'), loss_hist)
        
        return self.model, self.optimizer, loss_hist

    def train_Koopman(self):
        
        # This setup will only train the forward dynamics
        epoch_hist = []
        loss_hist = []
        epoch_loss = []
        validation_losses = []

        ES = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(self.num_epochs):
            for batch_idx, data_list in enumerate(self.train_loader):
                self.model.train()
                out, out_back = self.forward_pass(data_list)
                for k in range(self.steps):
                    if k == 0:
                        loss_fwd = self.criterion(out[k], data_list[k+1].to(self.device))
                    else:
                        loss_fwd += self.criterion(out[k], data_list[k+1].to(self.device))
                loss_identity = self.criterion(out[-1], data_list[0].to(self.device)) * self.steps

                loss = loss_fwd + self.lamb * loss_identity

                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip)
                self.optimizer.step()

            # schedule learning rate decay
            self.lr_scheduler(self.optimizer, epoch, lr_decay_rate=self.learning_rate_change, decayEpoch=self.epoch_update)
            loss_hist.append(loss.item())
            epoch_loss.append(epoch)

            # evaluation
            val_loss = self.evaluate_prediction()
            validation_losses.append(val_loss)

            if (epoch) % 20 == 0:
                    print('********** Epoche %s **********' %(epoch+1))
                    print("loss identity: ", loss_identity.item())
                    print("loss forward: ", loss_fwd.item())
                    print("loss sum: ", loss.item())
                    print("Validation loss: ", val_loss)

                    epoch_hist.append(epoch+1)

                    if hasattr(self.model.dynamics, 'dynamics'):
                        w, _ = np.linalg.eig(self.model.dynamics.dynamics.weight.data.cpu().numpy())
                        print(np.abs(w))



            # early stopping
            if self.early_stopping:
                ES(val_loss)
                if ES.early_stop:
                    print("Early stopping triggered at epoch ", epoch)
                    self.ES_epochs = epoch
                    break
            
        if self.save:
            # visualising the losses
            self.visualize(loss_hist, 'Prediction')
            
            # visualising the validation loss
            self.visualize(validation_losses, 'Prediction_Validation')
            
            # save the model
            torch.save(self.model.state_dict(), os.path.join('experiments', self.folder, 'model.pkl'))

            # save the losses
            np.save(os.path.join('experiments', self.folder, 'loss_hist.npy'), loss_hist)

        return self.model, self.optimizer, [epoch_hist, loss_fwd.item(), loss_identity]
    
    def train_sequential(self):
        # for the number of epochs the first 20% of the epochs will be used to train the autoencoder and the rest will be used to train the entire process
        total_epochs = self.num_epochs
        self.num_epochs = int(total_epochs * 0.2)
        self.save = False
        self.early_stopping = True
        self.train_AE()
        self.save = True
        self.num_epochs = total_epochs - self.ES_epochs
        self.early_stopping = False
        self.train_KoopmanAE()

    def train_custom(self):
        pass

    def forward_pass(self, data_list):
        out = []
        out_back = []
        with torch.no_grad():
            z = self.model.encoder(data_list[0].to(self.device))
        q = z.contiguous()
        for _ in range(self.steps):
            q = self.model.dynamics(q)
            with torch.no_grad():
                out.append(self.model.decoder(q))
        out.append(self.model.decoder(z.contiguous()))
        return out, out_back
    
    def visualize(self,loss,label):

        fig = plt.figure(figsize=(15,12))
        plt.plot(loss, 'o--', lw=3, label='Training ' + label + ' loss', color='#377eb8')
        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.locator_params(axis='y', nbins=10)
        plt.locator_params(axis='x', nbins=10)
        plt.ylabel('Loss', fontsize=22)
        plt.xlabel('Epoch', fontsize=22)
        plt.grid(False)
        plt.legend(fontsize=22)
        fig.tight_layout()
        plt.savefig(os.path.join('experiments', self.folder, f'{label}.png'))
        plt.close()    

    
    def evaluate_reconstruction(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data_list in enumerate(self.test_loader):
                out = self.model.encoder(data_list[0].to(self.device))
                out = self.model.decoder(out)
                loss = self.criterion(out, data_list[0].to(self.device))
                losses.append(loss.item())
        avg_loss = np.mean(losses)
        return avg_loss


    def evaluate_prediction(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data_list in enumerate(self.test_loader):
                
                out, out_back = self.model(data_list[0].to(self.device), mode='forward')
                for k in range(self.steps):
                    if k == 0:
                        loss_fwd = self.criterion(out[k], data_list[k+1].to(self.device))
                    else:
                        loss_fwd += self.criterion(out[k], data_list[k+1].to(self.device))
                losses.append(loss_fwd.item())
        avg_loss = np.mean(losses)
        return avg_loss



