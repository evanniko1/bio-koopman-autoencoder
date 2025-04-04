import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import torch.multiprocessing as mp

# Import local modules
from utils.tools import set_seed, get_device
from utils.config_args import get_args
from utils.Viz import *
from KoopmanTrainer import Trainer
from data_handler import KoopmanDataHandler
from models.autoencoder import AutoEncoder
from models.koopman_operator import Knet


try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # method already set

# Fix deterministic behavior
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

# Device is cuda else cpu
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
    json.dump(args_dict, f,indent=2)

#==============================================================================
# Load and prepare dataset
#==============================================================================
# Convert args to a dictionary for the data handler

config = {
    "dataset": args.dataset,
    "data_dir": args.data_dir,
    "train_size": args.train_size,
    "val_size": args.val_size,
    "batch_size": args.batch_size,
    "noise": args.noise,
    "orthogonal_projection": args.orthogonal_projection,
    "num_combinations": args.num_combinations,
    "num_samples": args.num_samples,
    "time_steps": args.time_steps,
    "max_time": args.max_time,
    "normalize": args.normalize,
    "prediction_length": args.prediction_length,
    "stride": args.stride,
    "device": device,
    "theta": args.theta
}

# Initialize data handler
data_handler = KoopmanDataHandler(config)
data_handler.load_data()
preprocessed_data = data_handler.preprocess()
dataloaders = data_handler.create_dataloaders()

train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

# Get data dimensions for model initialization
input_size = preprocessed_data['Xtrain'].shape[-1]


#==============================================================================
# Model
#==============================================================================
# Model definition
ae = AutoEncoder(
    input_size=input_size,
    encoded_size=args.bottleneck,
    encoder_hidden_layers=args.encoder_hidden_layers,
    decoder_hidden_layers=args.decoder_hidden_layers,
    network_type=args.network_type,
    batch_norm=False
)
knet = Knet(size=args.bottleneck)

# Combine AE and Knet as a ModuleDict
model = torch.nn.ModuleDict({'ae': ae, 'knet': knet})
model = model.to(device)

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('Total params: %.2fk' % (sum(p.numel() for p in model.parameters())/1000.0))
print('************')

#==============================================================================
# Model Architecture
#==============================================================================
print('**** Model Architecture ****')
print(model)
print('***************************')
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
trainer = Trainer(model, args, device, do_eval=True, train_loader=train_loader, val_loader=val_loader)

trainer.train_KoopmanAE()



#==============================================================================
# Visualization
#==============================================================================
# Get a test sample

traj = preprocessed_data["Xtest"][8].to(device)  # Get first trajectory from batch

# Get reconstructed trajectory
model.eval()
with torch.no_grad():
    Y, reconstructed_traj = model.ae(traj)


# Get predicted trajectory
predicted_traj = trainer.predict_new(X0=traj[0],steps=len(traj)-1)

# Convert to numpy for plotting
traj = traj.cpu().detach().numpy()
reconstructed_traj = reconstructed_traj.cpu().detach().numpy()
predicted_traj = predicted_traj.cpu().detach().numpy()

# Plot trajectories
if traj.shape[1] <= 3:
    plot_trajectories(traj, predicted_traj, reconstructed_traj, folder_path=folder_path)
else:
    plot_trajectories(traj[:, :3], predicted_traj[:, :3], reconstructed_traj[:, :3], folder_path=folder_path)
plot_trajectory_comparison(traj, predicted_traj, reconstructed_traj, folder_path=folder_path)
violin_plot(model, preprocessed_data["Xtest"], device, folder_path)



#==============================================================================
# print eigenvalues and visualize them
#==============================================================================

eigenvalues,eigenvectors = torch.linalg.eig(model.knet.net.weight)

# arange eigenvalues in descending order
eigenvalues = eigenvalues.cpu().detach().numpy()
idx = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues[idx]

# print eigenvalues
print('Eigenvalues:')
print(eigenvalues)

plot_eigenvalues_on_unit_circle(eigenvalues, folder_path)

trainer.test_KoopmanAE(test_loader)


print(f"Training and evaluation completed. Results saved in {folder_path}")