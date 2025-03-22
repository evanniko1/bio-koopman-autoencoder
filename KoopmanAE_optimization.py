import json
import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import joblib
import sys


from utils.tools import set_seed, get_device
from utils.config_args import get_args
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
args.opt_params = args.opt_params.split(',')
args.save = False

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

# Save configuration in json file
with open(os.path.join(folder_path, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

#================================================================================
# Parameter ranges for optimization
#================================================================================
batch_range = [32, 64, 128, 256, 512]
decoder_loss_weight_range = [1, 1e-1, 1e-2, 1e-3, 1e-4]
bottleneck_range = [10, 20, 30, 40, 50, 60]

#==============================================================================
# Load and prepare dataset outside the objective function
#==============================================================================
# Initialize data handler with initial config
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

# Get data dimensions for model initialization
input_size = preprocessed_data['Xtrain'].shape[-1]

# Store preprocessed data for reuse
train_data = preprocessed_data['Xtrain']
val_data = preprocessed_data['Xval']
test_data = preprocessed_data['Xtest']

#================================================================================
# Objective function
#================================================================================
def objective(trial):
    # Update parameters based on trial suggestions
    for opt_param in args.opt_params:
        if opt_param == 'lr':
            args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        elif opt_param == 'batch':
            args.batch_size = trial.suggest_categorical('batch_size', batch_range)
        elif opt_param == 'degree':
            args.degree = trial.suggest_int('degree', 2, 10)
        elif opt_param == 'spline_knots':
            args.spline_knots = trial.suggest_int('spline_knots', 2, 12)
        elif opt_param == 'bottleneck':
            args.bottleneck = trial.suggest_int('bottleneck', 10, 100, step=10)
        elif opt_param == 'decoder_loss_weight':
            args.decoder_loss_weight = trial.suggest_categorical('decoder_loss_weight', decoder_loss_weight_range)
        elif opt_param == 'encoder_layers':
            # Number of units in each hidden layer
            size = trial.suggest_int('encoder_size', 50, 200, step=50)
            depth = trial.suggest_int('encoder_depth', 1, 3)
            args.encoder_hidden_layers = [size] * depth
        elif opt_param == 'network_type':
            args.network_type = trial.suggest_categorical('network_type', ['MLP', 'KAN', 'PolyKAN'])
        else:
            raise ValueError(f'Unknown optimization parameter: {opt_param}')
    
    # Update batch size in config and create dataloaders with the new batch size
    data_handler.config['batch_size'] = args.batch_size
    dataloaders = data_handler.create_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Create model components
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

    # Create and run trainer
    trainer = Trainer(model, args, device, do_eval=True, 
                     train_loader=train_loader, val_loader=val_loader)
    
    # Only train for a smaller number of epochs for optimization trials
    original_epochs = args.epochs
    args.epochs = max(10, int(args.epochs * 0.2))
    trainer.num_epochs = args.epochs
    
    # Train the model
    stats = trainer.train_KoopmanAE()[2]
    
    # Restore original epochs
    args.epochs = original_epochs
    
    # Evaluate the model - use validation loss as optimization criterion
    reconstruction_loss = stats['recon_loss_va'][-1]
    prediction_loss = stats['pred_loss_va'][-1]
    
    return reconstruction_loss, prediction_loss

#================================================================================
# Run optimization
#================================================================================
study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=args.num_trials, n_jobs=1)

#================================================================================
# Save and visualize results
#================================================================================
print("Study statistics: ")
print(f"  Number of finished trials: {len(study.trials)}")

print("Best trials:")
pareto_trials = study.best_trials

best_trial_reconstruction = min(pareto_trials, key=lambda trial: trial.values[0])
best_trial_prediction = min(pareto_trials, key=lambda trial: trial.values[1])

print(f'Best trial for reconstruction error: {best_trial_reconstruction.number}')
print(f'  Value: {best_trial_reconstruction.values[0]}')
print(f'  Params: {best_trial_reconstruction.params}')

print(f'Best trial for prediction error: {best_trial_prediction.number}')
print(f'  Value: {best_trial_prediction.values[1]}')
print(f'  Params: {best_trial_prediction.params}')

# Save the best trial parameters to a JSON file
with open(os.path.join(folder_path, 'best_params_reconstruction.json'), 'w') as f:
    json.dump(best_trial_reconstruction.params, f, indent=2)
    
with open(os.path.join(folder_path, 'best_params_prediction.json'), 'w') as f:
    json.dump(best_trial_prediction.params, f, indent=2)

# Create visualization plots
try:
    # Parameter importance for reconstruction error
    fig1_1 = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="Reconstruction Error"
    )
    fig1_1.write_image(os.path.join(folder_path, 'param_importances_reconstruction.png'))
    
    # Parameter importance for prediction error
    fig1_2 = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[1], target_name="Prediction Error"
    )
    fig1_2.write_image(os.path.join(folder_path, 'param_importances_prediction.png'))
    
    # Slice plot for prediction error
    fig2 = optuna.visualization.plot_slice(
        study, target=lambda t: t.values[1], target_name="Prediction Error"
    )
    fig2.write_image(os.path.join(folder_path, 'slice_plot.png'))
    
    # Pareto front visualization
    fig3 = optuna.visualization.plot_pareto_front(
        study, target_names=["Reconstruction Error", "Prediction Error"]
    )
    fig3.write_image(os.path.join(folder_path, 'pareto_front.png'))
    
    # Contour plot if applicable
    if len(args.opt_params) >= 2:
        try:
            fig4 = optuna.visualization.plot_contour(
                study, target=lambda t: t.values[1], target_name="Prediction Error"
            )
            fig4.write_image(os.path.join(folder_path, 'contour_plot.png'))
        except:
            print("Could not create contour plot - requires at least two parameters.")
            
except ImportError:
    print("Could not create visualizations - please install plotly and kaleido for visualization support.")

# Save the study
joblib.dump(study, os.path.join(folder_path, 'study.pkl'))

print(f"Optimization completed. Results saved in {folder_path}")

# Optional: Train the model with the best parameters for prediction
print("Training model with best parameters for prediction...")

# Update args with best parameters
for param, value in best_trial_prediction.params.items():
    setattr(args, param, value)

# Create dataloaders with the best batch size
dataloaders = data_handler.create_dataloaders()
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]

# Create model with best parameters
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
best_model = torch.nn.ModuleDict({'ae': ae, 'knet': knet})
best_model = best_model.to(device)

# Save the best model
args.save = True
# Create and run trainer with full epochs
trainer = Trainer(best_model, args, device, do_eval=True, 
                 train_loader=train_loader, val_loader=val_loader)

# Train with full epochs
trainer.train_KoopmanAE()

# Save the best model
if args.save:
    torch.save(best_model.state_dict(), os.path.join(folder_path, 'best_model.pkl'))
    print(f"Best model saved to {os.path.join(folder_path, 'best_model.pkl')}")

print("Done!")