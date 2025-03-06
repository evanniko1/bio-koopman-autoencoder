import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns


#1 plot entire trajectory based on reconstructed data

def recontruction(model, test_data, device):
    test_data = test_data.to(device)
    reconstructed_data = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_data)):
            out_encoded = model.encoder(test_data[i])
            out_decoded = model.decoder(out_encoded)
            reconstructed_data.append(out_decoded)

    # calculating the mean square error
    loss = torch.nn.MSELoss()
    loss_val = loss(torch.stack(reconstructed_data).squeeze(2), test_data)
    
    return reconstructed_data, loss_val

'''
def prediction(model, input_data, device, prediction_steps):

    input_data = input_data.to(device)
    
    with torch.no_grad():
        encoded = model.encoder(input_data)
        prediction = []
        for i in range(prediction_steps):
            K = model.dynamics(encoded)
            K = K.view(-1, 32, 32)
            encoded = torch.bmm(encoded, K)
            decoded = model.decoder(encoded)
            prediction.append(decoded)

    return prediction
'''

def prediction(model, input_data, device, prediction_steps):
    input_data = input_data.to(device)
    prediction = []
    with torch.no_grad():
        q = model.encoder(input_data)
        for i in range(prediction_steps):
            q = model.dynamics(q)
            prediction.append(model.decoder(q))
    return prediction
            
    



def plot_recon_trajectory(model, test_data, device, folder, num_trajectories=10, traj_steps=50):

    
    total_trajectories = test_data.shape[0] // traj_steps
    
    # Randomly sample trajectory indices
    trajectory_indices = torch.randperm(total_trajectories)[:num_trajectories]
    
    M = test_data.shape[2]  # number of dimensions
    fig, axes = plt.subplots(M, 1, figsize=(12, 3*M))
    
    # Store losses for each trajectory
    trajectory_losses = []
    all_reconstructed = []
    all_original = []
    
    # Process each sampled trajectory
    for idx in trajectory_indices:
        start_idx = idx * traj_steps
        end_idx = (idx + 1) * traj_steps
        trajectory_data = test_data[start_idx:end_idx]
        
        reconstructed_data, loss = recontruction(model, trajectory_data, device)
        reconstructed_data = torch.stack(reconstructed_data).squeeze(2)
        
        trajectory_losses.append(loss.item())
        all_reconstructed.append(reconstructed_data)
        all_original.append(trajectory_data)
    
    # Find trajectories with highest and lowest loss
    min_loss_idx = torch.argmin(torch.tensor(trajectory_losses))
    max_loss_idx = torch.argmax(torch.tensor(trajectory_losses))
    
    # Plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for dim in range(M):
        # Plot all trajectories with low opacity
        for i in range(num_trajectories):
            if i not in [min_loss_idx, max_loss_idx]:
                axes[dim].plot(all_original[i][:, 0, dim, 0].cpu(), 
                             color=colors[i], alpha=0.3, linestyle='-')
                axes[dim].plot(all_reconstructed[i][:, 0, dim, 0].cpu(), 
                             color=colors[i], alpha=0.3, linestyle='--')
        
        # plot best loss trajectories with distinct styles
        # best Original - solid green line with markers
        axes[dim].plot(all_original[min_loss_idx][:, 0, dim, 0].cpu(), 
                     color='green', linestyle='-', marker='o', markersize=4,
                     markevery=5, label='Best Original')
        
        # best reconstructed - dashed green line with different markers
        axes[dim].plot(all_reconstructed[min_loss_idx][:, 0, dim, 0].cpu(), 
                     color='green', linestyle='--', marker='s', markersize=4,
                     markevery=5, label='Best Reconstructed')
        
        # worst Original - solid red line with markers
        axes[dim].plot(all_original[max_loss_idx][:, 0, dim, 0].cpu(), 
                     color='red', linestyle='-', marker='^', markersize=4,
                     markevery=5, label='Worst Original')
        
        # worst Reconstructed - dashed red line with different markers
        axes[dim].plot(all_reconstructed[max_loss_idx][:, 0, dim, 0].cpu(), 
                     color='red', linestyle='--', marker='v', markersize=4,
                     markevery=5, label='Worst Reconstructed')
        
        axes[dim].set_title(f'Dimension {dim}')
        axes[dim].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        
        axes[dim].text(0.02, 0.98, 
                      f'Best Loss: {trajectory_losses[min_loss_idx]:.2e}\n'
                      f'Worst Loss: {trajectory_losses[max_loss_idx]:.2e}',
                      transform=axes[dim].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{folder}/multiple_trajectories.png', bbox_inches='tight', dpi=300)
    
    return min_loss_idx, max_loss_idx, trajectory_losses


def violin_plot(model, test_data, device, folder):
    """
    Create violin plots showing reconstruction error distribution across dimensions
    """
    # Get reconstructed data
    reconstructed_data, loss_val = recontruction(model, test_data, device)
    reconstructed_data = torch.stack(reconstructed_data).squeeze(2)
    
    M = test_data.shape[2]  # number of dimensions
    fig, axes = plt.subplots(2, M, figsize=(5*M, 8))
    
    for dim in range(M):
        # Original vs Reconstructed Distribution
        orig_vals = test_data[:, 0, dim, 0].cpu().numpy()
        recon_vals = reconstructed_data[:, 0, dim, 0].cpu().numpy()
        
        df_dist = pd.DataFrame({
            'value': np.concatenate([orig_vals, recon_vals]),
            'type': ['Original']*len(orig_vals) + ['Reconstructed']*len(recon_vals)
        })
        
        sns.violinplot(data=df_dist, x='type', y='value', ax=axes[0, dim])
        axes[0, dim].set_title(f'Dimension {dim} - Distribution')
        
        # Error Distribution
        error = orig_vals - recon_vals
        df_error = pd.DataFrame({'error': error})
        
        sns.violinplot(data=df_error, y='error', ax=axes[1, dim])
        axes[1, dim].set_title(f'Dimension {dim} - Reconstruction Error')
        
    plt.tight_layout()
    plt.savefig(f'{folder}/violin_plot.png')



def plot_combined_trajectory(model, test_data, device, folder, num_trajectories=10, 
                           prediction_steps=20, traj_steps=50):
    """
    Plot trajectories showing original data, reconstruction, and prediction.
    Uses the last step of reconstruction as input for prediction.
    """
    # Calculate steps
    torch.seed()
    reconstruct_steps = traj_steps - prediction_steps
    
    total_trajectories = test_data.shape[0] // traj_steps
    
    trajectory_indices = torch.randperm(total_trajectories)
    trajectory_indices = trajectory_indices[:num_trajectories]

    print(trajectory_indices)
    
    M = test_data.shape[2]  # number of dimensions
    fig, axes = plt.subplots(M, 1, figsize=(15, 4*M))
    
    # Store results for each trajectory
    reconstruction_losses = []
    all_reconstructed = []
    all_predicted = []
    all_original = []
    
    # Process each sampled trajectory
    for idx in trajectory_indices:
        start_idx = idx * traj_steps
        end_idx = (idx + 1) * traj_steps
        trajectory_data = test_data[start_idx:end_idx]
        
        # Split data
        reconstruction_data = trajectory_data[:reconstruct_steps]
        prediction_target = trajectory_data[reconstruct_steps:]
        
        # Get reconstructions
        reconstructed_data, rec_loss = recontruction(model, reconstruction_data, device)
        reconstructed_data = torch.stack(reconstructed_data).squeeze(2)
        reconstruction_losses.append(rec_loss.item())
        
        # Get predictions using last reconstructed step
        last_step = prediction_target[0] # Use last step of original data
        predicted_data = prediction(model, last_step, device, prediction_steps)
        predicted_data = torch.stack(predicted_data).squeeze(2)
        
        # Store results
        all_reconstructed.append(reconstructed_data)
        all_predicted.append(predicted_data)
        all_original.append(trajectory_data)

    

    
    # Find best and worst trajectories based on reconstruction loss
    min_loss_idx = torch.argmin(torch.tensor(reconstruction_losses))
    max_loss_idx = torch.argmax(torch.tensor(reconstruction_losses))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for dim in range(M):
        # Plot background trajectories
        for i in range(num_trajectories):
            if i not in [min_loss_idx, max_loss_idx]:
                # Original full trajectory
                axes[dim].plot(range(traj_steps), 
                             all_original[i][:, 0, dim, 0].cpu(),
                             color='gray', alpha=0.1, linestyle='-')
                
                # Reconstruction
                axes[dim].plot(range(reconstruct_steps),
                             all_reconstructed[i][:, 0, dim, 0].cpu(),
                             color='gray', alpha=0.1, linestyle='--')
                
                # Prediction
                axes[dim].plot(range(reconstruct_steps-1, traj_steps),
                             torch.cat([all_original[i][reconstruct_steps-1:reconstruct_steps],
                                      all_predicted[i]], dim=0)[:, 0, dim, 0].cpu(),
                             color='gray', alpha=0.1, linestyle=':')
        
        # Plot best trajectory
        # Original
        axes[dim].plot(range(traj_steps),
                      all_original[min_loss_idx][:, 0, dim, 0].cpu(),
                      color='green', linestyle='-', marker='o', markersize=4,
                      markevery=5, label='Best Original', linewidth=2)
        
        # Reconstruction
        axes[dim].plot(range(reconstruct_steps),
                      all_reconstructed[min_loss_idx][:, 0, dim, 0].cpu(),
                      color='blue', linestyle='--', marker='s', markersize=4,
                      markevery=5, label='Best Reconstruction', linewidth=2)
        
        # Prediction
        axes[dim].plot(range(reconstruct_steps-1, traj_steps),
                      torch.cat([all_original[min_loss_idx][reconstruct_steps-1:reconstruct_steps],
                               all_predicted[min_loss_idx]], dim=0)[:, 0, dim, 0].cpu(),
                      color='green', linestyle=':', marker='^', markersize=4,
                      markevery=3, label='Best Prediction', linewidth=2)
        
        # Plot worst trajectory
        # Original
        axes[dim].plot(range(traj_steps),
                      all_original[max_loss_idx][:, 0, dim, 0].cpu(),
                      color='red', linestyle='-', marker='o', markersize=4,
                      markevery=5, label='Worst Original', linewidth=2)
        
        # Reconstruction
        axes[dim].plot(range(reconstruct_steps),
                      all_reconstructed[max_loss_idx][:, 0, dim, 0].cpu(),
                      color='orange', linestyle='--', marker='s', markersize=4,
                      markevery=5, label='Worst Reconstruction', linewidth=2)
        
        # Prediction
        axes[dim].plot(range(reconstruct_steps-1, traj_steps),
                      torch.cat([all_original[max_loss_idx][reconstruct_steps-1:reconstruct_steps],
                               all_predicted[max_loss_idx]], dim=0)[:, 0, dim, 0].cpu(),
                      color='red', linestyle=':', marker='^', markersize=4,
                      markevery=3, label='Worst Prediction', linewidth=2)
        
        # Add vertical line to separate reconstruction from prediction
        axes[dim].axvline(x=reconstruct_steps-1, color='gray', 
                         linestyle='--', alpha=0.5)
        
        # Add shaded regions to distinguish reconstruction and prediction areas
        axes[dim].axvspan(0, reconstruct_steps-1, alpha=0.1, color='blue', label='Reconstruction Region')
        axes[dim].axvspan(reconstruct_steps-1, traj_steps, alpha=0.1, color='purple', label='Prediction Region')
        
        axes[dim].set_title(f'Dimension {dim}')
        axes[dim].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add text annotations for reconstruction loss values
        axes[dim].text(0.02, 0.98, 
                      f'Best Reconstruction Loss: {reconstruction_losses[min_loss_idx]:.2e}\n'
                      f'Worst Reconstruction Loss: {reconstruction_losses[max_loss_idx]:.2e}',
                      transform=axes[dim].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Trajectory Analysis (Reconstruction: {reconstruct_steps} steps, Prediction: {prediction_steps} steps)', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{folder}/combined_trajectories.png', bbox_inches='tight', dpi=300)

    
    return min_loss_idx, max_loss_idx, reconstruction_losses, all_predicted
