import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import os

#**************** To do ****************
# 1. Plot multiple trajectories
#*************************************

def plot_eigenvalues_on_unit_circle(eigenvalues, folder_path, save=True):
    """
    Plot eigenvalues on the unit circle.
    
    Args:
        eigenvalues: Array of eigenvalues
        folder_path: Directory path to save the figure
    """
    plt.figure(figsize=(8, 8))
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle)

    # Plot eigenvalues
    plt.scatter(eigenvalues.real, eigenvalues.imag, c='blue', alpha=0.8)

    # Set equal aspect ratio
    plt.axis('equal')

    # Set limits slightly larger than the unit circle
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    plt.title('Eigenvalues on the Unit Circle')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Save the figure
    if save:
        plt.savefig(os.path.join(folder_path, 'eigenvalues_unit_circle.png'))
    else:
        plt.show()


def violin_plot(model, test_data, device, folder, save=True):
    """
    Create violin plots showing reconstruction error distribution across dimensions
    """
    model.eval()

    with torch.no_grad():
        X = test_data.to(device)
        Y, reconstructed_X = model.ae(X)
    
    M = X.shape[2]  # Number of dimensions
    fig, axes = plt.subplots(2, M, figsize=(5 * M, 8))
    
    for dim in range(M):
        # Original vs Reconstructed Distribution
        orig_vals = X[:, :, dim].cpu().detach().numpy().flatten()  # Flatten to 1D
        recon_vals = reconstructed_X[:, :, dim].cpu().detach().numpy().flatten()  # Flatten to 1D

        df_dist = pd.DataFrame({
            'value': np.concatenate([orig_vals, recon_vals]),
            'type': ['Original'] * len(orig_vals) + ['Reconstructed'] * len(recon_vals)
        })
        sns.violinplot(data=df_dist, x='type', y='value', ax=axes[0, dim])
        axes[0, dim].set_title(f'Dimension {dim} - Distribution')
        
        # Error Distribution
        error = orig_vals - recon_vals
        df_error = pd.DataFrame({'error': error})
        
        sns.violinplot(data=df_error, y='error', ax=axes[1, dim])
        axes[1, dim].set_title(f'Dimension {dim} - Reconstruction Error')

    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/violin_plot.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()



def plot_trajectories(ref_trajectory, pred_trajectory, recon_trajectory, folder_path='', save=True):
    """Plots reference, predicted, and reconstructed trajectories up to 3 dimensions."""
    ref_trajectory = np.asarray(ref_trajectory)
    pred_trajectory = np.asarray(pred_trajectory)
    recon_trajectory = np.asarray(recon_trajectory)
    
    if pred_trajectory.shape != ref_trajectory.shape or recon_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"All trajectories must have shape {ref_trajectory.shape}")

    dimensions = ref_trajectory.shape[1]
    if dimensions > 3:
        raise ValueError("Function supports up to 3 dimensions only")

    plt.figure(figsize=(10, 8))
    
    if dimensions == 1:
        plt.plot(ref_trajectory[:, 0], label='Reference', color='blue', linewidth=2, marker='o')
        plt.plot(pred_trajectory[:, 0], label='Predicted', color='red', linewidth=2, linestyle='--', marker='x')
        plt.plot(recon_trajectory[:, 0], label='Reconstructed', color='green', linewidth=2, linestyle='-.', marker='s')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
    elif dimensions == 2:
        plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], label='Reference', 
                 color='blue', linewidth=2, marker='o')
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], label='Predicted', 
                 color='red', linewidth=2, linestyle='--', marker='x')
        plt.plot(recon_trajectory[:, 0], recon_trajectory[:, 1], label='Reconstructed', 
                 color='green', linewidth=2, linestyle='-.', marker='s')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        
    elif dimensions == 3:
        ax = plt.axes(projection='3d')
        ax.plot3D(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
                  label='Reference', color='blue', linewidth=2, marker='o')
        ax.plot3D(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
                  label='Predicted', color='red', linewidth=2, linestyle='--', marker='x')
        ax.plot3D(recon_trajectory[:, 0], recon_trajectory[:, 1], recon_trajectory[:, 2], 
                  label='Reconstructed', color='green', linewidth=2, linestyle='-.', marker='s')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

    plt.title('Trajectory Comparison', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(folder_path, 'trajectories.png') if folder_path else 'trajectories.png'
    if save:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_trajectory_comparison(ref_trajectory, pred_trajectory, recon_trajectory, folder_path='', save=True):
    """
    Plots the reference, predicted, and reconstructed trajectories for each dimension over time.

    Parameters:
    ref_trajectory (numpy.ndarray): The reference trajectory, shape (length, dimensions).
    pred_trajectory (numpy.ndarray): The predicted trajectory, shape (length, dimensions).
    recon_trajectory (numpy.ndarray): The reconstructed trajectory, shape (length, dimensions).
    folder_path (str): Path to save the plot (optional).
    """
    # Ensure trajectories are numpy arrays
    ref_trajectory = np.asarray(ref_trajectory)
    pred_trajectory = np.asarray(pred_trajectory)
    recon_trajectory = np.asarray(recon_trajectory)
    
    # Get length and dimensions from reference trajectory
    length = ref_trajectory.shape[0]
    dimensions = ref_trajectory.shape[1]

    # Check if all trajectories have the same shape as the reference
    if pred_trajectory.shape != ref_trajectory.shape or recon_trajectory.shape != ref_trajectory.shape:
        raise ValueError(f"All trajectories must have shape {ref_trajectory.shape}")

    # Time array based on length
    time = np.linspace(0, 1, length)  # Length time steps from 0 to 1

    # Create subplots: one for each dimension
    fig, axes = plt.subplots(dimensions, 1, figsize=(10, 4 * dimensions), sharex=True)

    # Handle case of single dimension (axes won't be a list)
    if dimensions == 1:
        axes = [axes]

    # Plot each dimension
    for dim in range(dimensions):
        axes[dim].plot(time, ref_trajectory[:, dim], label=f'Reference (Dim {dim})', 
                       color='blue', linewidth=2)
        axes[dim].plot(time, pred_trajectory[:, dim], label=f'Predicted (Dim {dim})', 
                       color='red', linestyle='--', linewidth=2)
        axes[dim].plot(time, recon_trajectory[:, dim], label=f'Reconstructed (Dim {dim})', 
                       color='green', linestyle='-.', linewidth=2)
        axes[dim].set_title(f'Dimension {dim} Over Time', fontsize=14)
        axes[dim].set_ylabel('Value', fontsize=12)
        axes[dim].legend()
        axes[dim].grid(True)

    # Set x-label only on the last subplot
    axes[-1].set_xlabel('Time', fontsize=12)

    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(folder_path, 'trajectory_comparison.png') if folder_path else 'trajectory_comparison.png'
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    