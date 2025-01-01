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
    return reconstructed_data

def plot_trajectory(model, test_data, device,folder,t = 1, traj_steps = 50):

    test_data = test_data[(t-1)*traj_steps:t*traj_steps]
    reconstructed_data = recontruction(model, test_data, device)
    reconstructed_data = torch.stack(reconstructed_data).squeeze(2)
    
    M = test_data.shape[2]  # number of dimensions
    fig, axes = plt.subplots(M, 1, figsize=(10, 3*M))

    print(reconstructed_data.shape)
    
    for dim in range(M):
        axes[dim].plot(test_data[:, 0, dim, 0].cpu(), 'b-', label='Original', alpha=0.5)
        axes[dim].plot(reconstructed_data[:, 0, dim, 0].cpu(), 'r--', label='Reconstructed', alpha=0.5)
        axes[dim].set_title(f'Dimension {dim}')
        axes[dim].legend()
    
    plt.tight_layout()
    plt.savefig(f'{folder}/trajectory.png')

#2 plot a swarm plot of the error of reconstruction for sample data points
def swarm_plot(model, test_data, device,folder,t=100):
    """
    Create swarm plots showing reconstruction error distribution across dimensions
    """
    # Get reconstructed data
    test_data = test_data[:t]
    reconstructed_data = recontruction(model, test_data, device)
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
        
        sns.swarmplot(data=df_dist, x='type', y='value', ax=axes[0, dim])
        axes[0, dim].set_title(f'Dimension {dim} - Distribution')
        
        # Error Distribution
        error = orig_vals - recon_vals
        df_error = pd.DataFrame({'error': error})
        
        sns.swarmplot(data=df_error, y='error', ax=axes[1, dim])
        axes[1, dim].set_title(f'Dimension {dim} - Reconstruction Error')
        
    plt.tight_layout()
    plt.savefig(f'{folder}/swarm_plot.png')