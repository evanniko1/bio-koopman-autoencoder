# koopkan: Koopman Autoencoder X Kolmogorov-Arnold networks X Attention Free Transformer X Dynamic Reencoding

A new class of physics-based methods related to Koopman theory has been introduced, offering an alternative for processing nonlinear dynamics systems. Koopman theory is based on the insight that a nonlinear dynamical system can be fully encoded using an operator that describes how scalar functions propagate in time. The Koopman operator is *linear*, and thus preferable to work with in practice, as tools from linear algebra can be directly applied. The Koopman operator maps between function spaces and thus it is infinite-dimensional and can not be represented on a computer. However, most machine learning approaches hypothesize that there exists a data transformation under which an approximate finite-dimensional Koopman operator is available. Typically, this map is represented via an autoencoder network, embedding the input onto a low-dimensional latent space.

Concurrently, inspired by the Kolmogorov-Arnold representation theorem, Kolmogorov-Arnold Networks (KANs) have been proposed as a promising alternative to Multi-Layer Perceptrons (MLPs). While MLps have *fixed* activation functions on *nodes* ("neurons"), KANs have *learnable* activation functions on *edges* ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parameterized as a spline. Here, we bring together these two worlds by using KANs as the backbone for the Koopman autoencoder. 

Newly Introduced Methods

1. ### Attention-Free Transformer:
    This optimized attention mechanism is applied after processing the current time step in the encoder. It integrates the latest prediction used for the Koopman operator by incorporating a problem-specific time window. The most recent point is updated through attention from surrounding points, enabling the model to correct inaccurate predictions and improve long-term forecasting accuracy.

2. ### Dynamic Reencoding:
    Inspired by periodic reencoding, which ensures that latent space predictions remain within the Koopman invariant subspace and handles challenges like switching dynamics and multiple fixed points, Dynamic Reencoding automates this process. It does so by comparing the reencoding difference of the current point against the average difference over a previous window of predicted points, allowing the model to determine when reencoding is necessary.

## Setup Instructions
```bash
   git clone <repository-url>
   cd <repository-name>
   pip install -r requirements.txt
```

## Usage
### Training a Model
Run the training script with default parameters:
```bash
    python run_experiments.py --dataset discrete_spectrum --folder test
```

## Hyperparameter Optimization
Optimize the model using Optuna:
```bash
    python KoopmanAE_optimization.py
```

## Inference
Use the demo notebook for inference on trained models:
```bash
    jupyter notebook koopman_inference_demo.ipynb
```

# Project To-Do List  

## Tasks   
- [ ] Implement decoder architecture selection (Linear/Non-Linear)
- [ ] Complete comparative experiments across dynamical systems:
  - [ ] AFT vs standard approach (no AFT)
  - [ ] Linear vs Non-Linear Decoder performance
  - [ ] Inference model comparison (Dynamic Reencoding, No Reencoding, Periodic Reencoding, Per-timestep Reencoding)
- [ ] Develop sequential training policy
- [ ] Create system-specific optimization scripts for different dynamical systems