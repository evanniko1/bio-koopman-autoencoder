# koopkan: Koopman Autoencoder X Kolmogorov-Arnold networks

A new class of physics-based methods related to Koopman theory has been introduced, offering an alternative for processing nonlinear dynamics systems. Koopman theory is based on the insight that a nonlinear dynamical system can be fully encoded using an operator that describes how scalar functions propagate in time. The Koopman operator is *linear*, and thus preferable to work with in practice, as tools from linear algebra can be directly applied. The Koopman operator maps between function spaces and thus it is infinite-dimensional and can not be represented on a computer. However, most machine learning approaches hypothesize that there exists a data transformation under which an approximate finite-dimensional Koopman operator is available. Typically, this map is represented via an autoencoder network, embedding the input onto a low-dimensional latent space.

Concurrently, inspired by the Kolmogorov-Arnold representation theorem, Kolmogorov-Arnold Networks (KANs) have been proposed as a promising alternative to Multi-Layer Perceptrons (MLPs). While MLps have *fixed* activation functions on *nodes* ("neurons"), KANs have *learnable* activation functions on *edges* ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parameterized as a spline. Here, we bring together these two worlds by using KANs as the backbone for the Koopman autoencoder. 

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

- [ ] **Plot multiple trajectories** → Implement in `Viz.py` , where you define multiple trajectories and to specify the number of reconstruction timesteps and the number of prediction timesteps
- [ ] **Test network** → Analyze `testing result`  
- [ ] **Sequential training policy**  
- [ ] **Autoencoder optimization script**  
- [ ] **Pendulum** write different implementation of the pendulum problem.
- [ ] **Regularization discussion** → Should we use it?  