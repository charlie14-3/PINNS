# PINNS
# Physics-Informed Neural Networks (PINNs) for Hemodynamic Modeling

An AI-driven computational fluid dynamics (CFD) pipeline that leverages Physics-Informed Neural Networks (PINNs) to simulate and analyze complex blood flow profiles across a 2D arterial stenosis without relying on traditional mesh generation or labeled simulation data.

## Project Overview
Traditional CFD solvers (e.g., ANSYS Fluent) are computationally expensive and require intensive mesh generation to solve fluid dynamics equations. This project bypasses legacy pipelines by training a deep feedforward neural network that maps spatial coordinates $(x, y)$ directly to velocity fields $(u, v)$ and pressure fields $(p)$. By embedding the governing partial differential equations (PDEs) directly into the neural network's loss function, the model enforces physical laws algorithmically.

## Core Features
* **Physics-Embedded Loss Function:** Integrates the incompressible Navier-Stokes and continuity equations directly into the training loop via automatic differentiation.
* **Meshcomputational Mesh-Free Inference:** Simulates hemodynamics across a 50% arterial blockage using a continuous coordinate-based neural representation.
* **Clinical Metric Extraction:** Automatically computes diagnostic indicators including maximum pressure drops, peak velocity vectors, and localized Wall Shear Stress (WSS) using NumPy.
* **Interactive Visualization:** Features a web-based real-time diagnostic dashboard to render high-fidelity fluid velocity vectors and pressure field heatmaps.

## Technical Architecture & Math
The network minimizes a multi-objective loss function:
$$\mathcal{L} = \mathcal{L}_{\text{boundary}} + \lambda_{1}\mathcal{L}_{\text{continuity}} + \lambda_{2}\mathcal{L}_{\text{momentum}}$$

Where the Navier-Stokes residual forces mass and momentum conservation across the spatial domain:
$$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u}$$

## Tech Stack
* **Language:** Python, JavaScript
* **Libraries:** DeepXDE, SciPy, NumPy, Matplotlib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
