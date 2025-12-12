# IMEX-FEM and PINNs for Non-Linear Reaction-Diffusion Equation

This repository contains the source code and results for a comparative study investigating the numerical solution of a non-linear reaction-diffusion partial differential equation (PDE) using two distinct approaches:
1. **Implicit-Explicit (IMEX) Finite Element Method (FEM)**
2. **Physics-Informed Neural Networks (PINNs)**

The target PDE is a general reaction-diffusion model with a cubic reaction term, relevant to phenomena like wave propagation in biological tissues (e.g., cardiac tissue, nerve impulses) or phase transitions.

$$
\frac{\partial u}{\partial t} = \Sigma \Delta u - f(u)
$$

where $f(u) = a(u - f_r)(u - f_t)(u - f_d)$.

## 👥 Team Members
* **Marta Celio**
* **Sai Vijaya Bhargavi Akavarapu**
* **Bhanu Prakash Maruboyina**

## Project Overview

The project aims to:
1. Implement a stable and accurate IMEX-FEM scheme for the time-dependent PDE.
2. Formulate and train a PINN model to solve the same PDE.
3. Compare the solutions, activation times, and stability properties of both methods.

**Code Structure:**
* `fem1.m`: Main script for the IMEX-FEM solution and analysis.
* `assembleDiffusion.m`: Function to assemble the Diffusion (stiffness) matrix $A(\Sigma)$.
* `assembleReaction.m`: Function to assemble the Mass matrix $M$.
* `pinns2.ipynb`: Jupyter notebook containing the PINN implementation and training.
* `solution_FEM.avi` / `solution_PINNs.mp4`: Video outputs visualizing the time evolution of the solutions.

## Part One: IMEX Finite Element Method (FEM)

### 1. IMEX Integration Scheme

The time derivative is discretized using a forward difference, and the right-hand side is split: the diffusion term ($\Sigma \Delta u$) is treated **implicitly** (at time $t_{k+1}$), and the non-linear reaction term (-f(u)) is treated **explicitly** (at time $t_k$).


### 2. Weak Formulation and Discretization

The weak form of the problem:

Using a finite-dimensional basis $\{\varphi_i\}$ (piecewise linear on triangular elements), the problem is reduced to a linear system of equations:

$$
(\delta t A(\Sigma) + M) \hat{u}^{k+1} = M (\hat{u}^k - \delta t f(\hat{u}^k))
$$


### 3. Implementation Details

* **Matrix Assembly:** `assembleDiffusion.m` and `assembleReaction.m` implement the elemental contributions to $A(\Sigma)$ and $M$. The diffusion coefficient $\Sigma$ is implemented as an element-wise array $S$, allowing for spatially varying diffusivity ($\Sigma_h$ for healthy, $\Sigma_d$ for diseased domains).
* **Linear System Solver:** The main loop in `fem1.m` sets up the initial conditions, calculates the right-hand side using the solution from the previous timestep, and solves the resulting linear system for $\hat{u}^{k+1}$.
* **Stability and M-Matrix Check:** Additional code verifies if the left-hand side matrix $(\delta t A(\Sigma) + M)$ satisfies the M-matrix condition (non-positive off-diagonals and positive diagonals in its LU decomposition).

### 4. Key Results (FEM)

The activation time (time until the potential exceeds a threshold) and stability checks were performed for various timestep ($\delta t$) and mesh configurations, and for different ratios of diseased ($\Sigma_d$) to healthy ($\Sigma_h$) diffusivity.

| $\delta t$ | Mesh | $\Sigma_d = 10 \Sigma_h$ Activation Time | is M-matrix? | Potential Exceeds? |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | 128 | 25.2 ms | No | Yes |
| 0.025 | 256 | 22.6 ms | No | Barely |

| $\delta t$ | Mesh | $\Sigma_d = 0.1 \Sigma_h$ Activation Time | is M-matrix? | Potential Exceeds? |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | 128 | 34.4 ms | No | Yes |
| 0.025 | 256 | 31.5 ms | No | Yes |

* **Observation:** Activation time increases as the diffusivity $\Sigma_d$ decreases relative to $\Sigma_h$.
* **Stability:** In all tested cases, the left-hand side matrix was **not** a strict M-matrix. The potential *u* consistently exceeded the physical interval $[0, 1]$, although finer meshes or smaller $\delta t$ limited the magnitude of the overshoot ("Barely").
* **Mass Lumping:** Changing the Mass Matrix $M$ to its diagonal-lumped counterpart did not guarantee the M-matrix property and did not resolve the potential overshoot issue.

## Part Two: Physics-Informed Neural Networks (PINNs)

### 1. PINN Formulation

The PINN approach seeks an approximate solution $u_{NN}(x, y, t)$ learned by a neural network. The training is driven by minimizing a loss function based on the PDE residual.

The residual $\mathcal{R}$ is defined by bringing all terms of the PDE to one side:

$$
\mathcal{R}(x, y, t) = - \frac{\partial u_{NN}}{\partial t} + \Sigma \Delta u_{NN} - f(u_{NN}) = 0
$$

The total loss is typically a weighted sum of the PDE residual loss, Boundary Condition (BC) loss, and Initial Condition (IC) loss.


### 2. Implementation Details

* **Residual Calculation:** The derivatives ($\frac{\partial u_{NN}}{\partial t}$ and $\Delta u_{NN}$) are calculated using automatic differentiation (enabled by setting $u_{NN}$ to require a gradient). The Laplacian $\Delta u_{NN}$ is the sum of second-order spatial derivatives ($\frac{\partial^2 u_{NN}}{\partial x^2} + \frac{\partial^2 u_{NN}}{\partial y^2}$).
* **Collocation Points:** Training data (collocation points) are sampled randomly over the space-time domain $\Omega \times [0, T_f]$. A higher density of points is sampled within the non-healthy (diseased) domains (circles).
* **Initial Conditions (Hard Constraint):** The initial condition $u(x, y, 0) = \chi_{\{x \leq 0.1 \land y \geq 0.9\}}$ (a characteristic function defining a localized initial 'spike' or activation) is imposed as a **hard constraint**
* **Training:** A standard **Adam optimizer** with a learning rate of $0.01$ was used.

### 3. Key Results (PINNs)

The resulting solution from the PINN approach (visualized in `solution_PINNs.mp4`) shows a qualitative difference from the expected physical behavior and the FEM solution (`solution_FEM.avi`).

* **Observation:** The PINN solution tends to **monotonically increase** over time, suggesting the model failed to properly learn the complex balance between diffusion and the non-linear reaction term that governs wave propagation/decay in this system.

## Conclusion and Future Work

The **IMEX-FEM** approach successfully implemented the scheme and provided physically plausible solutions consistent with the expected reaction-diffusion behavior, though it demonstrated numerical stability challenges (potential overshoot, non-M-matrix property) especially for larger timesteps.

The **PINN** approach, despite careful formulation and hard-constrained initial conditions, failed to converge to a physically meaningful solution under the tested configuration and training regimen. This suggests that the complex non-linearity of $f(u)$ and the stiffness of the PDE pose a significant challenge for a standard PINN setup.
