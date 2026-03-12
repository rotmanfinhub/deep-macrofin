# Basak Cuoco Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/basak_cuoco/basak_cuoco.ipynb" target="_blank">basak_cuoco.ipynb</a>. 

The model has 3 agents and 2 shocks. Agent $u$ is unconstrained; agent $c$ is constrained; and agent $p$ is passive. Endowment follows a GBM with two independent shocks

$$
\frac{dY_t}{Y_t} = \mu dt + \sigma^1 dW_t^1 + \sigma^2 dW_t^2
$$

Risky asset price $P_t$ follows 

$$
\frac{dP_t}{P_t} = \mu_{P} dt + \sigma_P^1 dW_t^1 + \sigma_P^2 dW_t^2
$$

Returns are given by 

$$
\frac{dR_t}{R_t} = \frac{dP_t}{P_t} + \frac{Y_t}{P_t}dt = \left( \mu_P + \frac{Y_t}{P_t} \right)dt + \sigma_P^1 dW^1 + \sigma_P^2 dW^2
$$


## Parameters

| Parameter | Definition | Value |
|-----------|------------|-------|
| $n$ | Number of households | $n=3$ |
| $\gamma=(\gamma_u, \gamma_c, \gamma_p)$ | Risk aversion (unconstrained, constrained, passive) | $\gamma_u=1,\gamma_c=1,\gamma_p=1$ |
| $\rho$ | Discount rate | $\rho=0.05$ |
| $\bar{\alpha}$ | Leverage constrained ceiling | $\bar{\alpha}=1000.95$ |
| $\bar{\alpha_p}$ | Alpha passive agent | $\bar{\alpha_p}=0.0$ |
| $\theta$ | Alpha drift | $\theta=0.0$ |
| $\sigma_{\alpha_p}=(\sigma_{\alpha_p}^1, \sigma_{\alpha_p}^2)$ | Vol. of passive demand share | $\sigma_{\alpha_p}^1=0, \sigma_{\alpha_p}^2=0$ |
| $\nu$ | Correlation | $\nu=0.5$ |
| $\psi$ | IES | $\psi=1$ |
| $\mu$ | Mean of the lognormal distribution of the asset return | $\mu=0.22$ |
| $\sigma=(\sigma^1, \sigma^2)$ | Fundamental volatility | $\sigma^1=0.035, \sigma^2=0$ |
| $\kappa$ | Death rate | $\kappa=0$ |
| $\omega=(\omega_u,\omega_c,\omega_p)$ | Mass of agents | $\omega_u=0.25, \omega_c=0.25, \omega_p=0.5$ |
| $\alpha_{p,min},\alpha_{p,max}$ | Max/Min alpha for passive agent | $\alpha_{p,min}=\alpha_{p,max}=0$ |

## Variables

| Type | Definition |
|------|------------|
| State Variables | $(x_u, x_c, \alpha_p)$, $x_u,x_c\geq 0$, $x_u+x_c\leq 1$, $\alpha_p\in [\alpha_{p,min},\alpha_{p,max}]$ |
| Agents | $\xi = (\xi_u, \xi_c, \xi_p)$ |
| Endogenous Variables | $\pi$, $r$ (risk-free rate), $\mu_y$ (drift of endowment), $\sigma_y= (\sigma_y^1, \sigma_y^2)$ (volatility of endowment), $\alpha_c$ (alpha for constrained) |

## Equations

$$
\begin{align*}
    x_p &= 1 - x_u - x_c\\
    y &= x_u \xi_u + x_c \xi_c + x_p \xi_p\\
    \alpha_u &= \frac{1 - x_c \alpha_c - x_p \alpha_p}{x_u}\\
    \alpha_{state} & = (\alpha_u, \alpha_c)\\
    \alpha_{all} &= (\alpha_u, \alpha_c, \alpha_p)\\
    \sigma_R &= \sigma - \sigma_y\\
    \mu_p &= -\mu_{y,model} - \sigma_y\cdot \sigma_R\\
    \mu_P &= \mu_p + \mu + \sigma_R \cdot \sigma_R - \sigma_R \cdot \sigma\\
    \eta & = \frac{\pi}{\|\sigma_R\|}\\
    \sigma_{state} &= (\alpha_{state} - 1) \sigma_R^T\\
    \sigma_{\alpha_p} &= \begin{bmatrix}
        \sigma_{\alpha_p}^1 \nu & \sigma_{\alpha_p}^1 \sqrt{1-\nu^2}\\
        \sigma_{\alpha_p}^1 \nu & \sigma_{\alpha_p}^1 \sqrt{1-\nu^2}
    \end{bmatrix}\\
    \sigma_x & = \begin{bmatrix}
        \sigma_{state}\\
        \frac{\sigma_{\alpha_p}}{\alpha_p}
    \end{bmatrix}\\
    \mu_{\alpha_p} &= \frac{\theta (\bar{\alpha_p} - \alpha_p)}{\alpha_p}\\
    \sigma_{\xi} &= \frac{\partial \xi}{\partial x} \sigma_x \begin{bmatrix}
        x_u\\
        x_c\\
        \alpha_p
    \end{bmatrix}\\
    \hat{x} &= (x_u, x_c, x_p)\\
    \mu_{x,state} &= r + \eta \alpha_{all} \|\sigma_R\| - \xi - \mu_P + (1 - \alpha_{all}) \sigma_R\cdot \sigma_R + \kappa \frac{\omega-\hat{x}}{\hat{x}}\\
    \mu_{x} &= (\mu_{x,u}, \mu_{x,c}, \mu_{\alpha_p})\\
    \mu_{\xi_i,1} &= \frac{1}{\xi_i} \sum_j \frac{\partial\xi_i}{\partial x_j} \mu_{x,j}x_j\\
    \mu_{y_i,1} &= \frac{1}{y_i} \sum_j \frac{\partial y_i}{\partial x_j} \mu_{x,j}x_j\\
    \Sigma_x &= \sigma_x x\\
    a &= \Sigma_x \Sigma_x^T\\
    \mu_{\xi_i, 2} &= \frac{1}{2} \text{Tr}\left(\frac{D^2\xi_i}{\xi_i} a\right)\\
    \mu_{y_i, 2} &= \frac{1}{2} \text{Tr}\left(\frac{D^2y_i}{y_i} a\right)\\
    \mu_{\xi_i} &= \mu_{\xi_i,1}+\mu_{\xi_i,2}\\
    \mu_{y_i} &= \mu_{y_i,1} + \mu_{y_i,2}\\
    Var(\sigma) &= \frac{1-\gamma}{1-\psi} \frac{(\sigma_\xi^T \sigma_R)^T}{\sigma_R\cdot \sigma_R}\\
    HJB &= \rho\psi + (1-\psi) ( r + \eta \alpha_{all} \|\sigma_R\| - \frac{\gamma^T}{2} (\alpha_{all} \|\sigma_R\|)^2) + \mu_{\xi}\\
    &\quad + (1-\gamma^T) \sigma_\xi^T \sigma_R \alpha_{all} + \frac{\psi-\gamma^T}{1-\psi} \frac{1}{2} \sigma_{\xi}\cdot \sigma_{\xi} - \xi\\
    \Sigma_s &= \begin{bmatrix}
        \sigma^1, \sigma^2\\
        \sigma^1, \sigma^2\\
        \sigma_{\alpha_p}^1\nu, \sigma_{\alpha_p}^1\sqrt{1-\nu^2}\\
    \end{bmatrix}
\end{align*}
$$

When $\psi=1$:

$$
\begin{align*}
    Var(\sigma) &= 0\\
    HJB &= \rho + \mu_{\xi} - \xi + (1-\gamma^T) \sigma_\xi^T \sigma_R \alpha_{all}
\end{align*}
$$

## Loss Functions

$$
\begin{align*}
    HJB/\rho &= 0\\
    \alpha_u &= \frac{\pi}{\gamma_u \|\sigma_R\|^2} - Var(\sigma)_u\\
    \alpha_c &= \min \left\{\frac{\pi}{\gamma_c \|\sigma_R\|^2} - Var(\sigma)_c, \frac{\bar{\alpha}}{\|\sigma_R\|}\right\}\\
    \frac{\mu_P + y - r - \pi}{\rho} &= 0\\
    \frac{\mu_y - \mu_{y,model}}{y} &= 0\\
    \sigma_y &= \frac{\nabla y \cdot ((\alpha - 1, 1)\odot \sigma )}{y + \nabla y \cdot (\alpha - 1)}
\end{align*}
$$