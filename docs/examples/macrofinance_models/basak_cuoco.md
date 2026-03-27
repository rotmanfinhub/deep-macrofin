# Basak-Cuoco and Gârleanu-Panageas

The full solution for Basak-Cuoco model can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/basak_cuoco/basak_cuoco.ipynb" target="_blank">basak_cuoco.ipynb</a>. The solution for Gârleanu-Panageas model (with passive agent pricing) is at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/basak_cuoco/garleanu_panageas.ipynb" target="_blank">garleanu_panageas.ipynb</a>. 

The model has 3 agents and 2 shocks. Agent $u$ is unconstrained; agent $c$ is constrained; and agent $p$ is passive. Endowment follows a GBM with two independent shocks. Here we assume that the passive agent is not participating tin pricing.

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
| $\psi$ | IES | $\psi=1/\gamma$ |
| $\mu$ | Mean of the lognormal distribution of the asset return | $\mu=0.0183$ |
| $\sigma=(\sigma^1, \sigma^2)$ | Fundamental volatility | $\sigma^1=0.0357, \sigma^2=0$ |
| $\kappa$ | Death rate | $\kappa=0$ |
| $\omega=(\omega_u,\omega_c,\omega_p)$ | Mass of agents | $\omega_u=0.25, \omega_c=0.25, \omega_p=0.5$ |
| $\alpha_{p,min},\alpha_{p,max}$ | Max/Min alpha for passive agent | $\alpha_{p,min}=\alpha_{p,max}=0$ |

For passive pricing, we change the following parameters,

| Parameter | Definition | Value |
|-----------|------------|-------|
| $\gamma=(\gamma_u, \gamma_c, \gamma_p)$ | Risk aversion (unconstrained, constrained, passive) | $\gamma_u=1,\gamma_c=1,\gamma_p=1$ |
| $\psi$ | IES | $\psi=1.5$ |

## Variables

| Type | Definition |
|------|------------|
| State Variables | $(x_u, x_c, \alpha_p)$, $x_u,x_c\geq 0$, $x_u+x_c\leq 1$, $\alpha_p\in [\alpha_{p,min},\alpha_{p,max}]$ |
| Agents | $\xi = (\xi_u, \xi_c, \xi_p)$ |
| Endogenous Variables | $\alpha_u$, $\alpha_c$ |

## Equations

$$
\begin{align*}
    x_p &= 1 - x_u - x_c\\
    \alpha_{all} &= (\alpha_u, \alpha_c, \alpha_p)\\
    y &= x_u \xi_u + x_c \xi_c + x_p \xi_p\\
    \sigma_R &= \sigma - \sigma_y\\
    \sigma_{\alpha_p} &= \begin{bmatrix}
        \sigma_{\alpha_p}^1 \nu & \sigma_{\alpha_p}^1 \sqrt{1-\nu^2}\\
        \sigma_{\alpha_p}^1 \nu & \sigma_{\alpha_p}^1 \sqrt{1-\nu^2}
    \end{bmatrix}\\
    A &= \frac{\partial y}{\partial x_u} x_u (\alpha_u-1) + \frac{\partial y}{\partial x_c} x_c (\alpha_c-1)\\
    \sigma_y &= \frac{A\sigma + \frac{\partial y}{\partial \alpha_p} \sigma_{\alpha_p}}{y+A}\\
    \sigma_R &= \sigma - \sigma_y\\
    \sigma_x &= \begin{bmatrix}
       x_u (\alpha_u-1)\sigma_R & x_c(\alpha_c-1)\sigma_R & \sigma_{\alpha_p}
    \end{bmatrix}\\
    \sigma_\xi &= \frac{1}{\xi} \nabla \xi \sigma_x\\
    Var(\sigma) &= \frac{1-\frac{1}{\gamma}}{1-\psi} \frac{\sigma_\xi \cdot \sigma_R}{\|\sigma_R\|^2}\\
    q &= \gamma_u (\alpha_u + Var(\sigma)_u)\\
    \pi &= q \|\sigma_R\|^2\\
    \eta &= q \|\sigma_R\|\\
    \mu_x &= \begin{bmatrix}
        x_u(y-\xi_u+(1-\alpha_u)(1-q)\|\sigma_R\|^2) + \kappa (\omega_u-x_u)\\
        x_c(y-\xi_c+(1-\alpha_u)(1-q)\|\sigma_R\|^2) + \kappa (\omega_c-x_c)\\
        \theta(\bar{\alpha_p}-\alpha_p)\\
    \end{bmatrix}^T\\
    \Sigma &= \sigma_x \sigma_x^T\\
    \mu_\xi &= \frac{1}{\xi} \nabla_x \xi \mu_x + \frac{1}{2} \frac{1}{\xi} \text{Tr}(D^2\xi \Sigma)\\
    \mu_y &= \frac{1}{y} \nabla_x y \mu_x + \frac{1}{2} \frac{1}{y} \text{Tr}(D^2y \Sigma)\\
    \mu_P &= \mu - \mu_y + \sigma_y \cdot(\sigma_y -\sigma)\\
    r &= y + \mu_P - \pi\\
\end{align*}
$$

$$
HJB = \rho\psi + (1-\psi) \left(r+\eta \alpha_{all}\|\sigma_R\| - \frac{1}{2} \gamma(\alpha_{all}\|\sigma_R\|)^2 \right) + \mu_\xi + (1-\gamma) \sigma_\xi \sigma_R \alpha_{all} + \frac{1}{2} \frac{\psi-\gamma}{1-\psi} \sigma_\xi\cdot \sigma_\xi - \xi
$$

## Loss Functions
HJB residual (scaled to impose stronger penalty):

$$HJB/\rho = 0$$


Market clearning:

$$x_u\alpha_u+x_c\alpha_c+x_p\alpha_p=1$$

First order conditions:

$$\alpha_c = \min \left(\frac{q}{\gamma_c} - Var(\sigma)_c, \frac{\bar{\alpha_p}}{\|\sigma_R\|}\right)$$

If passive agent is pricing:

$$\alpha_p = \frac{q}{\gamma_p} - Var(\sigma)_p$$

Pricing:

$$\pi = \frac{(1+x_uVar(\sigma)_u + x_c Var(\sigma)_c)\|\sigma_R\|^2}{\frac{x_u}{\gamma_u} + \frac{x_c}{\gamma_c}}$$

If passive agent is pricing:

$$\pi = \frac{(1+x_uVar(\sigma)_u + x_c Var(\sigma)_c + x_p Var(\sigma)_p)\|\sigma_R\|^2}{\frac{x_u}{\gamma_u} + \frac{x_c}{\gamma_c} + \frac{x_p}{\gamma_p}}$$