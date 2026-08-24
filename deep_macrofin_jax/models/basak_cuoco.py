"""JAX port of the three-agent Basak-Cuoco Deep-MacroFin example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..networks import MLP, MLPConfig, directional_second_derivative

Array = jax.Array


@dataclass(frozen=True)
class BasakCuocoConfig:
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rho: float = 0.05
    alpha_bar: float = 1000.95
    alpha_p_bar: float = 0.0
    theta: float = 0.0
    sigma_alpha_p: tuple[float, float] = (0.0, 0.0)
    nu: float = 0.5
    mu: float = 0.0183
    sigma: tuple[float, float] = (0.0357, 0.0)
    kappa: float = 0.0
    omega: tuple[float, float, float] = (0.25, 0.25, 0.50)
    alpha_p_min: float = 0.0
    alpha_p_max: float = 0.0
    hidden_units: tuple[int, ...] = (30, 30, 30, 30)
    activation: str = "tanh"
    min_share: float = 0.05
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.alpha_p_max < self.alpha_p_min:
            raise ValueError("alpha_p_max must be at least alpha_p_min")
        if self.min_share < 0 or 3 * self.min_share >= 1:
            raise ValueError("min_share must leave positive mass for all agents")
        if any(value <= 0 for value in self.gamma):
            raise ValueError("risk aversion must be positive")


class BasakCuocoModel:
    """Three-state, five-policy Basak-Cuoco neural residual model."""

    name = "basak_cuoco"

    def __init__(self, config: BasakCuocoConfig = BasakCuocoConfig()) -> None:
        self.config = config
        self.gamma = jnp.asarray(config.gamma)
        self.psi = 1.0 / self.gamma
        self.sigma = jnp.asarray(config.sigma)
        self.omega = jnp.asarray(config.omega)
        self.network = MLP(
            input_dim=3,
            output_dim=5,
            config=MLPConfig(
                hidden_units=config.hidden_units,
                activation=config.activation,
                positive_outputs=(0, 1, 2),
                output_bias=(-2.970628, -2.970628, -2.970628, 1.0, 1.0),
            ),
            lower_bounds=(config.min_share, config.min_share, config.alpha_p_min),
            upper_bounds=(
                1.0 - 2.0 * config.min_share,
                1.0 - 2.0 * config.min_share,
                config.alpha_p_max,
            ),
        )

    def init(self, key: Array) -> Any:
        return self.network.init(key)

    def sample(self, key: Array, batch_size: int) -> Array:
        share_key, alpha_key = jax.random.split(key)
        raw = jax.random.dirichlet(share_key, jnp.ones((3,)), shape=(batch_size,))
        shares = self.config.min_share + (1.0 - 3.0 * self.config.min_share) * raw
        if self.config.alpha_p_max == self.config.alpha_p_min:
            alpha_p = jnp.full((batch_size, 1), self.config.alpha_p_min)
        else:
            alpha_p = jax.random.uniform(
                alpha_key,
                (batch_size, 1),
                minval=self.config.alpha_p_min,
                maxval=self.config.alpha_p_max,
            )
        return jnp.concatenate((shares[:, :2], alpha_p), axis=1)

    def predict(self, params: Any, states: Array) -> dict[str, Array]:
        values = self.network.apply(params, states)
        return {
            "xi": values[..., :3],
            "alpha_u": values[..., 3:4],
            "alpha_c": values[..., 4:5],
        }

    def _point_quantities(self, params: Any, state: Array) -> dict[str, Array]:
        cfg = self.config
        eps = cfg.epsilon

        def unknowns(point: Array) -> Array:
            return self.network.apply(params, point)

        def xi_fn(point: Array) -> Array:
            return unknowns(point)[:3]

        def y_fn(point: Array) -> Array:
            values = unknowns(point)
            xu, xc = point[:2]
            shares = jnp.asarray([xu, xc, 1.0 - xu - xc])
            return jnp.dot(shares, values[:3])

        values = unknowns(state)
        xi = values[:3]
        alpha_u, alpha_c = values[3], values[4]
        xu, xc, alpha_p = state
        xp = 1.0 - xu - xc
        y = y_fn(state)
        xi_jac = jax.jacrev(xi_fn)(state)
        y_grad = jax.grad(y_fn)(state)

        sigma_alpha_1 = cfg.sigma_alpha_p[0]
        sigma_alpha = jnp.asarray(
            [sigma_alpha_1 * cfg.nu, sigma_alpha_1 * jnp.sqrt(1.0 - cfg.nu**2)]
        )
        a_term = (
            y_grad[0] * xu * (alpha_u - 1.0)
            + y_grad[1] * xc * (alpha_c - 1.0)
        )
        sigma_y = (a_term * self.sigma + y_grad[2] * sigma_alpha) / (
            y + a_term + eps
        )
        sigma_r = self.sigma - sigma_y
        sigma_r_sq = jnp.dot(sigma_r, sigma_r)
        sigma_r_norm = jnp.sqrt(sigma_r_sq + eps)
        sigma_x = jnp.stack(
            (
                xu * (alpha_u - 1.0) * sigma_r,
                xc * (alpha_c - 1.0) * sigma_r,
                sigma_alpha,
            ),
            axis=0,
        )
        sigma_xi = (xi_jac / (xi[:, None] + eps)) @ sigma_x
        coefficient = (1.0 - 1.0 / self.gamma) / (1.0 - self.psi + eps)
        varsigma = coefficient * (sigma_xi @ sigma_r) / (sigma_r_sq + eps)

        q = self.gamma[0] * (alpha_u + varsigma[0])
        pi = q * sigma_r_sq
        eta = q * sigma_r_norm
        alpha_all = jnp.asarray([alpha_u, alpha_c, alpha_p])
        mu_x = jnp.asarray(
            [
                xu * (y - xi[0] + (1.0 - alpha_u) * (1.0 - q) * sigma_r_sq)
                + cfg.kappa * (self.omega[0] - xu),
                xc * (y - xi[1] + (1.0 - alpha_c) * (1.0 - q) * sigma_r_sq)
                + cfg.kappa * (self.omega[1] - xc),
                cfg.theta * (cfg.alpha_p_bar - alpha_p),
            ]
        )

        xi_hessian_contraction = jnp.zeros_like(xi)
        y_hessian_contraction = jnp.asarray(0.0)
        for shock in range(sigma_x.shape[1]):
            direction = sigma_x[:, shock]
            xi_hessian_contraction = xi_hessian_contraction + directional_second_derivative(
                xi_fn, state, direction
            )
            y_hessian_contraction = y_hessian_contraction + directional_second_derivative(
                y_fn, state, direction
            )

        mu_xi = (xi_jac @ mu_x + 0.5 * xi_hessian_contraction) / (xi + eps)
        mu_y = (jnp.dot(y_grad, mu_x) + 0.5 * y_hessian_contraction) / (y + eps)
        mu_price = cfg.mu - mu_y + jnp.dot(sigma_y, sigma_y - self.sigma)
        interest_rate = y + mu_price - pi

        hjb = (
            cfg.rho * self.psi
            + (1.0 - self.psi)
            * (
                interest_rate
                + eta * alpha_all * sigma_r_norm
                - 0.5 * self.gamma * jnp.square(alpha_all * sigma_r_norm)
            )
            + mu_xi
            + (1.0 - self.gamma) * (sigma_xi @ sigma_r) * alpha_all
            + (self.psi - self.gamma)
            / (1.0 - self.psi + eps)
            * jnp.sum(jnp.square(sigma_xi), axis=1)
            / 2.0
            - xi
        ) / cfg.rho

        market_clearing = xu * alpha_u + xc * alpha_c + xp * alpha_p - 1.0
        alpha_c_foc = alpha_c - jnp.minimum(
            q / self.gamma[1] - varsigma[1], cfg.alpha_bar / (sigma_r_norm + eps)
        )
        pricing = (
            (1.0 + xu * varsigma[0] + xc * varsigma[1])
            * sigma_r_sq
            / (xu / self.gamma[0] + xc / self.gamma[1] + eps)
        )
        pricing_residual = pi - pricing

        return {
            "xi": xi,
            "alpha_u": alpha_u,
            "alpha_c": alpha_c,
            "y": y,
            "q": q,
            "pi": pi,
            "eta": eta,
            "sigma_r": sigma_r,
            "sigma_r_norm": sigma_r_norm,
            "mu_x": mu_x,
            "mu_price": mu_price,
            "interest_rate": interest_rate,
            "varsigma": varsigma,
            "hjb": hjb,
            "market_clearing": market_clearing,
            "alpha_c_foc": alpha_c_foc,
            "pricing": pricing_residual,
        }

    def evaluate(self, params: Any, states: Array) -> dict[str, Array]:
        return jax.vmap(lambda state: self._point_quantities(params, state))(states)

    def loss_terms(self, params: Any, states: Array) -> dict[str, Array]:
        values = self.evaluate(params, states)
        return {
            "hjb": values["hjb"],
            "market_clearing": values["market_clearing"],
            "alpha_c_foc": values["alpha_c_foc"],
            "pricing": values["pricing"],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": 3,
            "outputs": ["xi_u", "xi_c", "xi_p", "alpha_u", "alpha_c"],
            "gamma": list(self.config.gamma),
            "rho": self.config.rho,
            "network": self.network.metadata(),
            "second_order_method": "directional_jvp",
        }
