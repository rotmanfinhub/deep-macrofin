"""Scalable stationary Lucas-tree model from the Deep-MacroFin examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..deepset import DeepSet, DeepSetConfig
from ..networks import MLP, MLPConfig, directional_second_derivative

Array = jax.Array


@dataclass(frozen=True)
class TreeConfig:
    n_trees: int = 2
    gamma: float = 5.0
    rho: float = 0.05
    mu_ys: tuple[float, ...] | None = None
    sigma_ys: tuple[float, ...] | None = None
    hidden_units: tuple[int, ...] = (80, 80, 80, 80)
    architecture: str = "deepset"
    deepset_phi_units: tuple[int, ...] = (80, 80)
    deepset_rho_units: tuple[int, ...] = (80, 80)
    activation: str = "tanh"
    min_share: float = 0.01
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if self.n_trees < 2:
            raise ValueError("n_trees must be at least two")
        if self.min_share < 0 or self.n_trees * self.min_share >= 1:
            raise ValueError("min_share must leave positive mass for every tree")
        if self.mu_ys is not None and len(self.mu_ys) != self.n_trees:
            raise ValueError("mu_ys must contain one drift per tree")
        if self.sigma_ys is not None and len(self.sigma_ys) != self.n_trees:
            raise ValueError("sigma_ys must contain one volatility per tree")
        if self.architecture not in {"deepset", "mlp"}:
            raise ValueError("architecture must be 'deepset' or 'mlp'")


class TreeModel:
    """A vector-output neural solution for an arbitrary number of trees.

    The model uses directional Hessian products for the one-factor diffusion.
    Consequently, second-order differentiation does not allocate a full Hessian
    for every output and observation.
    """

    name = "tree"

    def __init__(self, config: TreeConfig = TreeConfig()) -> None:
        self.config = config
        self.n_trees = config.n_trees
        self.state_dim = config.n_trees - 1
        self.mu_ys = jnp.asarray(
            config.mu_ys
            if config.mu_ys is not None
            else tuple(0.01 * (i + 1) for i in range(config.n_trees))
        )
        self.sigma_ys = jnp.asarray(
            config.sigma_ys
            if config.sigma_ys is not None
            else tuple(0.01 * (i + 1) for i in range(config.n_trees))
        )
        upper = 1.0 - (config.n_trees - 1) * config.min_share
        if config.architecture == "deepset":
            self.network = DeepSet(
                input_size_sym=self.state_dim,
                output_size=self.n_trees,
                lower_bound=config.min_share,
                upper_bound=upper,
                config=DeepSetConfig(
                    phi_units=config.deepset_phi_units,
                    rho_units=config.deepset_rho_units,
                    activation=config.activation,
                    positive=True,
                ),
            )
        else:
            self.network = MLP(
                self.state_dim,
                self.n_trees,
                MLPConfig(
                    hidden_units=config.hidden_units,
                    activation=config.activation,
                    positive_outputs=tuple(range(config.n_trees)),
                    output_bias=(0.5413248546,) * config.n_trees,
                ),
                lower_bounds=(config.min_share,) * self.state_dim,
                upper_bounds=(upper,) * self.state_dim,
            )

    def init(self, key: Array) -> Any:
        return self.network.init(key)

    def sample(self, key: Array, batch_size: int) -> Array:
        raw_shares = jax.random.dirichlet(
            key, jnp.ones((self.n_trees,)), shape=(batch_size,)
        )
        shares = self.config.min_share + (
            1.0 - self.n_trees * self.config.min_share
        ) * raw_shares
        return shares[:, :-1]

    def predict(self, params: Any, states: Array) -> Array:
        return self.network.apply(params, states)

    def _point_quantities(self, params: Any, state: Array) -> dict[str, Array]:
        eps = self.config.epsilon

        def all_shares(point: Array) -> Array:
            return jnp.concatenate((point, jnp.asarray([1.0 - jnp.sum(point)])))

        def kappa_fn(point: Array) -> Array:
            return self.network.apply(params, point)

        def q_fn(point: Array) -> Array:
            return all_shares(point) / jnp.maximum(kappa_fn(point), eps)

        z_all = all_shares(state)
        weighted_mu = jnp.dot(self.mu_ys, z_all)
        weighted_sigma = jnp.dot(self.sigma_ys, z_all)
        mu_z_geo = (
            self.mu_ys[:-1]
            - weighted_mu
            + weighted_sigma * (weighted_sigma - self.sigma_ys[:-1])
        )
        sigma_z_geo = self.sigma_ys[:-1] - weighted_sigma
        mu_z_ari = mu_z_geo * state
        sigma_z_ari = sigma_z_geo * state

        mu_last_ari = -jnp.sum(mu_z_ari)
        sigma_last_ari = -jnp.sum(sigma_z_ari)
        last_share = jnp.maximum(z_all[-1], eps)
        mu_z_geo_all = jnp.concatenate((mu_z_geo, jnp.asarray([mu_last_ari / last_share])))
        sigma_z_geo_all = jnp.concatenate(
            (sigma_z_geo, jnp.asarray([sigma_last_ari / last_share]))
        )

        kappa = kappa_fn(state)
        dkappa = jax.jacrev(kappa_fn)(state)
        d2_kappa_sigma = directional_second_derivative(
            kappa_fn, state, sigma_z_ari
        )
        q = q_fn(state)
        dq = jax.jacrev(q_fn)(state)
        d2_q_sigma = directional_second_derivative(q_fn, state, sigma_z_ari)

        mu_q = (dq @ mu_z_ari + 0.5 * d2_q_sigma) / jnp.maximum(q, eps)
        sigma_q = (dq @ sigma_z_ari) / jnp.maximum(q, eps)
        mu_kappa = mu_z_geo_all - mu_q + sigma_q * (
            sigma_q - sigma_z_geo_all
        )
        sigma_kappa = sigma_z_geo_all - sigma_q

        hjb = dkappa @ mu_z_ari + 0.5 * d2_kappa_sigma - mu_kappa * kappa
        consistency = dkappa @ sigma_z_ari - sigma_kappa * kappa
        equal_kappa = kappa[1:] - kappa[:1]
        interest_rate = (
            self.config.rho
            + self.config.gamma * weighted_mu
            - 0.5
            * self.config.gamma
            * (self.config.gamma + 1.0)
            * jnp.sum(jnp.square(self.sigma_ys * z_all))
        )

        return {
            "kappa": kappa,
            "q": q,
            "z_all": z_all,
            "mu_z_ari": mu_z_ari,
            "sigma_z_ari": sigma_z_ari,
            "mu_kappa": mu_kappa,
            "sigma_kappa": sigma_kappa,
            "interest_rate": interest_rate,
            "hjb": hjb,
            "consistency": consistency,
            "equal_kappa": equal_kappa,
        }

    def evaluate(self, params: Any, states: Array) -> dict[str, Array]:
        return jax.vmap(lambda state: self._point_quantities(params, state))(states)

    def loss_terms(self, params: Any, states: Array) -> dict[str, Array]:
        values = self.evaluate(params, states)
        return {
            "hjb": values["hjb"],
            "consistency": values["consistency"],
            "equal_kappa": values["equal_kappa"],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_trees": self.n_trees,
            "state_dim": self.state_dim,
            "gamma": self.config.gamma,
            "rho": self.config.rho,
            "mu_ys": [float(x) for x in self.mu_ys],
            "sigma_ys": [float(x) for x in self.sigma_ys],
            "architecture": self.config.architecture,
            "network": self.network.metadata(),
            "second_order_method": "directional_jvp",
        }
