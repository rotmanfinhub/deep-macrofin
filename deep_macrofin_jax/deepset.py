"""Permutation-equivariant DeepSet network implemented with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .networks import MLP, MLPConfig

Array = jax.Array


@dataclass(frozen=True)
class DeepSetConfig:
    phi_units: tuple[int, ...] = (80, 80)
    rho_units: tuple[int, ...] = (80, 80)
    activation: str = "tanh"
    positive: bool = False
    positive_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not self.phi_units or any(width < 1 for width in self.phi_units):
            raise ValueError("phi_units must contain positive widths")
        if any(width < 1 for width in self.rho_units):
            raise ValueError("rho_units must contain positive widths")


class DeepSet:
    """Shared element encoder with equivariant and invariant output heads.

    The first ``input_size_sym`` outputs permute with the inputs. Any remaining
    outputs come from a pooled invariant head. This matches the representation
    used by Deep-MacroFin's scalable tree notebook, where the final tree share is
    implied by the other shares.
    """

    def __init__(
        self,
        input_size_sym: int,
        output_size: int,
        lower_bound: float,
        upper_bound: float,
        config: DeepSetConfig = DeepSetConfig(),
    ) -> None:
        if input_size_sym < 1:
            raise ValueError("input_size_sym must be positive")
        if output_size < input_size_sym:
            raise ValueError("output_size cannot be smaller than input_size_sym")
        self.input_size_sym = input_size_sym
        self.output_size = output_size
        self.config = config
        embedding_size = config.phi_units[-1]
        positive_equivariant = (0,) if config.positive else ()
        positive_invariant = (
            tuple(range(output_size - input_size_sym)) if config.positive else ()
        )

        self.phi = MLP(
            1,
            embedding_size,
            MLPConfig(
                hidden_units=config.phi_units[:-1],
                activation=config.activation,
            ),
            lower_bounds=(lower_bound,),
            upper_bounds=(upper_bound,),
        )
        self.rho = MLP(
            2 * embedding_size,
            1,
            MLPConfig(
                hidden_units=config.rho_units,
                activation=config.activation,
                positive_outputs=positive_equivariant,
                positive_floor=config.positive_floor,
                output_bias=(0.5413248546,),
                normalize_inputs=False,
            ),
            lower_bounds=(0.0,) * (2 * embedding_size),
            upper_bounds=(1.0,) * (2 * embedding_size),
        )
        self.pooler = None
        extra_outputs = output_size - input_size_sym
        if extra_outputs:
            self.pooler = MLP(
                embedding_size,
                extra_outputs,
                MLPConfig(
                    hidden_units=config.rho_units,
                    activation=config.activation,
                    positive_outputs=positive_invariant,
                    positive_floor=config.positive_floor,
                    output_bias=(0.5413248546,) * extra_outputs,
                    normalize_inputs=False,
                ),
                lower_bounds=(0.0,) * embedding_size,
                upper_bounds=(1.0,) * embedding_size,
            )

    def init(self, key: Array) -> dict[str, Any]:
        phi_key, rho_key, pooler_key = jax.random.split(key, 3)
        params = {"phi": self.phi.init(phi_key), "rho": self.rho.init(rho_key)}
        if self.pooler is not None:
            params["pooler"] = self.pooler.init(pooler_key)
        return params

    def apply(self, params: dict[str, Any], states: Array) -> Array:
        encoded = self.phi.apply(params["phi"], states[..., :, None])
        aggregate = jnp.mean(encoded, axis=-2, keepdims=True)
        aggregate_each = jnp.broadcast_to(aggregate, encoded.shape)
        combined = jnp.concatenate((encoded, aggregate_each), axis=-1)
        equivariant = self.rho.apply(params["rho"], combined)[..., 0]
        if self.pooler is None:
            return equivariant
        invariant = self.pooler.apply(params["pooler"], aggregate[..., 0, :])
        return jnp.concatenate((equivariant, invariant), axis=-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "deepset",
            "input_size_sym": self.input_size_sym,
            "output_size": self.output_size,
            "phi_units": list(self.config.phi_units),
            "rho_units": list(self.config.rho_units),
            "activation": self.config.activation,
            "positive": self.config.positive,
        }
