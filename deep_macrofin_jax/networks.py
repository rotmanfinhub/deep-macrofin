"""Small functional neural-network building blocks with JAX-friendly pytrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
Params = tuple[dict[str, Array], ...]


@dataclass(frozen=True)
class MLPConfig:
    """Static MLP configuration.

    ``positive_outputs`` are transformed with softplus. ``output_bias`` is the
    raw, pre-transform bias and is useful for choosing an economically sensible
    initial scale.
    """

    hidden_units: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    positive_outputs: tuple[int, ...] = ()
    positive_floor: float = 1e-6
    output_bias: tuple[float, ...] | None = None
    normalize_inputs: bool = True


def _activation(name: str) -> Callable[[Array], Array]:
    activations = {
        "tanh": jnp.tanh,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose {tuple(activations)}") from exc


class MLP:
    """A normalized-input, vector-output functional MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: MLPConfig,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        if len(lower_bounds) != input_dim or len(upper_bounds) != input_dim:
            raise ValueError("bounds must have one entry per input")
        if config.output_bias is not None and len(config.output_bias) != output_dim:
            raise ValueError("output_bias must have one entry per output")
        if any(i < 0 or i >= output_dim for i in config.positive_outputs):
            raise ValueError("positive output index is outside the network output")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.lower = jnp.asarray(lower_bounds)
        self.upper = jnp.asarray(upper_bounds)
        self._activation_fn = _activation(config.activation)

    def init(self, key: Array) -> Params:
        widths = (self.input_dim, *self.config.hidden_units, self.output_dim)
        keys = jax.random.split(key, len(widths) - 1)
        layers: list[dict[str, Array]] = []
        for i, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            weight = jax.random.uniform(
                keys[i], (fan_in, fan_out), minval=-limit, maxval=limit
            )
            bias = jnp.zeros((fan_out,), dtype=weight.dtype)
            layers.append({"w": weight, "b": bias})
        if self.config.output_bias is not None:
            layers[-1]["b"] = jnp.asarray(
                self.config.output_bias, dtype=layers[-1]["b"].dtype
            )
        return tuple(layers)

    def normalize(self, states: Array) -> Array:
        if not self.config.normalize_inputs:
            return states
        width = self.upper - self.lower
        safe_width = jnp.where(width > 0, width, 1.0)
        normalized = 2.0 * (states - self.lower) / safe_width - 1.0
        return jnp.where(width > 0, normalized, 0.0)

    def apply(self, params: Params, states: Array) -> Array:
        value = self.normalize(states)
        for layer in params[:-1]:
            value = self._activation_fn(value @ layer["w"] + layer["b"])
        value = value @ params[-1]["w"] + params[-1]["b"]
        if self.config.positive_outputs:
            indices = jnp.asarray(self.config.positive_outputs)
            positive = jax.nn.softplus(value[..., indices]) + self.config.positive_floor
            value = value.at[..., indices].set(positive)
        return value

    def metadata(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_units": list(self.config.hidden_units),
            "activation": self.config.activation,
            "positive_outputs": list(self.config.positive_outputs),
            "positive_floor": self.config.positive_floor,
            "normalize_inputs": self.config.normalize_inputs,
            "lower_bounds": [float(x) for x in self.lower],
            "upper_bounds": [float(x) for x in self.upper],
        }


def directional_second_derivative(
    function: Callable[[Array], Array], point: Array, direction: Array
) -> Array:
    """Return ``direction.T @ Hessian(function) @ direction``.

    This uses nested Jacobian-vector products, so it does not materialize a
    ``state_dim x state_dim`` Hessian. It works for scalar or vector outputs.
    The direction is intentionally held fixed at ``point`` as required by the
    local Itô contraction.
    """

    first_directional = lambda x: jax.jvp(function, (x,), (direction,))[1]
    return jax.jvp(first_directional, (point,), (direction,))[1]
