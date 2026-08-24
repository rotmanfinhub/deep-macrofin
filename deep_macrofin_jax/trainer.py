"""Compiled optimization loop for callable PDE residual models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Protocol

import jax
import jax.numpy as jnp
import optax

from .checkpoint import save_checkpoint

Array = jax.Array


class ResidualModel(Protocol):
    name: str

    def init(self, key: Array) -> Any: ...

    def sample(self, key: Array, batch_size: int) -> Array: ...

    def loss_terms(self, params: Any, states: Array) -> Mapping[str, Array]: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    compile_chunk_size: int = 100
    seed: int = 42
    loss_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.compile_chunk_size < 1:
            raise ValueError("steps, batch_size, and compile_chunk_size must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")


@dataclass
class TrainingResult:
    params: Any
    history: list[dict[str, float]]
    elapsed_seconds: float
    checkpoint_path: Path | None = None


class PDETrainer:
    """Train a static callable residual graph with JIT and ``lax.scan``."""

    def __init__(self, model: ResidualModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        weights = dict(config.loss_weights)

        def loss_fn(params: Any, states: Array) -> tuple[Array, Mapping[str, Array]]:
            terms = model.loss_terms(params, states)
            mean_squares = {
                name: jnp.mean(jnp.square(value)) for name, value in terms.items()
            }
            total = sum(
                weights.get(name, 1.0) * value for name, value in mean_squares.items()
            )
            return total, mean_squares

        self._loss_fn = loss_fn

    def loss(self, params: Any, states: Array) -> tuple[Array, Mapping[str, Array]]:
        return self._loss_fn(params, states)

    def fit(
        self,
        params: Any | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> TrainingResult:
        config = self.config
        key = jax.random.PRNGKey(config.seed)
        key, init_key = jax.random.split(key)
        if params is None:
            params = self.model.init(init_key)

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adam(config.learning_rate),
        )
        optimizer_state = optimizer.init(params)

        def build_chunk(num_steps: int):
            @jax.jit
            def run_chunk(params: Any, optimizer_state: Any, key: Array):
                def step(carry: tuple[Any, Any, Array], _: None):
                    current_params, current_state, current_key = carry
                    current_key, sample_key = jax.random.split(current_key)
                    states = self.model.sample(sample_key, config.batch_size)
                    (loss_value, terms), grads = jax.value_and_grad(
                        self._loss_fn, has_aux=True
                    )(current_params, states)
                    updates, current_state = optimizer.update(
                        grads, current_state, current_params
                    )
                    current_params = optax.apply_updates(current_params, updates)
                    metrics = {"loss": loss_value, **terms}
                    return (current_params, current_state, current_key), metrics

                return jax.lax.scan(
                    step, (params, optimizer_state, key), xs=None, length=num_steps
                )

            return run_chunk

        full_chunk = build_chunk(config.compile_chunk_size)
        remainder = config.steps % config.compile_chunk_size
        last_chunk = build_chunk(remainder) if remainder else None
        history: list[dict[str, float]] = []
        started = perf_counter()
        completed = 0

        while completed < config.steps:
            remaining = config.steps - completed
            if remaining >= config.compile_chunk_size:
                runner = full_chunk
                count = config.compile_chunk_size
            else:
                assert last_chunk is not None
                runner = last_chunk
                count = remaining
            (params, optimizer_state, key), metrics = runner(params, optimizer_state, key)
            jax.block_until_ready(metrics["loss"])
            completed += count
            record = {"step": float(completed)}
            record.update({name: float(values[-1]) for name, values in metrics.items()})
            history.append(record)

        elapsed = perf_counter() - started
        saved_path: Path | None = None
        if checkpoint_path is not None:
            metadata = {
                "model": self.model.metadata(),
                "training": {
                    "steps": config.steps,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "grad_clip": config.grad_clip,
                    "seed": config.seed,
                },
                "final_metrics": history[-1],
            }
            saved_path = save_checkpoint(checkpoint_path, params, metadata)
        return TrainingResult(params, history, elapsed, saved_path)
