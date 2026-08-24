"""Measure compiled residual/gradient time as tree state dimension grows."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from deep_macrofin_jax.models import TreeConfig, TreeModel


def benchmark(n_trees: int, batch_size: int, repeats: int) -> dict[str, float]:
    model = TreeModel(
        TreeConfig(
            n_trees=n_trees,
            deepset_phi_units=(32, 32),
            deepset_rho_units=(32, 32),
        )
    )
    params = model.init(jax.random.PRNGKey(n_trees))
    states = model.sample(jax.random.PRNGKey(n_trees + 1), batch_size)

    def loss_fn(current_params, current_states):
        terms = model.loss_terms(current_params, current_states)
        return sum(jnp.mean(jnp.square(term)) for term in terms.values())

    step = jax.jit(jax.value_and_grad(loss_fn))
    started = time.perf_counter()
    first = step(params, states)
    jax.block_until_ready(first[0])
    compile_and_first = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(repeats):
        result = step(params, states)
    jax.block_until_ready(result[0])
    steady = (time.perf_counter() - started) / repeats
    return {
        "trees": float(n_trees),
        "states": float(n_trees - 1),
        "parameters": float(
            sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        ),
        "compile_and_first_seconds": compile_and_first,
        "steady_step_seconds": steady,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, nargs="+", default=[2, 7, 12])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for count in args.trees:
        print(benchmark(count, args.batch_size, args.repeats))


if __name__ == "__main__":
    main()
