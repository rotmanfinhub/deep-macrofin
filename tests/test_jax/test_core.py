from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import numpy as np

from deep_macrofin_jax import (
    PDETrainer,
    TrainingConfig,
    directional_second_derivative,
    load_checkpoint,
)
from deep_macrofin_jax.models import TreeConfig, TreeModel


def test_directional_second_derivative_matches_explicit_hessian():
    def function(x):
        return jnp.asarray(
            [x[0] ** 3 + x[0] * x[1], jnp.sin(x[0] - 2.0 * x[1])]
        )

    point = jnp.asarray([0.4, -0.2])
    direction = jnp.asarray([0.7, 0.3])
    actual = directional_second_derivative(function, point, direction)
    hessian = jax.jacfwd(jax.jacrev(function))(point)
    expected = jnp.einsum("i,oij,j->o", direction, hessian, direction)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_training_and_pickle_free_warm_start_round_trip(tmp_path):
    model = TreeModel(
        TreeConfig(
            n_trees=2, deepset_phi_units=(8,), deepset_rho_units=(8,)
        )
    )
    trainer = PDETrainer(
        model,
        TrainingConfig(
            steps=2,
            batch_size=4,
            learning_rate=1e-3,
            compile_chunk_size=2,
            seed=7,
        ),
    )
    result = trainer.fit(checkpoint_path=tmp_path / "tree_warmstart")
    loaded, metadata = load_checkpoint(result.checkpoint_path)

    original_leaves = jax.tree_util.tree_leaves(result.params)
    loaded_leaves = jax.tree_util.tree_leaves(loaded)
    assert len(original_leaves) == len(loaded_leaves)
    for original, restored in zip(original_leaves, loaded_leaves):
        np.testing.assert_array_equal(original, restored)
    assert metadata["model"]["name"] == "tree"
    assert metadata["training"]["steps"] == 2
    assert np.isfinite(result.history[-1]["loss"])
