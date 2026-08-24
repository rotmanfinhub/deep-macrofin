from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import numpy as np

from deep_macrofin_jax.models import (
    BasakCuocoConfig,
    BasakCuocoModel,
    TreeConfig,
    TreeModel,
)


def _mean_square_loss(model, params, states):
    return sum(
        jnp.mean(jnp.square(value))
        for value in model.loss_terms(params, states).values()
    )


def test_basak_cuoco_residuals_and_parameter_gradients_are_finite():
    model = BasakCuocoModel(BasakCuocoConfig(hidden_units=(8, 8)))
    params = model.init(jax.random.PRNGKey(0))
    states = model.sample(jax.random.PRNGKey(1), 4)

    loss_and_grad = jax.jit(
        jax.value_and_grad(
            lambda current_params, batch: _mean_square_loss(
                model, current_params, batch
            )
        )
    )
    loss, grads = loss_and_grad(params, states)
    values = jax.jit(model.evaluate)(params, states)

    assert states.shape == (4, 3)
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)
    )
    np.testing.assert_allclose(
        values["pi"],
        values["q"] * jnp.square(values["sigma_r_norm"]),
        rtol=2e-5,
        atol=2e-7,
    )
    assert jnp.all(values["xi"] > 0)


def test_tree_matches_accounting_identity_and_has_finite_gradients():
    model = TreeModel(
        TreeConfig(
            n_trees=2, deepset_phi_units=(8, 8), deepset_rho_units=(8, 8)
        )
    )
    params = model.init(jax.random.PRNGKey(2))
    states = model.sample(jax.random.PRNGKey(3), 5)

    values = jax.jit(model.evaluate)(params, states)
    loss_and_grad = jax.jit(
        jax.value_and_grad(
            lambda current_params, batch: _mean_square_loss(
                model, current_params, batch
            )
        )
    )
    loss, grads = loss_and_grad(params, states)

    np.testing.assert_allclose(
        values["q"] * values["kappa"], values["z_all"], rtol=2e-5, atol=2e-7
    )
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)
    )


def test_tree_handles_five_additional_state_variables_on_the_simplex():
    # The original two-tree model has one state. Seven trees have six states:
    # exactly five additional state variables.
    model = TreeModel(
        TreeConfig(
            n_trees=7, deepset_phi_units=(8, 8), deepset_rho_units=(8, 8)
        )
    )
    params = model.init(jax.random.PRNGKey(4))
    states = model.sample(jax.random.PRNGKey(5), 3)
    values = jax.jit(model.evaluate)(params, states)
    terms = jax.jit(model.loss_terms)(params, states)
    loss_and_grad = jax.jit(
        jax.value_and_grad(
            lambda current_params, batch: _mean_square_loss(
                model, current_params, batch
            )
        )
    )
    loss, grads = loss_and_grad(params, states)

    assert states.shape == (3, 6)
    assert values["kappa"].shape == (3, 7)
    assert terms["hjb"].shape == (3, 7)
    assert terms["consistency"].shape == (3, 7)
    assert jnp.all(values["z_all"] >= model.config.min_share)
    np.testing.assert_allclose(jnp.sum(values["z_all"], axis=1), 1.0, atol=2e-7)
    assert all(jnp.all(jnp.isfinite(value)) for value in terms.values())
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)
    )

    permutation = jnp.asarray([2, 0, 5, 1, 4, 3])
    original_prediction = model.predict(params, states[:1])[0]
    permuted_prediction = model.predict(params, states[:1, permutation])[0]
    np.testing.assert_allclose(
        permuted_prediction[:-1], original_prediction[:-1][permutation], atol=2e-6
    )
    np.testing.assert_allclose(
        permuted_prediction[-1], original_prediction[-1], atol=2e-6
    )
