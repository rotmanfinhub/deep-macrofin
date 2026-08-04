"""
Unit tests for the experimental "stacked" (vmap-batched) evaluation of
agents/endogenous variables.

The stacked path is confined to ``local_function_dict`` (the forward values and
their derivatives) and ONLY supports networks configured with
``batch_jac_hes=True``. These tests verify:
    - numerical parity between stacked and per-network evaluation,
    - a stackable MLP with ``batch_jac_hes=False`` raises ``ValueError``,
    - correct architecture-signature grouping (final Softplus splits groups,
      hardcoded / non-stackable variables fall back),
    - gradients still flow back to every underlying network through the stacked
      parameter tensors,
    - the ``stacked`` config flag / lazy evaluator wiring behaves.
"""

import unittest

import pytest
import torch

from deep_macrofin import PDEModel, PDEModelTimeStep, set_seeds
from deep_macrofin.models import (StackedFunctionEvaluator,
                                  StackedNetworkGroup)

TOL = dict(atol=1e-5, rtol=1e-4)


def _run_update(model, SV):
    """Run a single forward through ``update_variables`` into a fresh dict."""
    vd = {}
    model.update_variables(SV, vd)
    return vd


def _assert_dicts_close(test, ref, other, keys):
    for key in keys:
        test.assertIn(key, ref, f"{key} missing from reference dict")
        test.assertIn(key, other, f"{key} missing from stacked dict")
        a, b = ref[key], other[key]
        test.assertEqual(a.shape, b.shape, f"{key}: shape {a.shape} vs {b.shape}")
        test.assertTrue(
            torch.allclose(a, b, **TOL),
            f"{key}: max abs diff {torch.max(torch.abs(a - b)).item():.3e}",
        )


class TestStackedParityStationary(unittest.TestCase):
    """Stacked vs. per-network parity for the stationary ``PDEModel``."""

    def _build(self):
        set_seeds(0)
        model = PDEModel("stationary", config={"batch_size": 32})
        model.set_state(["x", "y"], {"x": [0, 1], "y": [0, 1]})

        def cfg(**kw):
            return {"hidden_units": [8, 8], "batch_jac_hes": True, **kw}

        # three identical MLP agents -> one stacked group
        model.add_agents(["a1", "a2", "a3"], {n: cfg() for n in ["a1", "a2", "a3"]})
        # two positive (final softplus) endog vars -> a separate stacked group
        model.add_endogs(["p1", "p2"], {n: cfg(positive=True) for n in ["p1", "p2"]})
        # a hardcoded endog var -> not stackable, must fall back to per-network eval
        model.add_endog(
            "hc",
            {"hardcode_function": lambda x: x[..., 0:1] ** 2 + x[..., 1:] ** 3,
             "batch_jac_hes": True},
        )
        return model

    def test_parity(self):
        model = self._build()
        SV = torch.rand((32, 2), requires_grad=True)

        model.stacked = False
        model._invalidate_stacked_evaluator()
        ref = _run_update(model, SV)

        model.stacked = True
        model._invalidate_stacked_evaluator()
        got = _run_update(model, SV)

        _assert_dicts_close(self, ref, got, list(model.local_function_dict.keys()))

    def test_hardcoded_matches_and_is_not_stacked(self):
        model = self._build()
        evaluator = StackedFunctionEvaluator(model.agents, model.endog_vars)
        covered = evaluator.covered_keys()
        # hardcoded variable and its derivatives are never covered by stacking
        self.assertNotIn("hc", covered)
        self.assertNotIn("hc_Jac", covered)
        self.assertNotIn("hc_Hess", covered)


class TestStackedRequiresBatchJacHes(unittest.TestCase):
    """A stackable MLP with batch_jac_hes=False must raise."""

    def test_batch_jac_hes_false_raises(self):
        set_seeds(5)
        model = PDEModel("needs_bjh", config={"batch_size": 8, "stacked": True})
        model.set_state(["x"], {"x": [0, 1]})
        model.add_agent("a", {"hidden_units": [8, 8], "batch_jac_hes": False})
        with pytest.raises(ValueError):
            model._get_stacked_evaluator()

    def test_batch_jac_hes_false_raises_on_forward(self):
        set_seeds(6)
        model = PDEModel("needs_bjh2", config={"batch_size": 8, "stacked": True})
        model.set_state(["x"], {"x": [0, 1]})
        model.add_agent("a", {"hidden_units": [8, 8], "batch_jac_hes": False})
        SV = torch.rand((8, 1), requires_grad=True)
        with pytest.raises(ValueError):
            model.update_variables(SV, {})


class TestStackedGrouping(unittest.TestCase):
    """Architecture-signature grouping behaviour."""

    def _model(self):
        set_seeds(1)
        model = PDEModel("grouping", config={"batch_size": 8})
        model.set_state(["x", "y"], {"x": [0, 1], "y": [0, 1]})
        model.add_agents(
            ["a1", "a2", "a3"],
            {n: {"hidden_units": [8, 8], "batch_jac_hes": True} for n in ["a1", "a2", "a3"]},
        )
        model.add_endogs(
            ["p1", "p2"],
            {n: {"hidden_units": [8, 8], "positive": True, "batch_jac_hes": True}
             for n in ["p1", "p2"]},
        )
        model.add_endog("hc", {"hardcode_function": lambda x: x[..., 0:1]})
        return model

    def test_softplus_splits_into_two_groups(self):
        model = self._model()
        evaluator = StackedFunctionEvaluator(model.agents, model.endog_vars)
        # {a1,a2,a3} and {p1,p2}; hardcoded 'hc' excluded entirely
        self.assertEqual(len(evaluator.groups), 2)
        sizes = sorted(len(g.variables) for g in evaluator.groups)
        self.assertEqual(sizes, [2, 3])

    def test_covered_keys(self):
        model = self._model()
        evaluator = StackedFunctionEvaluator(model.agents, model.endog_vars)
        covered = evaluator.covered_keys()
        for name in ["a1", "a2", "a3", "p1", "p2"]:
            self.assertIn(name, covered)
            self.assertIn(f"{name}_Jac", covered)
            self.assertIn(f"{name}_Hess", covered)
        self.assertNotIn("hc", covered)

    def test_single_network_group_still_stacks(self):
        set_seeds(2)
        model = PDEModel("single", config={"batch_size": 8})
        model.set_state(["x"], {"x": [0, 1]})
        model.add_agent("a", {"hidden_units": [8, 8], "batch_jac_hes": True})
        evaluator = StackedFunctionEvaluator(model.agents, model.endog_vars)
        self.assertEqual(len(evaluator.groups), 1)
        self.assertIsInstance(evaluator.groups[0], StackedNetworkGroup)


class TestStackedGradients(unittest.TestCase):
    """Gradients must flow back to each underlying network in a stacked group."""

    def test_gradients_reach_each_network(self):
        set_seeds(3)
        model = PDEModel("grads", config={"batch_size": 16})
        model.set_state(["x", "y"], {"x": [0, 1], "y": [0, 1]})
        model.add_agents(
            ["a1", "a2"],
            {n: {"hidden_units": [8, 8], "batch_jac_hes": True} for n in ["a1", "a2"]},
        )
        model.stacked = True
        model._invalidate_stacked_evaluator()

        SV = torch.rand((16, 2), requires_grad=True)
        vd = _run_update(model, SV)
        loss = vd["a1"].sum() + vd["a2"].sum()
        loss.backward()

        for name in ["a1", "a2"]:
            grads = [p.grad for p in model.agents[name].parameters()]
            self.assertTrue(all(g is not None for g in grads),
                            f"{name} has parameters without gradients")
            self.assertTrue(any(torch.any(g != 0) for g in grads),
                            f"{name} received only zero gradients")


class TestStackedParityTimeStep(unittest.TestCase):
    """Stacked vs. per-network parity for the time-stepping model."""

    def _build(self):
        set_seeds(4)
        model = PDEModelTimeStep("timestep", config={"batch_size": 24})
        # time is appended automatically -> state = [z1, z2, time]
        model.set_state(["z1", "z2"], {"z1": [0.01, 0.99], "z2": [0.01, 0.99]})
        model.add_endogs(
            ["k1", "k2", "k3"],
            {n: {"hidden_units": [16, 16], "positive": True, "batch_jac_hes": True}
             for n in ["k1", "k2", "k3"]},
        )
        return model

    def test_parity(self):
        model = self._build()
        n_dim = len(model.state_variables)
        SV = torch.rand((24, n_dim), requires_grad=True)

        model.stacked = False
        model._invalidate_stacked_evaluator()
        ref = _run_update(model, SV)

        model.stacked = True
        model._invalidate_stacked_evaluator()
        got = _run_update(model, SV)

        _assert_dicts_close(self, ref, got, list(model.local_function_dict.keys()))


class TestStackedConfigWiring(unittest.TestCase):
    """The ``stacked`` config flag and lazy evaluator lifecycle."""

    def test_flag_defaults_false(self):
        model = PDEModel("cfg_default", config={"batch_size": 8})
        model.set_state(["x"], {"x": [0, 1]})
        self.assertFalse(model.stacked)

    def test_flag_from_config(self):
        model = PDEModel("cfg_on", config={"batch_size": 8, "stacked": True})
        model.set_state(["x"], {"x": [0, 1]})
        self.assertTrue(model.stacked)

    def test_evaluator_is_lazy_and_invalidated(self):
        model = PDEModel("cfg_lazy", config={"batch_size": 8, "stacked": True})
        model.set_state(["x"], {"x": [0, 1]})
        model.add_agent("a", {"hidden_units": [8, 8], "batch_jac_hes": True})
        # not built until first use
        self.assertIsNone(model._stacked_evaluator)
        first = model._get_stacked_evaluator()
        self.assertIsNotNone(first)
        # adding a variable invalidates the cached evaluator
        model.add_agent("b", {"hidden_units": [8, 8], "batch_jac_hes": True})
        self.assertIsNone(model._stacked_evaluator)
        second = model._get_stacked_evaluator()
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()
