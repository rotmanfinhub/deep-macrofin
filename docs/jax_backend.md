# JAX backend

The optional JAX backend is a JAX-native neural PDE solver built from the
equations used by Deep-MacroFin's Basak–Cuoco and tree examples. It is a
compiled implementation, not a wrapper around PyTorch.

The first release deliberately supports the two requested model families:

- `BasakCuocoModel`: three states, three positive value/consumption policies,
  and two portfolio policies.
- `TreeModel`: an arbitrary number of trees, represented by `n_trees - 1`
  simplex states and a positive, permutation-equivariant DeepSet by default.

## Why the tree implementation scales better

The PyTorch notebook constructs full Hessians. A full Hessian has quadratic
storage in the number of states. These models only need contractions of the form

`sigma.T @ Hessian(f) @ sigma`.

This package computes that quantity with nested JAX Jacobian-vector products,
without materializing the Hessian. It also draws tree shares from a simplex;
independent uniform samples are not valid tree shares once the dimension grows.
The default DeepSet shares its encoder and decoder across tree states, so its
parameter count does not grow when more trees are added. A regular vector MLP is
also available with `TreeConfig(architecture="mlp")` for notebook parity.

This removes the most avoidable quadratic-memory term. It does **not** eliminate
the underlying curse of dimensionality: more states still require more samples,
network capacity, compilation time, and training time.

## Install

For users:

```bash
python -m pip install "deep_macrofin[jax]"
```

The JAX implementation is runtime-isolated in the `deep_macrofin_jax` package
and does not import or execute PyTorch. It is distributed with Deep-MacroFin,
however, so installation also includes the project's existing required PyTorch
dependency.

For an editable development checkout:

```bash
python -m pip install -e ".[jax]"
```

For a CUDA machine, install the appropriate JAX GPU wheel first, following the
official JAX installation instructions, and then install this package.

## Train and save a model

```python
from deep_macrofin_jax import PDETrainer, TrainingConfig
from deep_macrofin_jax.models import TreeConfig, TreeModel

model = TreeModel(TreeConfig(n_trees=7))  # six states: five more than tree-2
trainer = PDETrainer(
    model,
    TrainingConfig(steps=10_000, batch_size=256, learning_rate=1e-3),
)
result = trainer.fit(checkpoint_path="checkpoints/tree_7.npz")
print(result.history[-1])
```

Training runs in compiled chunks using `jax.jit` and `jax.lax.scan`. The saved
checkpoint is a compressed, pickle-free `.npz` file containing parameters plus
the model/training metadata needed to validate a warm start.

## Warm-start for different economic parameters

Network weights are independent of the optimizer state, so load them and create
a new trainer with the changed model parameters:

```python
from deep_macrofin_jax import PDETrainer, TrainingConfig, load_checkpoint
from deep_macrofin_jax.models import TreeConfig, TreeModel

params, old_metadata = load_checkpoint("checkpoints/tree_7.npz")
model = TreeModel(
    TreeConfig(
        n_trees=7,
        gamma=6.0,
        mu_ys=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07),
        sigma_ys=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07),
    )
)
result = PDETrainer(model, TrainingConfig(steps=2_000)).fit(
    params=params,
    checkpoint_path="checkpoints/tree_7_gamma6.npz",
)
```

Warm starts require the same network input/output dimensions and architecture.
Changing economic parameters is fine. Changing the number of states or hidden
layers requires a new network (or an explicit parameter-transfer rule).

The command-line examples provide the same workflow:

```bash
python examples/macro_problems/tree/train_tree_jax.py --trees 7 --steps 10000
python examples/macro_problems/tree/train_tree_jax.py --trees 7 --steps 2000 \
  --warm-start checkpoints/tree.npz --checkpoint checkpoints/tree_updated.npz
python examples/macro_problems/basak_cuoco/train_basak_cuoco_jax.py --steps 20000
```

## Evaluate a trained solution

```python
import jax.numpy as jnp
from deep_macrofin_jax import load_checkpoint
from deep_macrofin_jax.models import TreeConfig, TreeModel

params, metadata = load_checkpoint("checkpoints/tree_7.npz")
model = TreeModel(TreeConfig(n_trees=7))
states = jnp.asarray([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
fields = model.evaluate(params, states)
print(fields["kappa"], fields["q"])
```

## Tests and scaling benchmark

```bash
pytest
python examples/macro_problems/tree/tree_scaling_jax.py --trees 2 7 12 --batch-size 32
```

The test suite differentiates through both model residuals, verifies model
identities, checks checkpoint/warm-start round trips, and runs the tree model at
seven trees (six states, exactly five additional states over the two-tree case).
Recorded CPU smoke-test results are in [the JAX validation report](jax_validation.md).

## Current scope

This is the performance-oriented core needed by the two requested examples. It
uses Python callables for equations so JAX sees an immutable graph at compile
time. Deep-MacroFin's runtime string/LaTeX equation parser, KAN/MultKAN layers,
time-stepping solver, plotting helpers, and other example models are not yet
ported. Those features should be added around this callable core rather than by
putting dynamic `eval` calls inside the compiled training step.
