# Validation report

Validation was run on macOS arm64 with one CPU device, JAX 0.7.2, and Optax
0.2.6. These are smoke/performance checks for implementation correctness, not
claims about final economic calibration accuracy.

## Automated tests

All five tests pass. They cover:

- Basak–Cuoco residual evaluation and reverse-mode gradients through the PDE;
- the tree accounting identity `q * kappa = z` and PDE gradients;
- seven trees / six independent states, which adds exactly five state variables
  to the original two-tree / one-state case;
- equivalence of the directional second derivative to an explicitly constructed
  Hessian contraction; and
- two optimizer steps followed by an exact parameter checkpoint round trip.

## Optimization smoke test

With 16-unit hidden layers, batch size 64, and 500 Adam steps:

| Model | Loss at step 100 | Loss at step 500 | Elapsed |
|---|---:|---:|---:|
| Tree (2 trees, DeepSet) | 1.2121e-5 | 1.0928e-7 | 2.07 s |
| Basak–Cuoco | 4.7395e-2 | 8.2622e-3 | 1.61 s |

The batch is resampled, so these figures only establish that compiled optimizer
updates execute and improve the sampled objective over this short run.

## Tree scaling smoke test

Command:

```bash
python examples/macro_problems/tree/tree_scaling_jax.py --trees 2 7 12 --batch-size 32 --repeats 5
```

The benchmark uses two 32-unit hidden layers and measures a compiled loss plus
parameter-gradient evaluation.

| Trees | Independent states | Parameters | Compile + first step | Steady step |
|---:|---:|---:|---:|---:|
| 2 | 1 | 6,434 | 1.580 s | 0.392 ms |
| 7 | 6 | 6,434 | 1.478 s | 1.420 ms |
| 12 | 11 | 6,434 | 1.374 s | 3.243 ms |

Compilation timings are noisy and shape-specific. The useful observation is
that the requested six-state model compiles and differentiates without forming
per-output full Hessian tensors, while the shared DeepSet keeps parameter count
constant. GPU performance should be measured separately on the intended Colab
runtime.
