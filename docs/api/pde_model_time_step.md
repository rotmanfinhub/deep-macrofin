# deep_macrofin.pde_model_time_step

This implements the time stepping scheme with neural network. It shares the same abstract base class [`BasePDEModel`](./base_pde_model.md) as [`PDEModel`](./pde_model.md), so it has exactly the same variable/equation/constraint definition API; only the training loop and a few sampling helpers differ.

## PDEModelTimeStep
```py
class PDEModelTimeStep(BasePDEModel):
'''
PDEModelTimeStep uses time stepping scheme + neural network to solve for optimality

PDEModel class to assign variables, equations & constraints, etc.

Also initialize the neural network architectures for each agent/endogenous variables 
with some config dictionary.
'''
```

The time-stepping config additionally accepts the experimental `stacked` flag (default False), which batches all same-architecture agents/endogenous variables into a single `vmap` forward/derivative call. See [Experimental: stacked evaluation](../usage.md#experimental-stacked-vmap-batched-evaluation).

### set_initial_guess
```py
def set_initial_guess(self, initial_guess: Dict[str, float])
```

Set the initial guess (uniform value across the state variable domain) for agents or endogenous variables. This is the boundary condition at $t=T$ in the first time iteration.