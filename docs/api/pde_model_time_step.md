# deep_macrofin.pde_model_time_step

This is a subclass of [PDEModel](./pde_model.md) that implements the time stepping scheme with neural network. It has exactly the same API as the base class.

## PDEModelTimeStep
```py
class PDEModelTimeStep(PDEModel):
'''
PDEModelTimeStep uses time stepping scheme + neural network to solve for optimality

PDEModel class to assign variables, equations & constraints, etc.

Also initialize the neural network architectures for each agent/endogenous variables 
with some config dictionary.
'''
```