# deep_macrofin.base_pde_model

`BasePDEModel` is the abstract base class shared by [`PDEModel`](./pde_model.md) (stationary solver) and [`PDEModelTimeStep`](./pde_model_time_step.md) (time-stepping solver). It is not meant to be instantiated directly; instead, use one of the two concrete subclasses.

It holds everything the two solvers have in common:

- **Problem definition API**: `set_state`, `set_state_constraints`, `add_param(s)`, `add_learnable_param(s)`, `add_agent(s)`, `add_agent_condition`, `add_endog(s)`, `add_endog_condition`, `add_equation(s)`, `add_endog_equation`, `add_constraint`, `add_hjb_equation`, `add_system`, `register_function(s)`.
- **Forward evaluation**: `update_variables`, `_eval_local_functions`, `loss_fn`, `closure`, `closure_soft_attention`, `test_step`.
- **Sampling helpers**: `sample_uniform`, `sample_fixed_grid`.
- **Validation / persistence**: `validate_model_setup`, `save_model`, `load_model`, `__str__`.

Subclasses implement `train_model` and `eval_model` (which differ fundamentally between the stationary and time-stepping schemes), plus their own sampling/validation helpers.

## Overriding the forward computation

All forward evaluation of agents/endogenous variables flows through a single method:

```py
def update_variables(self, SV, vd=None):
    if vd is None:
        vd = self.variable_val_dict
    for i, sv_name in enumerate(self.state_variables):
        vd[sv_name] = SV[:, i:i+1]
    vd["SV"] = SV
    self._eval_local_functions(SV, vd)   # agent/endog forward + derivatives
    for eq_name in self.equations:       # user-defined equations
        lhs = self.equations[eq_name].lhs.formula_str
        vd[lhs] = self.equations[eq_name].eval(self.custom_function_dict, vd)
```

Because training, validation, residual-based refinement and plotting all route through `update_variables`, overriding it (or just `_eval_local_functions`) in a subclass changes the forward computation everywhere consistently, without touching equations, endogenous equations, constraints or loss computation.

### _eval_local_functions
```py
def _eval_local_functions(self, SV, vd)
```
Evaluate every agent/endogenous-variable function (forward values and their derivatives) into `vd`. This is the only place the agent/endog networks are forwarded. By default it loops over `self.local_function_dict`, calling each function individually. When the `stacked` config flag is enabled, it batches all same-architecture networks into a single `vmap` call (see [Experimental: stacked evaluation](../usage.md#experimental-stacked-vmap-batched-evaluation)) and then evaluates any remaining, non-stackable functions individually. Equations and loss computation are untouched.
