'''
Experimental: vmap-batched ("stacked") evaluation of agents/endogenous variables.

The library normally evaluates every agent/endogenous-variable network (and its
derivatives) one at a time through ``local_function_dict``. When many networks
share the same architecture, this is wasteful: each network does its own forward
and autograd pass. This module batches all same-architecture networks into a
single ``torch.func.vmap`` call across both the network dimension and the batch
dimension, computing values, jacobians and hessians at once.

It is confined to the ``local_function_dict`` computation: it produces exactly the
same variable/derivative keys the per-network path would (``name``, ``name_Jac``,
``name_Hess``), so equations, endogenous equations, constraints, losses, etc. are
unaffected.

Scope / limitations (experimental):
    - Stacked evaluation ONLY supports networks configured with
      ``batch_jac_hes=True``. Such networks expose ``name_Jac`` / ``name_Hess``
      (the full jacobian/hessian). A stackable network (MLP, no
      ``hardcode_function``) configured with ``batch_jac_hes=False`` raises a
      ``ValueError`` -- set ``batch_jac_hes=True`` to use stacked evaluation.
    - Only ``LayerType.MLP`` networks are stacked. Other layer types (KAN,
      MultKAN, DeepSet, DGM, ResNet) and any variable with a ``hardcode_function``
      fall back to the per-network evaluation automatically.
    - Networks are grouped by architecture signature, which INCLUDES the final
      elementwise activation. Thus a set of otherwise-identical networks that only
      differ by, e.g., a final ``Softplus`` are split into (at most) a couple of
      groups, each evaluated with one vmap call -- correctness is preserved while
      still avoiding per-network loops.
'''

from typing import Dict, List

import torch
from torch.func import functional_call, hessian, jacrev, vmap

from .model_utils import LayerType


def _derivative_order(var):
    cfg = var.config
    return int(cfg.get("derivative_order", cfg.get("input_size", len(var.state_variables))))

def _stack_module_state(modules):
    '''
    Stack the parameters and buffers of a list of identically-structured modules.

    The stacked parameters remain connected (via ``torch.stack``) to the original
    leaf parameters, so gradients flow back to each underlying network during
    training.
    '''
    all_params = [dict(m.named_parameters()) for m in modules]
    all_buffers = [dict(m.named_buffers()) for m in modules]
    params = {k: torch.stack([p[k] for p in all_params]) for k in all_params[0]}
    if all_buffers and all_buffers[0]:
        buffers = {k: torch.stack([b[k] for b in all_buffers]) for k in all_buffers[0]}
    else:
        buffers = {}
    return params, buffers


def _signature(var):
    '''
    Architecture signature used to decide which variables can share a single
    stacked (vmap) call. Includes the final activation flags so networks that
    differ only by a final Softplus/Sigmoid are grouped separately (a single
    ``functional_call`` template bakes in one activation graph).
    '''
    cfg = var.config
    return (
        cfg.get("layer_type"),
        tuple(cfg.get("hidden_units", [])),
        cfg.get("activation_type"),
        cfg.get("input_size"),
        cfg.get("output_size", 1),
        bool(cfg.get("positive", False)),
        bool(cfg.get("sigmoid", False)),
        _derivative_order(var),
        tuple(var.sv_subset_idx),
    )


def _can_stack(var):
    '''
    Whether ``var`` participates in stacked evaluation.

    Non-MLP layers and hardcoded functions are not stackable and fall back to the
    per-network path (returns ``False``). A stackable MLP MUST use
    ``batch_jac_hes=True``; otherwise a ``ValueError`` is raised.
    '''
    cfg = var.config
    if "hardcode_function" in cfg:
        return False
    if cfg.get("layer_type") != LayerType.MLP:
        return False
    if _derivative_order(var) > 2:
        # only up to 2nd order derivatives are available from a single hessian
        return False
    if not cfg.get("batch_jac_hes", False) and _derivative_order(var) > 0:
        raise ValueError(
            f"Stacked evaluation requires batch_jac_hes=True or derivative_order=0, but variable "
            f"'{var.name}' has batch_jac_hes=False and requires derivative computation."
            f"Set batch_jac_hes=True on this variable to use stacked evaluation."
        )
    return True


class StackedNetworkGroup:
    '''
    A group of same-architecture ``LearnableVar`` objects evaluated together with
    a single ``vmap`` call across networks and batch.

    ``fill(SV, vd)`` writes the value and derivative keys (``name``, ``name_Jac``,
    ``name_Hess``) of every member into the value dictionary ``vd``, matching
    exactly what the per-network ``batch_jac_hes`` path produces.
    '''

    def __init__(self, variables):
        if len(variables) == 0:
            raise ValueError("StackedNetworkGroup needs at least one variable.")
        self.variables = list(variables)
        template_var = self.variables[0]
        self.template = template_var.model
        self.sv_subset_idx = list(template_var.sv_subset_idx)
        self.output_size = template_var.config.get("output_size", 1)
        self.derivative_order = _derivative_order(template_var)
        self.device = template_var.device
        self.names = [v.name for v in self.variables]

    def covered_keys(self):
        out = set()
        for v in self.variables:
            out.add(v.name)
            out.add(v.name + "_Jac")
            out.add(v.name + "_Hess")
        return out

    def _compute(self, SV):
        params, buffers = _stack_module_state([v.model for v in self.variables])
        template = self.template
        x_all = SV[..., self.sv_subset_idx].to(self.device)

        def fwd(p, b, x):
            return functional_call(template, (p, b), (x,))          # (O,)

        def val_net(p, b):
            return vmap(lambda x: fwd(p, b, x))(x_all)
        val = vmap(val_net)(params, buffers).transpose(0, 1).contiguous()   # (B,N,O)

        jac = None
        hess = None

        if self.derivative_order >= 1:
            def jac_net(p, b):
                return vmap(jacrev(lambda x: fwd(p, b, x)))(x_all)
            jac = vmap(jac_net)(params, buffers).transpose(0, 1).contiguous()   # (B,N,O,D)

        if self.derivative_order >= 2:
            def hess_net(p, b):
                return vmap(hessian(lambda x: fwd(p, b, x)))(x_all)
            hess = vmap(hess_net)(params, buffers).transpose(0, 1).contiguous()  # (B,N,O,D,D)
        return val, jac, hess

    def fill(self, SV, vd):
        val, jac, hess = self._compute(SV)
        for i, v in enumerate(self.variables):
            name = v.name
            vd[name] = val[:, i]              # (B,O)
            if jac is not None:
                vd[name + "_Jac"] = jac[:, i]     # (B,O,D)
            if hess is not None:
                vd[name + "_Hess"] = hess[:, i]   # (B,O,D,D)


class StackedFunctionEvaluator:
    '''
    Partitions all agents and endogenous variables into architecture-signature
    groups and evaluates each stackable group with one vmap call.

    Non-stackable variables (hardcoded functions, non-MLP layers) are left out and
    handled by the caller's per-network fallback. Stackable MLPs must use
    ``batch_jac_hes=True`` (see ``_can_stack``).
    '''

    def __init__(self, agents: Dict, endog_vars: Dict):
        variables = list(agents.values()) + list(endog_vars.values())
        buckets = {}
        for v in variables:
            if not _can_stack(v):
                continue
            buckets.setdefault(_signature(v), []).append(v)

        self.groups: List[StackedNetworkGroup] = []
        self._covered = set()
        for members in buckets.values():
            group = StackedNetworkGroup(members)
            self.groups.append(group)
            self._covered |= group.covered_keys()

    def covered_keys(self):
        return self._covered

    def fill(self, SV, vd):
        '''
        Evaluate all stackable groups into ``vd`` and return the set of keys that
        were populated (so the caller can skip them in the per-network loop).
        '''
        for group in self.groups:
            group.fill(SV, vd)
        return self._covered
