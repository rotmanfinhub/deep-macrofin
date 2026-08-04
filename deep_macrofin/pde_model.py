import atexit
import gc
import json
import os
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from torch import nn

from .base_pde_model import BasePDEModel
from .evaluations import *
from .event_handler import *
from .models import *
from .utils import *


class PDEModel(BasePDEModel):
    '''
    PDEModel class to assign variables, equations & constraints, etc.

    Also initialize the neural network architectures for each agent/endogenous variables 
    with some config dictionary.

    This is the stationary (non-time-stepping) solver. Shared functionality lives
    in :class:`BasePDEModel`.
    '''
    
    '''
    Methods to initialize the model, define variables, and constraints
    '''
    def __init__(self, name: str, 
                 config: Dict[str, Any] = DEFAULT_CONFIG, 
                 latex_var_mapping: Dict[str, str] = {}):
        '''
        Initialize a model with the provided name and config. 
        The config should include the basic training configs, 

        The optimizer is default to AdamW

        DEFAULT_CONFIG={
            "batch_size": 100,
            "num_epochs": 1000,
            "lr": 1e-3,
            "loss_log_interval": 50,
            "optimizer_type": OptimizerType.AdamW,
            "sampling_method": SamplingMethod.UniformRandom,
            "refinement_rounds": 5,
            "loss_balancing": False,
            "bernoulli_prob": 0.9999,
            "loss_balancing_temp": 0.1,
            "loss_balancing_alpha": 0.999,
            "soft_adapt_interval": -1,
            "loss_soft_attention": False,
            "stacked": False,
        }

        loss_log_interval: the interval at which loss should be reported/recorded, this is the time when validation set is used.

        latex_var_mapping should include all possible latex to python name conversions. Otherwise latex parsing will fail. Can be omitted if all the input equations/formula are not in latex form. For details, check `Formula` class defined in `evaluations/formula.py`
        '''
        self.name = name
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.batch_size = self.config.get("batch_size", 100)
        self.num_epochs = self.config.get("num_epochs", 1000)
        self.lr = self.config.get("lr", 1e-3)
        self.loss_log_interval = self.config.get("loss_log_interval", 50)
        self.optimizer_type = self.config.get("optimizer_type", OptimizerType.AdamW)
        self.sampling_method = self.config.get("sampling_method", SamplingMethod.UniformRandom)

        self.SAMPLING_METHOD_MAP = {
            SamplingMethod.UniformRandom: self.sample_uniform,
            SamplingMethod.FixedGrid: self.sample_fixed_grid,
            SamplingMethod.ActiveLearning: self.sample_rar_greedy,
            SamplingMethod.RARG: self.sample_rar_greedy,
            SamplingMethod.RARD: self.sample_rar_distribution,
        }
        self.sample = self.SAMPLING_METHOD_MAP[self.sampling_method]

        self.latex_var_mapping = latex_var_mapping

        self._init_common()

        self.OnTrainingStart = EventHandler()
        self.OnTrainingStep = EventHandler()

    def set_config(self, config: Dict[str, Any] = DEFAULT_CONFIG):
        '''
        This function overwrites the existing configurations.

        DEFAULT_CONFIG={
            "batch_size": 100,
            "num_epochs": 1000,
            "lr": 1e-3,
            "loss_log_interval": 100,
            "optimizer_type": OptimizerType.AdamW,
            "sampling_method": SamplingMethod.UniformRandom,
            "refinement_rounds": 5,
            "loss_balancing": False,
            "bernoulli_prob": 0.9999,
            "loss_balancing_temp": 0.1,
            "loss_balancing_alpha": 0.999,
            "soft_adapt_interval": -1,
            "loss_soft_attention": False,
            "stacked": False,
        }
        '''
        self.config.update(config)
        self.batch_size = self.config.get("batch_size", 100)
        self.num_epochs = self.config.get("num_epochs", 1000)
        self.lr = self.config.get("lr", 1e-3)
        self.loss_log_interval = self.config.get("loss_log_interval", 100)
        self.optimizer_type = self.config.get("optimizer_type", OptimizerType.AdamW)
        self.sampling_method = self.config.get("sampling_method", SamplingMethod.UniformRandom)
        self.sample = self.SAMPLING_METHOD_MAP[self.sampling_method]
        self.refinement_rounds: int = self.config.get("refinement_rounds", 5)
        self.stacked = self.config.get("stacked", False)

    def __get_refinement_loss_dict(self, epoch):
        '''
        Sample a dense subset of the problem domain, compute the loss and return total loss for each point sampled. Used for Residual-based Adaptive Refinement and Active Learning

        Returns:
            {
                "SV": sampled state variables, shape (10000, len(self.state_variables))
                "loss": total loss computed at each sv, shape (10000, 1)
            }
        '''
        # because we need a set of dense points to compute residual for adaptive sampling
        # we set all models to evaluation models so that gradients won't be computed.
        # it speeds up the computation and reduces memory usages
        self.set_all_model_eval()

        # Temporarily set a large batch size
        self.batch_size = 1000
        SV = self.sample_uniform(epoch)
        SV.requires_grad_(True)
        # make a copy of variable value mapping
        # so that we don't break the top level training routine
        variable_val_dict_ = self.variable_val_dict.copy()
        total_loss = torch.zeros((self.batch_size, 1), device=self.device)

        # forward pass (agent/endog + derivatives + equations) through the single
        # overridable evaluation path
        self.update_variables(SV, vd=variable_val_dict_)

        # compute total losses, without reducing to a single value, keep the original dimension, but summing up using abs values
        # Note that the conditions (IC/BC, or user pre-defined sampling regions) are not considered
        # Systems are not considered
        for label in self.endog_equations:
            total_loss += torch.abs(self.endog_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)).reshape((-1, 1))

        for label in self.constraints:
            total_loss += torch.abs(self.constraints[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)).reshape((-1, 1))

        for label in self.hjb_equations:
            total_loss += torch.abs(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)).reshape((-1, 1))

        for label in self.systems:
            total_loss += torch.abs(self.systems[label].eval_no_loss(self.custom_function_dict, variable_val_dict_, self.batch_size)).reshape((-1, 1))

        self.batch_size = self.config.get("batch_size", 100) # reset the batch size for normal computation
        self.set_all_model_training() # reset the model for training stage

        return {
            "SV": SV.detach(),
            "loss": total_loss,
        }
        
    def sample_rar_greedy(self, epoch):
        if epoch % (self.num_epochs // self.refinement_rounds) == 0 and epoch > 0:
            # check epoch > 0 so we don't need to resample in the first epoch
            refinement_loss_dict = self.__get_refinement_loss_dict(epoch)
            SV = refinement_loss_dict["SV"]
            all_losses = refinement_loss_dict["loss"]
            X_ids = torch.topk(all_losses, self.batch_size//self.refinement_rounds, dim=0)[1].squeeze(-1)
            self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))
        sv = self.sample_uniform(epoch)
        return torch.vstack((sv, self.anchor_points))
        

    def sample_rar_distribution(self, epoch):
        if epoch % (self.num_epochs // self.refinement_rounds) == 0 and epoch > 0:
            # check epoch > 0 so we don't need to resample in the first epoch
            refinement_loss_dict = self.__get_refinement_loss_dict(epoch)
            SV = refinement_loss_dict["SV"]
            all_losses = refinement_loss_dict["loss"] ** 2
            err_eq = all_losses / all_losses.mean()
            err_eq_normalized = (err_eq / err_eq.sum())[:, 0]
            X_ids = np.random.choice(a=SV.shape[0], size=self.batch_size//self.refinement_rounds, replace=False, p=err_eq_normalized.detach().cpu().numpy())
            self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))
        fixed_grid_sv = self.sample_uniform(epoch)
        return torch.vstack((fixed_grid_sv, self.anchor_points))
    
    def init_loss_balancing(self, *args, **kwargs):
        '''
        Initialize variables for relative loss balancing with random lookback
        https://arxiv.org/pdf/2110.09813
        '''
        self.loss_weight_log_dict = defaultdict(list)
        self.bernoulli_rho = torch.distributions.Bernoulli(self.config.get("bernoulli_prob", 0.9999))
        self.temp = self.config.get("loss_balancing_temp", 0.1)
        self.alpha = self.config.get("loss_balancing_alpha", 0.999)
        self.init_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)
        self.prev_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)

    def loss_balancing_step(self, *args, **kwargs):
        epoch = kwargs.get("epoch", 0)
        if epoch == 0:
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                self.init_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
                self.prev_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
        elif epoch % self.loss_log_interval == 0:
            # relative loss balancing with random lookback
            # https://arxiv.org/pdf/2110.09813
            curr_loss_tensor = torch.zeros_like(self.prev_loss_tensor, device=self.device)
            prev_loss_weight_tensor = torch.zeros_like(self.prev_loss_tensor, device=self.device)
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                curr_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
                prev_loss_weight_tensor[i] = self.loss_weight_dict[loss_label]
            
            ratio_prev = curr_loss_tensor / (self.temp * self.prev_loss_tensor)
            ratio_zero = curr_loss_tensor / (self.temp * self.init_loss_tensor)
            bal_prev = len(self.loss_val_dict) * torch.nn.functional.softmax(ratio_prev, dim=-1)
            bal_zero = len(self.loss_val_dict) * torch.nn.functional.softmax(ratio_zero, dim=-1)
            rho = self.bernoulli_rho.sample()
            weight_hist = rho * prev_loss_weight_tensor + (1 - rho) * bal_zero
            new_weight = self.alpha * weight_hist + (1 - self.alpha) * bal_prev
            for i, k in enumerate(self.loss_weight_dict):
                self.loss_weight_dict[k] = new_weight[i].item()
                
            self.loss_weight_log_dict["epoch"].append(epoch)
            for k, v in self.loss_weight_dict.items():
                self.loss_weight_log_dict[k].append(v)
            
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                self.prev_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss).item()

    def init_soft_adapt(self, *args, **kwargs):
        '''
        Initialize variables for soft adapt
        https://arxiv.org/pdf/1912.12355
        '''
        self.loss_adaption_interval = self.config.get("soft_adapt_interval", -1)
        self.beta = self.config.get("loss_balancing_temp", 0.1)
        self.loss_weight_log_dict = defaultdict(list)
        self.loss_hist = {}
        self.loss_hist_count = 0
        for label in self.loss_val_dict:
            self.loss_hist[label] = torch.zeros(self.loss_adaption_interval, device=self.device)

    def soft_adapt_step(self, *args, **kwargs):
        epoch = kwargs.get("epoch", 0)
        for loss_label, loss in self.loss_val_dict.items():
            self.loss_hist[loss_label][self.loss_hist_count] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
        self.loss_hist_count += 1
        if self.loss_hist_count == self.loss_adaption_interval:
            # soft adapt implementation following https://arxiv.org/pdf/1912.12355
            # Loss Weighted + normalized
            # the idea is to assign higher weight to loss functions that decreases more slowly.
            self.loss_hist_count = 0
            rates_of_change = torch.zeros(len(self.loss_hist))
            avg_loss_values = torch.zeros(len(self.loss_hist))
            for i, label in enumerate(self.loss_hist):
                # use the mean rate of change, instead of using finite difference for faster computation
                diff = self.loss_hist[label][1:] - self.loss_hist[label][:-1]
                rates_of_change[i] = torch.mean(diff)
                avg_loss_values[i] = torch.mean(self.loss_hist[label])
            # normalization
            rates_of_change = rates_of_change / torch.sum(torch.abs(rates_of_change))
            # loss weighted softmax
            new_weights = torch.nn.functional.softmax(self.beta * (rates_of_change - rates_of_change.max()), dim=-1)
            new_weights = avg_loss_values * new_weights / torch.sum(avg_loss_values * new_weights)
            
            for i, label in enumerate(self.loss_hist):
                self.loss_weight_dict[label] = new_weights[i].item()
                self.loss_hist[label] = torch.zeros(self.loss_adaption_interval)
            
            self.loss_weight_log_dict["epoch"].append(epoch)
            for k, v in self.loss_weight_dict.items():
                self.loss_weight_log_dict[k].append(v)

    def init_soft_attention(self, *args, **kwargs):
        # a random sample to determine batch size
        assert self.sampling_method == SamplingMethod.FixedGrid, "Soft Attention only works for Fixed Grid sampling."
        SV = self.sample(0)
        B = SV.shape[0]
        self.B = B
        self.loss_weight_log_dict = defaultdict(list)
        all_params = []
        # start with uniform weights for each training point
        for k in self.loss_weight_dict:
            if k in self.agent_conditions or k in self.endog_var_conditions:
                # it's either in agent condition or in endog var condition
                cond = self.agent_conditions.get(k, self.endog_var_conditions[k])
                bs = None
                for kk, v in cond.lhs_state.items():
                    if isinstance(v, torch.Tensor):
                        bs = v.shape[0]
                        break
                if bs is None:
                    for kk, v in cond.rhs_state.items():
                        if isinstance(v, torch.Tensor):
                            bs = v.shape[0]
                            break
                curr_params = nn.Parameter(torch.ones((bs, 1), device=self.device))
            else:
                curr_params = nn.Parameter(torch.ones((B, 1), device=self.device))
            self.loss_weight_dict[k] = curr_params
            all_params += [curr_params]
        # perform gradient ascent, the weights should never go negative.
        self.optimizer.add_param_group({"params": all_params, "maximize": True})

    def soft_attention_step(self, *args, **kwargs):
        # The optimization step has been performed in the main loop, so we only need to record the weights
        epoch = kwargs.get("epoch", 0)
        SV = kwargs.get("SV")
        for i in range(SV.shape[0]):
            self.loss_weight_log_dict["epoch"].append(epoch)
            for n, sv_name in enumerate(self.state_variables):
                self.loss_weight_log_dict[sv_name].append(SV[i, n].item())
            for k, v in self.loss_weight_dict.items():
                if k in self.agent_conditions or k in self.endog_var_conditions:
                    continue
                self.loss_weight_log_dict[k].append(v[i, 0].item())

    def __compute_changes(self, SV):
        temp_dict = {}
        # forward pass (agent/endog + derivatives + equations) through the single
        # overridable evaluation path
        self.update_variables(SV, vd=temp_dict)

        new_vals = {}
        for k in self.prev_vals:
            new_vals[k] = temp_dict[k].detach()
        
        max_abs_change = 0.
        max_rel_change = 0.
        all_changes = {}
        for k in self.prev_vals:
            mean_new_val = torch.mean(new_vals[k]).item()
            abs_change = torch.mean(torch.abs(new_vals[k] - self.prev_vals[k])).item()
            rel_change = torch.mean(torch.abs((new_vals[k] - self.prev_vals[k]) / self.prev_vals[k])).item()
            all_changes[f"{k}_mean_val"] = mean_new_val
            all_changes[f"{k}_abs"] = abs_change
            all_changes[f"{k}_rel"] = rel_change
            max_abs_change = max(max_abs_change, abs_change)
            max_rel_change = max(max_rel_change, rel_change)

        # Update for next iteration
        for k in self.prev_vals:
            self.prev_vals[k] = new_vals[k]

        total_rel_change = min(max_abs_change, max_rel_change)
        all_changes["total"] = total_rel_change
        return all_changes
    
    def __validation(self, SV_CHECK: torch.Tensor):
        self.set_all_model_eval()
        temp = self.loss_val_dict.copy()
        SV = SV_CHECK.detach().clone()
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV

        self.update_variables(SV)
        self.loss_fn()
        loss_dict = self.loss_val_dict.copy()
        if self.config.get("loss_soft_attention", False):
            for k in loss_dict:
                loss_dict[k] = torch.mean(loss_dict[k])

        total_loss = 0
        for loss_label, loss in loss_dict.items():
            total_loss += torch.nanmean(self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss))

        loss_dict["total_loss"] = total_loss
        self.loss_val_dict = temp
        self.set_all_model_training()
        return loss_dict

    def train_model(self, model_dir: str="./", filename: str=None, full_log=False, variables_to_track: List[str]=[]):
        '''
        The entire loop of training
        '''

        if self.anchor_points is None:
            self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device)

        min_loss = torch.inf
        epoch_loss_dict = defaultdict(list)
        min_loss_dict = defaultdict(list)
        all_params = []
        
        model_has_kan = False
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
            if agent.config["layer_type"] in [LayerType.KAN, LayerType.MultKAN]:
                model_has_kan = True
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())
            if endog_var.config["layer_type"] in [LayerType.KAN, LayerType.MultKAN]:
                model_has_kan = True
        for learnable_param_name in self.learnable_params:
            self.variable_val_dict[learnable_param_name] = self.variable_val_dict[learnable_param_name].detach().to(self.device).requires_grad_(True)
            all_params += [self.variable_val_dict[learnable_param_name]]
        
        if model_has_kan:
            # KAN can only be trained with LBFGS, 
            # as long as there is one model with KAN, we must route to the default LBFGS
            self.optimizer = LBFGS(all_params, lr=self.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_type](all_params, self.lr)

        os.makedirs(model_dir, exist_ok=True)
        if filename is None:
            filename = filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_epochs}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")

        @atexit.register
        def cleanup_file():
            # make sure the log file is properly closed even after exception
            log_file.close()

        print(str(self), file=log_file, flush=True)
        try:
            self.validate_model_setup(model_dir)
        except Exception as e:
            # close the file on exception. This should be the only place for it...
            log_file.close()
            raise e
        gc.collect()
        torch.cuda.empty_cache()
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()

        if self.config.get("loss_balancing", False):
            self.OnTrainingStart += self.init_loss_balancing
            self.OnTrainingStep += self.loss_balancing_step
        elif self.config.get("loss_soft_attention", False):
            self.OnTrainingStart += self.init_soft_attention
            self.OnTrainingStep += self.soft_attention_step
            # override the closure function
            self.closure = self.closure_soft_attention
        elif self.config.get("soft_adapt_interval", -1) > 0:
            self.OnTrainingStart += self.init_soft_adapt
            self.OnTrainingStep += self.soft_adapt_step
        
        set_seeds(0)
        SV_CHECK = self.sample(0)
        change_dict = defaultdict(list)
        self.prev_vals = {}
        for agent_name in self.agents:
            self.prev_vals[agent_name] = torch.zeros_like(SV_CHECK[:, 0:1], device=self.device)
        for endog_name in self.endog_vars:
            self.prev_vals[endog_name] = torch.zeros_like(SV_CHECK[:, 0:1], device=self.device)
        for var in variables_to_track:
            if var in self.variable_val_dict and var not in self.prev_vals:
                self.prev_vals[var] =  torch.zeros_like(SV_CHECK[:, 0:1], device=self.device)

        self.OnTrainingStart()
        set_seeds(0)
        pbar = tqdm(range(self.num_epochs), dynamic_ncols=True)
        for epoch in pbar:
            epoch_start_time = time.time()
            
            SV = self.sample(epoch)
            SV.requires_grad_(True)
            for i, sv_name in enumerate(self.state_variables):
                self.variable_val_dict[sv_name] = SV[:, i:i+1]
            self.variable_val_dict["SV"] = SV

            self.optimizer.step(lambda: self.closure(SV))
            total_loss = 0
            for loss_label, loss in self.loss_val_dict.items():
                total_loss += torch.nanmean(self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss))

            loss_dict = self.loss_val_dict.copy()
            if self.config.get("loss_soft_attention", False):
                for k in loss_dict:
                    loss_dict[k] = torch.mean(loss_dict[k])
            loss_dict["total_loss"] = total_loss

            if full_log:
                formatted_train_loss = ",\n".join([f'{k}: {v:.2e}' for k, v in loss_dict.items()])
            else:
                formatted_train_loss = "%.2e" % loss_dict["total_loss"].item()

            self.OnTrainingStep(epoch=epoch, SV=SV)

            all_changes = self.__compute_changes(SV_CHECK)
            change_dict["epoch"].append(epoch)
            for k, v in all_changes.items():
                change_dict[k].append(v)

            if epoch % self.loss_log_interval == 0 or epoch == self.num_epochs - 1:
                # evaluate every log interval time and also evaluate at the end of the final epoch
                loss_dict = self.__validation(SV_CHECK)
                if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
                    # if we get a lower loss on the validation set, save as the latest best model
                    min_loss = loss_dict["total_loss"].item()
                    self.save_model(model_dir, f"{file_prefix}_best.pt")
                    min_loss_dict["epoch"].append(len(min_loss_dict["epoch"]))
                    for k, v in loss_dict.items():
                        min_loss_dict[k].append(v.item())
                    pbar.set_description("Min loss: {0:.2e}".format(min_loss))

                epoch_loss_dict["epoch"].append(epoch)
                for k, v in loss_dict.items():
                    epoch_loss_dict[k].append(v.item())
            print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
            self.save_model(model_dir, f"{file_prefix}_best.pt")
        print(f"Best model saved to {model_dir}/{file_prefix}_best.pt if valid")
        self.save_model(model_dir, filename, verbose=True)
        pd.DataFrame(epoch_loss_dict).to_csv(f"{model_dir}/{file_prefix}_loss.csv", index=False)
        pd.DataFrame(min_loss_dict).to_csv(f"{model_dir}/{file_prefix}_min_loss.csv", index=False)
        pd.DataFrame(change_dict).to_csv(f"{model_dir}/{file_prefix}_change_dict.csv", index=False)

        if self.anchor_points.shape[0] > 0:
            anchor_points_np = self.anchor_points.detach().cpu().numpy()
            np.save(f"{model_dir}/{file_prefix}_anchor_points.npy", anchor_points_np)
            print(f"Anchor points saved to {model_dir}/{file_prefix}_anchor_points.npy")

        if hasattr(self, "loss_weight_log_dict"):
            pd.DataFrame(self.loss_weight_log_dict).to_csv(f"{model_dir}/{file_prefix}_loss_weight.csv", index=False)

        return loss_dict
    
    def eval_model(self, full_log=False):
        '''
        The entire loop of evaluation
        '''
        self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device)
        for learnable_param_name in self.learnable_params:
            self.variable_val_dict[learnable_param_name] = self.variable_val_dict[learnable_param_name].detach().to(self.device)
        
        self.validate_model_setup()
        self.set_all_model_eval()
        print("{0:=^80}".format("Evaluating"))
        SV = self.sample(0)
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV
        loss_dict = self.test_step(SV)

        if full_log:
            formatted_loss = ",\n".join([f'{k}: {v:.2e}' for k, v in loss_dict.items()])
        else:
            formatted_loss = "%.2e" % loss_dict["total_loss"].item()
        print(f"loss :: {formatted_loss}")
        return loss_dict

    def plot_vars(self, vars_to_plot: List[str], ncols: int=4, elev=30, azim=-135, roll=0):
        '''
        Inputs:
        - vars_to_plot: variable names to plot, can be an equation defining a new variable. If Latex, need to be enclosed by $$ symbols
        - ncols: number of columns to plot, default: 4
        - elev, azim, roll: view angles for 3D plots.  https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        This function is only supported for 1D or 2D state_variables.
        '''
        assert len(self.state_variables) <= 2, "Plot is only supported for problems with no more than 2 state variables"

        variable_var_dict_ = self.variable_val_dict.copy()
        var_to_latex = {}
        for k, v in self.latex_var_mapping.items():
            var_to_latex[v] = k

        sv_ls = [0] * len(self.state_variables)
        for i in range(len(self.state_variables)):
            sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                      self.state_variable_constraints["sv_high"][i], steps=100, device=self.device)
        X = torch.cartesian_prod(*sv_ls)
        
        nrows = len(vars_to_plot) // ncols
        if len(vars_to_plot) % ncols > 0:
            nrows += 1
        if len(self.state_variables) == 1:
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
            SV = X.unsqueeze(-1)
            SV.requires_grad_(True)
            X = X.detach().cpu().numpy().reshape(-1)
            # forward pass (agent/endog + derivatives + equations) through the single
            # overridable evaluation path
            self.update_variables(SV, vd=variable_var_dict_)

            sv_text = self.state_variables[0]
            if self.state_variables[0] in var_to_latex:
                sv_text = f"${var_to_latex[self.state_variables[0]]}$"

            for i, curr_var in enumerate(vars_to_plot):
                curr_row = i // ncols
                curr_col = i % ncols
                if nrows == 1:
                    if ncols == 1:
                        curr_ax = ax
                    else:
                        curr_ax = ax[curr_col]
                else:
                    curr_ax = ax[curr_row][curr_col]
                if "$" in curr_var:
                    # parse latex and potentially equation
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
                        curr_ax.plot(X, variable_var_dict_[lhs].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        lhs_unparsed = curr_var.split("=")[0].replace("$", "").strip()
                        curr_ax.set_ylabel(f"${lhs_unparsed}$")
                        curr_ax.set_title(f"${lhs_unparsed}$ vs {sv_text}")
                    else:
                        base_var = curr_var.replace("$", "").strip()
                        base_var_non_latex = self.latex_var_mapping.get(base_var, base_var)
                        curr_ax.plot(X, variable_var_dict_[base_var_non_latex].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs {sv_text}")
                else:
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
                        curr_ax.plot(X, variable_var_dict_[lhs].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(lhs)
                        curr_ax.set_title(f"{lhs} vs {sv_text}")
                    else:
                        curr_ax.plot(X, variable_var_dict_[curr_var].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs {sv_text}")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), subplot_kw={"projection": "3d"})
            SV = torch.clone(X)
            X, Y = torch.meshgrid(sv_ls, indexing="ij")
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            SV.requires_grad_(True)
            # forward pass (agent/endog + derivatives + equations) through the single
            # overridable evaluation path
            self.update_variables(SV, vd=variable_var_dict_)

            sv_text0 = self.state_variables[0]
            sv_text1 = self.state_variables[1]
            if self.state_variables[0] in var_to_latex:
                sv_text0 = f"${var_to_latex[self.state_variables[0]]}$"
            if self.state_variables[1] in var_to_latex:
                sv_text1 = f"${var_to_latex[self.state_variables[1]]}$"

            for i, curr_var in enumerate(vars_to_plot):
                curr_row = i // ncols
                curr_col = i % ncols
                if nrows == 1:
                    if ncols == 1:
                        curr_ax = ax
                    else:
                        curr_ax = ax[curr_col]
                else:
                    curr_ax = ax[curr_row][curr_col]
                if "$" in curr_var:
                    # parse latex and potentially equation
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[lhs].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        lhs_unparsed = curr_var.split("=")[0].replace("$", "").strip()
                        curr_ax.set_zlabel(f"${lhs_unparsed}$")
                        curr_ax.set_title(f"${lhs_unparsed}$ vs ({sv_text0}, {sv_text1})")
                    else:
                        base_var = curr_var.replace("$", "").strip()
                        base_var_non_latex = self.latex_var_mapping.get(base_var, base_var)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[base_var_non_latex].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs ({sv_text0}, {sv_text1})")
                else:
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[lhs].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{lhs} vs ({sv_text0}, {sv_text1})")
                    else:
                        curr_ax.plot_surface(X, Y, variable_var_dict_[curr_var].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs ({sv_text0}, {sv_text1})")
                curr_ax.view_init(elev, azim, roll)
                curr_ax.set_box_aspect(None, zoom=0.85)
            plt.tight_layout()
            plt.show()
