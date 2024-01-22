import numpy as np
import pickle
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple


def use_experts(layer_idx):
    return True


def process_ffn(model):
    if model.config.model_type == "bert":
        inner_model = model.bert
    else:
        raise ValueError("Model type not recognized.")

    for i in range(model.config.num_hidden_layers):
        model_layer = inner_model.encoder.layer[i]
        if model_layer.use_experts:
            if model.config.moebert == "diffmoe":
                model_layer.importance_processor.load_diffexperts(
                    diff_model_layer=model_layer,
                    shared_size=model.config.moebert_share_importance,
                )
            else :
                model_layer.importance_processor.load_experts(model_layer)


class ImportanceProcessor:
    def __init__(self, config, layer_idx, num_local_experts, local_group_rank):
        self.num_experts = config.moebert_expert_num  # total number of experts
        self.num_local_experts = num_local_experts  # number of experts on this device
        self.local_group_rank = local_group_rank  # rank in the current process group
        self.intermediate_size = config.moebert_expert_dim  # FFN hidden dimension
        self.share_importance = config.moebert_share_importance  # number of shared FFN dimension

        importance = ImportanceProcessor.load_importance_single(config.moebert_load_importance)[layer_idx, :]
        self.importance = self._split_importance(importance)

        self.is_moe = False  # safety check

    @staticmethod
    def load_importance_single(importance_files):
        with open(importance_files, "rb") as file:
            data = pickle.load(file)
            data = data["idx"]
        return np.array(data)

    def _split_importance(self, arr):
        result = []
        top_importance = arr[:self.share_importance]
        remain = arr[self.share_importance:]
        all_experts_remain = []
        # reorder the remaining rows in the order of assigning to each expert
        for i in range(self.num_experts):
            all_experts_remain.append(remain[i::self.num_experts])
        all_experts_remain = np.array(all_experts_remain)

        for i in range(self.num_local_experts):
            temp = all_experts_remain[self.num_local_experts * self.local_group_rank + i]
            temp = np.concatenate((top_importance, temp))
            temp = temp[:self.intermediate_size]
            result.append(temp.copy())
        # Array of indices for each expert
        result = np.array(result)
        return result

    def load_experts(self, model_layer):
        expert_list = model_layer.experts.experts
        fc1_weight_data = model_layer.intermediate.dense.weight.data
        fc1_bias_data = model_layer.intermediate.dense.bias.data
        fc2_weight_data = model_layer.output.dense.weight.data
        fc2_bias_data = model_layer.output.dense.bias.data
        layernorm_weight_data = model_layer.output.LayerNorm.weight.data
        layernorm_bias_data = model_layer.output.LayerNorm.bias.data
        for i in range(self.num_local_experts):
            idx = self.importance[i]

            # Modified
            # Maintain weight data shape. Overwrite data in specific indices.
            # Keep rest in initialized state.
            expert_list[i].fc1.weight.data[:idx.shape[0],:] = fc1_weight_data[idx, :].clone()
            expert_list[i].fc1.bias.data[:idx.shape[0]]     = fc1_bias_data[idx].clone()
            expert_list[i].fc2.weight.data[:,:idx.shape[0]] = fc2_weight_data[:, idx].clone()
            expert_list[i].fc2.bias.data                    = fc2_bias_data.clone()
            expert_list[i].LayerNorm.weight.data            = layernorm_weight_data.clone()
            expert_list[i].LayerNorm.bias.data              = layernorm_bias_data.clone()

        del model_layer.intermediate
        del model_layer.output
        self.is_moe = True

    def load_diffexperts(self, diff_model_layer, shared_size):
        expert_list             = diff_model_layer.experts.experts
        fc1_weight_data         = diff_model_layer.intermediate.dense.weight.data
        fc1_bias_data           = diff_model_layer.intermediate.dense.bias.data
        fc2_weight_data         = diff_model_layer.output.dense.weight.data
        fc2_bias_data           = diff_model_layer.output.dense.bias.data
        layernorm_weight_data   = diff_model_layer.output.LayerNorm.weight.data
        layernorm_bias_data     = diff_model_layer.output.LayerNorm.bias.data

        for i in range(self.num_local_experts):
            idx = self.importance[i]

            shared_idx = idx[:shared_size]
            unique_idx = idx[shared_size:]
            
            # shared important weights
            expert_list[i].fc1_shared.weight.data   = fc1_weight_data[shared_idx, :].clone()
            expert_list[i].fc1_shared.bias.data     = fc1_bias_data[shared_idx].clone()
            expert_list[i].fc2_shared.weight.data   = fc2_weight_data[:, shared_idx].clone()
            expert_list[i].fc2_shared.bias.data     = fc2_bias_data.clone()

            # unique weights

            # Modified
            # Maintain weight data shape. Overwrite data in specific indices.
            # Keep rest in initialized state.
            expert_list[i].fc1_unique.weight.data[:unique_idx.shape[0],:]  = fc1_weight_data[unique_idx,:].clone()
            expert_list[i].fc1_unique.bias.data[:unique_idx.shape[0]]       = fc1_bias_data[unique_idx].clone()
            expert_list[i].fc2_unique.weight.data[:,:unique_idx.shape[0]]    = fc2_weight_data[:,unique_idx].clone()
            expert_list[i].fc2_unique.bias.data                             = fc2_bias_data.clone()

            expert_list[i].LayerNorm.weight.data = layernorm_weight_data.clone()
            expert_list[i].LayerNorm.bias.data = layernorm_bias_data.clone()

            ###########################################
            # initiate diff parametrization
            expert_list[i]._add_diff_parametrizations()
            ###########################################

        del diff_model_layer.intermediate
        del diff_model_layer.output
        self.is_moe = True


class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size, dropout):
        nn.Module.__init__(self)

        # first layer
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # second layer
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor):
        input_tensor = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


from torch.nn.parameter import Parameter
import torch.nn.utils.parametrize as parametrize
from typing import Union, List, Tuple, Callable, Dict, Optional

# Referenced from https://github.com/sirluk/sparse_transformers/blob/main/src/utils.py
def concrete_stretched(
    alpha: torch.Tensor,
    l: Union[float, int] = -1.5,
    r: Union[float, int] = 1.5,
    deterministic: bool = False
) -> torch.Tensor:
    if not deterministic:
        u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
        u_term = u.log() - (1-u).log()
    else:
        u_term = 0.
    s = (torch.sigmoid(u_term + alpha))
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z

# Referenced from https://github.com/sirluk/sparse_transformers/blob/main/src/models/weight_parametrizations.py
class DiffWeightFinetune(nn.Module):

    def __init__(
        self,
        weight: nn.Parameter,
        alpha_init: float,
        concrete_lower: float,
        concrete_upper: float,
        structured: bool,
        shared_size: int, # shared dimension size
        ffn_idx: int, # determine structured pruning axis (ffn1:0, ffn2:1)
    ):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured
        self.shared_size = shared_size
        self.ffn_idx = ffn_idx

        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))

        if structured:
            # ffn1
            if ffn_idx == 0:
                alpha_group_shape = (self.shared_size, 1)
            # ffn2
            elif ffn_idx == 1:
                alpha_group_shape = (1, self.shared_size)
            else :
                assert 0, "Undefined alpha_group axis {}".format(ffn_idx)
            
            self.register_parameter("alpha_group", Parameter(torch.zeros(alpha_group_shape, device=weight.device) + alpha_init))

        self.active = True

    def forward(self, X):
        if not self.active: return X
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.diff_weight(X)

    def diff_weight(self, X):
        return self.z * (self.finetune - X)

    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z

    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, alpha) -> torch.Tensor:
        return concrete_stretched(
            alpha,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )

    def set_frozen(self, frozen: bool) -> None:
        self.finetune.requires_grad = not frozen
        self.alpha.requires_grad = not frozen
        if self.structured:
            self.alpha_group.requires_grad = not frozen
        if frozen:
            self.eval()
        else:
            self.train()

# Referenced from https://github.com/sirluk/sparse_transformers/blob/main/src/models/weight_parametrizations.py
class DiffWeightFixmask(nn.Module):

    def __init__(self, diff_weight: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("diff_weight", Parameter(diff_weight * mask))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
        self.active = True

    def forward(self, X):
        if not self.active: return X
        return X + self.mask * self.diff_weight

    def set_frozen(self, frozen: bool) -> None:
        self.diff_weight.requires_grad = not frozen

# Define Diff Expert FFN
# Adapted from https://github.com/sirluk/sparse_transformers/blob/main/src/models/model_base.py
class DiffFeedForward(nn.Module):
    # shared_size : size of the shared expert dimension
    def __init__(
        self,
        config,
        intermediate_size,
        shared_size,
        dropout=0.1,
        fixmask_init=False,
        alpha_init=5.0,
        concrete_lower=-1.5,
        concrete_upper=1.5,
        structured=True,
        sparsity_pen=1.25e-7,
    ):
        nn.Module.__init__(self)

        # config
        self.intermediate_size  = intermediate_size
        self.shared_size        = shared_size
        self.unique_size        = intermediate_size - shared_size
        self.dropout            = dropout

        # DiffPruning
        self.fixmask_init       = fixmask_init
        self.alpha_init         = alpha_init
        self.concrete_lower     = concrete_lower
        self.concrete_upper     = concrete_upper
        self.structured         = structured
        self.sparsity_pen       = sparsity_pen

        assert intermediate_size >= shared_size, \
            "Shared size {} is greater than intermediate size {}".format(
                                                                        shared_size,
                                                                        intermediate_size,
                                                                    )

        # shared important layers
        self.fc1_shared = nn.Linear(config.hidden_size, self.shared_size)
        self.fc2_shared = nn.Linear(self.shared_size, config.hidden_size)

        # expert-unique layers
        self.fc1_unique = nn.Linear(config.hidden_size, self.unique_size)
        self.fc2_unique = nn.Linear(self.unique_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.model_state = "FINETUNING" # (FINETUNING, FIXMASK, TASK)

    def _add_diff_parametrizations(self):
        # Apply diff parametrization to shared submatrices
        for i, base_module in enumerate([self.fc1_shared, self.fc2_shared]):
            for n,p in list(base_module.named_parameters()):

                # skip bias term
                if "bias" in n:
                    continue

                # freeze the shared important submatrices
                p.requires_grad = False

                # parametrize the shared important submatrices
                if self.fixmask_init:
                    self.model_state = "FIXMASK"
                    parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                            torch.zeros_like(p), torch.ones_like(p, dtype=bool)
                        )
                    )
                else:
                    self.model_state = "FINETUNING"
                    parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p,
                                                                                            self.alpha_init,
                                                                                            self.concrete_lower,
                                                                                            self.concrete_upper,
                                                                                            self.structured,
                                                                                            self.shared_size,
                                                                                            i
                                                                                            ))

    @property
    def parametrized(self) -> bool:
        return (self.model_state == "FINETUNING" or self.model_state == "FIXMASK")

    @property
    def fixmask_state(self) -> bool:
        return self.model_state == "FIXMASK"

    @property
    def finetune_state(self) -> bool:
        return self.model_state == "FINETUNING"

    @property
    def n_parametrizations(self) -> int:
        return len(list(self.get_base_modules()[0].parametrizations.values())[0])

    def get_log_ratio(self) -> int:
        import math
        return 0 if (self.concrete_lower == 0) else math.log(-self.concrete_lower / self.concrete_upper)

    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()

    def get_base_modules(self, return_names: bool = False):
        if self.parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.named_modules() if check_fn(m)]

    def _get_sparsity_loss(self, idx: int = 0) -> torch.Tensor:

        l0_pen = 0.
        
        # Return only for stage-1 finetuning
        if self.finetune_state:
            for base_module in self.get_base_modules():
                module_pen = 0.
                for n, par_list in list(base_module.parametrizations.items()):
                    for a in par_list[idx].alpha_weights:
                        module_pen += self.get_l0_norm_term(a, self.get_log_ratio())
                l0_pen += (module_pen * self.sparsity_pen)

        return l0_pen

    def _parametrizations_fn(self, fn: Callable, idx: Optional[int] = None):
        for base_module in list(self.get_base_modules()):
            try:
                for par_list in base_module.parametrizations.values():
                    if idx is not None:
                        try:
                            fn(par_list[idx])
                        except IndexError:
                            pass
                    else:
                        for par in par_list:
                            fn(par)
            except AttributeError:
                pass
    
    def _freeze_parametrizations(self, frozen: bool, idx: Optional[int] = None):
        fn = lambda x: x.set_frozen(frozen)
        self._parametrizations_fn(fn, idx)

    def _remove_parametrizations(self, leave_parametrized: bool = True) -> None:
        self._freeze_parametrizations(True)
        self.model_state = "TASK"
        for module in list(self.get_base_modules()):
            try:
                for n in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, n, leave_parametrized=leave_parametrized)
            except AttributeError:
                pass

    @torch.no_grad()
    def _finetune_to_fixmask(
        self,
        pct: Optional[float] = None,
        sequential: Union[bool, list, tuple] = True,
        merged_cutoff: bool = False,
        merged_min_pct: float = 0.01
    ) -> None:

        if isinstance(sequential, (list, tuple)):
            assert len(sequential) == self.n_parametrizations, "if sequential is list, needs to equal self.n_parametrizations"
        else:
            sequential = [sequential] * self.n_parametrizations

        def _get_cutoff(values, pct, abs = True):
            k = int(round(len(values) * pct, 0))
            if abs: values = torch.abs(values)
            return torch.topk(values, k+1, largest=True, sorted=True)[0][-1]

        assert self.model_state == "FINETUNING", "model needs to be in finetuning state, currently {}".format(self.model_state)

        with self.deterministic():

            if pct is not None:

                # Find absolute value of all weight diff vectors
                diff_weights_abs = [torch.tensor([])] * self.n_parametrizations
                for base_module in list(self.get_base_modules()):
                    for n, par_list in list(base_module.parametrizations.items()):
                        w = par_list.original.detach()
                        for idx, seq in enumerate(sequential):
                            diff_weight = par_list[idx].diff_weight(w)
                            diff_weights_abs[idx] = torch.cat([diff_weights_abs[idx], torch.abs(diff_weight.flatten().cpu())])
                            if seq: w = diff_weight + w

                # find cutoff value for each weight
                if merged_cutoff and (self.n_parametrizations > 1):
                    min_cutoffs = [_get_cutoff(x, merged_min_pct, abs=False) for x in diff_weights_abs]
                    if merged_min_pct >= pct:
                        print(f"merged_min_pct >= pct, using target sparsity merged_min_pct={merged_min_pct}")
                        cutoffs = min_cutoffs
                    else:
                        remaining = torch.cat([x[x<c] for x,c in zip(diff_weights_abs, min_cutoffs)])
                        remaining_cutoff = _get_cutoff(remaining, pct - merged_min_pct)
                        cutoffs = [min(remaining_cutoff, c) for c in min_cutoffs]
                else:
                    cutoffs = [_get_cutoff(x, pct, abs=False) for x in diff_weights_abs]

            for base_module in self.get_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    diff_weights = []
                    w = par_list.original
                    for idx, seq in enumerate(sequential):
                        diff_weight = par_list[idx].diff_weight(w)
                        if pct is not None:
                            i = 0 if merged_cutoff else idx
                            diff_mask = (torch.abs(diff_weight) > cutoffs[i])
                        else:
                            diff_mask = ~torch.isclose(diff_weight, torch.tensor(0.), rtol=1e-8)
                        diff_weights.append((diff_weight, diff_mask))
                        if seq: w = diff_weight + w

                    parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)
                    for (diff_weight, diff_mask) in diff_weights:
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))

        self.model_state = "FIXMASK"
        self.fixmask_pct = pct

    def _get_diff_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        learning_rate_alpha: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> list:

        if idx is None:
            idx_len = 0
            idx = ""
        else:
            idx_len = len(str(idx))

        if self.model_state == "FIXMASK":
            return [
                {
                    "name": "fixmask_diff",
                    "params": [p for n,p in self.named_parameters() if n[-(12+idx_len):] == f"{idx}.diff_weight"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                }
            ]
        elif self.model_state == "FINETUNING":
            return [
                {
                    "name": "finetune_weights",
                    "params": [p for n,p in self.named_parameters() if n[-(9+idx_len):] == f"{idx}.finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "name": "finetune_alpha_params",
                    "params": [p for n,p in self.named_parameters() if n[-(6+idx_len):]==f"{idx}.alpha" or n[-(12+idx_len):]==f"{idx}.alpha_group"],
                    "lr": learning_rate_alpha
                }
            ]
        else :
            assert 0, "Undefined Model State {}".format(self.model_state)

    import contextlib
    @contextlib.contextmanager
    def deterministic(self):
        tmp_state = self.training
        if tmp_state: self.eval()
        yield
        if tmp_state: self.train()
    
    def get_diff_weights(self, idx: int, as_module: bool = False):
        from functools import reduce

        res = []
        p_names = [n[:-9] for n, _ in self.named_parameters() if n[-9:]==".original"]
        with torch.no_grad():
            for p_name in p_names:
                par_list = reduce(lambda a,b: getattr(a,b), [self.encoder] + p_name.split("."))
                par = par_list[idx]
                if isinstance(par, DiffWeightFixmask):
                    diff_weight = par.mask * par.diff_weight
                elif isinstance(par, DiffWeightFinetune):
                    w = par_list.original.detach()
                    diff_weight = par.diff_weight(w)
                res.append((p_name.replace(".parametrizations", ""), diff_weight))

        if as_module:
            return self._as_module(res)
        else:
            return res

    @torch.no_grad()
    def _count_non_zero_params(self, *args, **kwargs) -> Tuple[int, int, int]:
        l = [self._count_non_zero_params_for_module(m, *args, **kwargs) for m in list(self.get_base_modules())]
        return [sum(x) for x in list(zip(*l))]

    @torch.no_grad()
    def _count_non_zero_params_for_module(self, m: torch.nn.Module, idx: Optional[int] = None, merged: bool = False) -> Tuple[int, int, int]:
            def count_fn(p, binary: bool):
                if binary:
                    p = p.bool()
                    n_p = p.numel()
                    n_p_zero = (~p).sum()
                    n_p_one = (n_p - n_p_zero)
                else:
                    n_p = p.numel()
                    n_p_zero = (p == 0.).sum()
                    n_p_one = (p == 1.).sum()
                return torch.tensor([n_p, n_p_zero, n_p_one])

            assert hasattr(m, "parametrizations"), "module has no parametrizations"
            p_counts = torch.zeros((3,), dtype=int)
            with self.deterministic():
                for n, par_list in list(m.parametrizations.items()):
                    if merged:
                        if isinstance(par_list[0], DiffWeightFixmask):
                            p = torch.stack([x.mask for x in par_list]).sum(0)
                        else:
                            p = torch.stack([(x.z != 0.) for x in par_list]).sum(0)
                        p_counts += count_fn(p, True)
                    else:
                        if idx is not None: par_list = [par_list[idx]]
                        for par in par_list:
                            p = par.mask if isinstance(par, DiffWeightFixmask) else par.z
                            p_counts += count_fn(p, p.dtype==torch.bool)

            return p_counts.tolist()

    def forward(self, hidden_states: Tensor):
        input_tensor = hidden_states

        # shared important layers
        hidden_states_shared = self.fc1_shared(hidden_states)
        hidden_states_shared = self.intermediate_act_fn(hidden_states_shared)
        hidden_states_shared = self.fc2_shared(hidden_states_shared)

        # expert-unique layers
        hidden_states_unique = self.fc1_unique(hidden_states)
        hidden_states_unique = self.intermediate_act_fn(hidden_states_unique)
        hidden_states_unique = self.fc2_unique(hidden_states_unique)

        # add partial sums
        hidden_states = hidden_states_shared + hidden_states_unique

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


@dataclass
class MoEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None


@dataclass
class MoEModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    gate_loss: torch.FloatTensor = None
