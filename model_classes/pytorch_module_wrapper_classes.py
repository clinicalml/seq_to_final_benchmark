import torch
import torchvision
import numpy as np
from abc import ABC
from copy import deepcopy

class module_wrapper(torch.nn.Module):
    
    def get_parameter(self,
                      target):
        '''
        Get the parameter given by target if it exists, otherwise throws an error.
        Implementation taken from PyTorch v2.1
        This method is added here to support earlier versions of PyTorch
        @param self:
        @param target: str, fully-qualified name of the parameter to look for
        @return: torch.nn.Parameter, the parameter referenced by target
        '''
        module_path, _, param_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: torch.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an "
                                 "nn.Parameter")

        return param
    
    def get_submodule(self,
                      target):
        '''
        Get the sub-module specified by target if it exists, otherwise throws an error.
        Implementation taken from PyTorch v2.1
        This method is added here to support earlier versions of PyTorch
        @param self:
        @param target: str, fully-qualified name of the sub-module to look for
        @return: torch.nn.Module, the sub-module referenced by target
        '''
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

class linear_wrapper(torch.nn.Linear, module_wrapper):
    
    def compute_param_norm(self,
                           norm = 'l2'):
        '''
        Compute norm of parameters
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = 1
        '''
        assert norm in {'l1', 'l2'}
        if norm == 'l2':
            norm_func = torch.square
        else:
            norm_func = torch.abs
        param_norm = torch.sum(norm_func(self.weight))
        if self.bias is not None:
            param_norm += torch.sum(norm_func(self.bias))
        return param_norm
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        if self.bias is not None:
            weights = torch.cat((torch.flatten(self.weight), self.bias))
        else:
            weights = torch.flatten(self.weight)
        if torch.cuda.is_available():
            weights = weights.cpu()
        return weights.detach().numpy()

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: linear_wrapper
        @param layer2: linear_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: linear_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        assert (layer1.bias is None and layer2.bias is None) or (layer1.bias is not None and layer2.bias is not None)
        assert layer1.in_features  == layer2.in_features
        assert layer1.out_features == layer2.out_features
        assert interpolation_val > 0 and interpolation_val < 1
        interpolated_layer = cls(layer1.in_features,
                                 layer1.out_features,
                                 bias = (layer1.bias is not None))
        interpolated_layer.weight = torch.nn.Parameter((1. - interpolation_val) * layer1.weight + interpolation_val * layer2.weight)
        if layer1.bias is not None:
            interpolated_layer.bias = torch.nn.Parameter((1. - interpolation_val) * layer1.bias + interpolation_val * layer2.bias)
        return interpolated_layer

    def shrink_weights(self,
                       shrink_factor):
        '''
        Multiply all weights by shrink_factor
        @param self:
        @param shrink_factor: float, between 0 and 1 exclusive
        @return: None
        '''
        assert shrink_factor > 0 and shrink_factor < 1
        self.weight = torch.nn.Parameter(self.weight * shrink_factor)
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias * shrink_factor)
    
class conv2d_wrapper(torch.nn.Conv2d, module_wrapper):
    
    def compute_param_norm(self,
                           norm = 'l2'):
        '''
        Compute norm of parameters
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = 1
        '''
        assert norm in {'l1', 'l2'}
        if norm == 'l2':
            norm_func = torch.square
        else:
            norm_func = torch.abs
        param_norm = torch.sum(norm_func(self.weight))
        if self.bias is not None:
            param_norm += torch.sum(norm_func(self.bias))
        return param_norm
    
    def ablate_zero_output(self):
        '''
        Set weights and biases to zero and require grad to False
        @param self:
        @return: None
        '''
        with torch.no_grad():
            zero_weight = torch.zeros(self.weight.shape)
            if torch.cuda.is_available():
                zero_weight = zero_weight.cuda()
            self.weight = torch.nn.Parameter(zero_weight, requires_grad = False)
            if self.bias is not None:
                zero_bias = torch.zeros(self.bias.shape)
                if torch.cuda.is_available():
                    zero_bias = zero_bias.cuda()
                self.bias = torch.nn.Parameter(zero_bias, requires_grad = False)
                
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        if self.bias is not None:
            weights = torch.cat((torch.flatten(self.weight), self.bias))
        else:
            weights = torch.flatten(self.weight)
        if torch.cuda.is_available():
            weights = weights.cpu()
        return weights.detach().numpy()

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: conv2d_wrapper
        @param layer2: conv2d_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: conv2d_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        assert (layer1.bias is None and layer2.bias is None) or (layer1.bias is not None and layer2.bias is not None)
        assert layer1.in_channels  == layer2.in_channels
        assert layer1.out_channels == layer2.out_channels
        assert layer1.kernel_size  == layer2.kernel_size
        assert layer1.stride       == layer2.stride
        assert layer1.padding      == layer2.padding
        assert layer1.dilation     == layer2.dilation
        assert layer1.groups       == layer2.groups
        assert layer1.padding_mode == layer2.padding_mode
        assert interpolation_val > 0 and interpolation_val < 1
        interpolated_layer = cls(layer1.in_channels,
                                 layer1.out_channels,
                                 layer1.kernel_size,
                                 layer1.stride,
                                 layer1.padding,
                                 layer1.dilation,
                                 layer1.groups,
                                 bias         = (layer1.bias is not None),
                                 padding_mode = layer1.padding_mode)
        interpolated_layer.weight = torch.nn.Parameter((1. - interpolation_val) * layer1.weight + interpolation_val * layer2.weight)
        if layer1.bias is not None:
            interpolated_layer.bias = torch.nn.Parameter((1. - interpolation_val) * layer1.bias + interpolation_val * layer2.bias)
        return interpolated_layer

    def shrink_weights(self,
                       shrink_factor):
        '''
        Multiply all weights by shrink_factor
        @param self:
        @param shrink_factor: float, between 0 and 1 exclusive
        @return: None
        '''
        assert shrink_factor > 0 and shrink_factor < 1
        self.weight = torch.nn.Parameter(self.weight * shrink_factor)
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias * shrink_factor)
        
class sequential_wrapper(torch.nn.Sequential, module_wrapper):
    
    def compute_param_norm(self,
                           norm = 'l2'):
        '''
        Compute norm of parameters
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = 1
        '''
        assert norm in {'l1', 'l2'}
        param_norm = 0.
        for module in self:
            param_norm += module.compute_param_norm(norm)
        return param_norm
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        weights = []
        for module in self:
            weights.append(module.get_param_vec())
        return np.concatenate(weights)

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: sequential_wrapper
        @param layer2: sequential_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: sequential_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        assert len(layer1) == len(layer2)
        assert interpolation_val > 0 and interpolation_val < 1
        new_modules = []
        for module1, module2 in zip(layer1, layer2):
            assert type(module1) is type(module2)
            new_modules.append(type(module1).interpolate_weights(module1,
                                                                 module2,
                                                                 interpolation_val))
        return cls(*new_modules)

    def shrink_weights(self,
                       shrink_factor):
        '''
        Multiply all weights by shrink_factor
        @param self:
        @param shrink_factor: float, between 0 and 1 exclusive
        @return: None
        '''
        assert shrink_factor > 0 and shrink_factor < 1
        for module in self:
            module.shrink_weights(shrink_factor)
    
class module_list_over_time(torch.nn.ModuleList):
    
    def forward(self,
                x,
                t,
                sum_up_to_t = False):
        '''
        Compute output from list of layers over time by grouping samples from the same time step
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @param sum_up_to_t: bool, whether to sum outputs up to t
        @return: torch FloatTensor, output
        '''
        if isinstance(t, int):
            assert t >= 0 and t < len(self)
            if sum_up_to_t:
                out = 0
                for time_idx in range(t + 1):
                    out += self[time_idx](x)
            else:
                out = self[t](x)
        else:
            # create batches of samples from the same time step
            out = torch.empty(x.shape)
            for time_idx in range(len(self)):
                time_sample_idxs = torch.nonzero(t == time_idx, as_tuple = True)[0]
                if len(time_sample_idxs) > 0:
                    if sum_up_to_t:
                        for sum_time_idx in range(time_idx + 1):
                            out[time_sample_idxs] += self[sum_time_idx](x[time_sample_idxs])
                    else:
                        out[time_sample_idxs] = self[time_idx](x[time_sample_idxs])
        return out
    
    def freeze_params_up_to_time(self,
                                 t):
        '''
        Set require grad to False for parameters at all time steps up to and including t
        @param self:
        @param t: int, time step to freeze up to
        @return: None
        '''
        assert t >= 0 and t < len(self)
        for time_idx in range(t + 1):
            for param in self[time_idx].parameters():
                param.requires_grad = False
    
    def ablate_modules_after_time0(self):
        '''
        Assumes modules after time 0 are conv2d_wrapper objects
        Set weights and biases to 0 and require grad to False
        @param self:
        @return: None
        '''
        for time_idx in range(1, len(self)):
            assert isinstance(self[time_idx], conv2d_wrapper)
            self[time_idx].ablate_zero_output()
            
    def ablate_adapters(self):
        '''
        Assumes modules after time 0 are adapters so all entries in list are set to 0th module
        @param self:
        @return: None
        '''
        for time_idx in range(1, len(self)):
            self[time_idx] = self[0]
            
    def load_partial_state_dict(self,
                                pretrained_module,
                                use_partial_to_init_final = False,
                                all_modules               = False):
        '''
        Load state into initial modules of this list.
        May load from a list with at most the same number of modules as this list
        or from a single module into either the initial time step or all time steps of this list
        @param self:
        @param pretrained_module: module_list_over_time, torch.nn.ModuleList, or torch.nn.Module to load weights from
        @param use_partial_to_init_final: bool, whether to use module at time T - 1 to initialize module at T
        @param all_modules: bool, load pretrained module weights into ALL modules in this list
        @return: None
        '''
        with torch.no_grad():
            if isinstance(pretrained_module, torch.nn.ModuleList):
                if use_partial_to_init_final:
                    assert len(pretrained_module) == len(self) - 1
                    self[-1].load_state_dict(pretrained_module[-1].state_dict())
                else:
                    assert len(pretrained_module) <= len(self)
                for module_idx in range(len(pretrained_module)):
                    self[module_idx].load_state_dict(pretrained_module[module_idx].state_dict())
            else:
                self[0].load_state_dict(pretrained_module.state_dict())
                if all_modules:
                    for module_idx in range(1, len(self)):
                        self[module_idx].load_state_dict(pretrained_module.state_dict())
                elif use_partial_to_init_final:
                    assert len(self) == 2
                    self[-1].load_state_dict(pretrained_module.state_dict())
            
    def compute_fisher_info(self):
        '''
        Compute Fisher info of each parameter at each time step
        @param self:
        @return: list over time of dicts mapping str parameter name 
                 to torch FloatTensor containing Fisher info
        '''
        fisher_info = [dict() for _ in range(len(self))]
        for time_idx in range(len(self)):
            for name, param in self[time_idx].named_parameters():
                with torch.no_grad():
                    fisher_info[time_idx][name] = torch.square(param.grad.detach())
        return fisher_info
            
    def compute_adjacent_param_norm_at_each_time_step(self,
                                                      norm        = 'l2',
                                                      fisher_info = None):
        '''
        Compute sum of adjacent parameter norms at each time step
        Assumes same modules at each time point and fisher info contains all parameters at each time step if provided
        @param self:
        @param norm: str, l1 or l2 norm
        @param fisher_info: list over time of dicts mapping str parameter name 
                            to torch FloatTensor containing Fisher info,
                            used to weight adjacent parameter norms when computing sum,
                            default: weight 1 for each parameter
        @return: torch FloatTensor
        '''
        assert norm in {'l1', 'l2'}
        if fisher_info is not None:
            assert len(fisher_info) == len(self) - 1
        adjacent_norm = torch.zeros(len(self) - 1)
        if torch.cuda.is_available():
            adjacent_norm = adjacent_norm.cuda()
        for time_idx in range(len(self) - 1):
            for name, param in self[time_idx].named_parameters():
                if norm == 'l2':
                    param_diff_norm = torch.square(self[time_idx + 1].get_parameter(name) - param)
                else:
                    param_diff_norm = torch.abs(self[time_idx + 1].get_parameter(name) - param)
                if fisher_info is not None:
                    param_diff_norm *= fisher_info[time_idx][name]
                adjacent_norm[time_idx] += torch.sum(param_diff_norm)
        return adjacent_norm
            
    def compute_param_norm_at_each_time_step(self,
                                             norm = 'l2'):
        '''
        Compute norm of parameters at each time step
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = # time steps
        '''
        assert norm in {'l1', 'l2'}
        param_norms = torch.zeros(len(self))
        if torch.cuda.is_available():
            param_norms = param_norms.cuda()
        for time_idx in range(len(self)):
            param_norms[time_idx] = self[time_idx].compute_param_norm(norm)        
        return param_norms
    
    def get_param_vec(self,
                      t):
        '''
        Create a flat vector of all parameters flattened and concatenated together at index t
        @param self:
        @param t: int, index to compute
        @return: np array
        '''
        assert t >= 0 and t < len(self)
        return self[t].get_param_vec()

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: module_list_over_time
        @param layer2: module_list_over_time
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: module_list_over_time
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        assert len(layer1) == len(layer2)
        assert interpolation_val > 0 and interpolation_val < 1
        modules = []
        for module1, module2 in zip(layer1, layer2):
            assert type(module1) is type(module2)
            modules.append(type(module1).interpolate_weights(module1,
                                                             module2,
                                                             interpolation_val))
        return cls(modules)
    
class no_param_wrapper(module_wrapper, ABC):
    
    def compute_param_norm(self,
                           norm = 'l2'):
        '''
        Compute norm of parameters
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = 1
        '''
        assert norm in {'l1', 'l2'}
        return 0.
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array, length 0
        '''
        return np.empty(0)

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Return a new instance
        @param cls:
        @param layer1: no_param_wrapper
        @param layer2: no_param_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: no_param_wrapper
        '''
        assert type(layer1) is type(layer2)
        assert interpolation_val > 0 and interpolation_val < 1
        return deepcopy(layer1)

    def shrink_weights(self,
                       shrink_factor):
        '''
        Does nothing
        @param self:
        @param shrink_factor: float, between 0 and 1 exclusive
        @return: None
        '''
        assert shrink_factor > 0 and shrink_factor < 1
        return
    
class batchnorm2d_wrapper(torch.nn.BatchNorm2d, no_param_wrapper):
    
    pass
    
class relu_wrapper(torch.nn.ReLU, no_param_wrapper):
    
    pass

class avgpool2d_wrapper(torch.nn.AvgPool2d, no_param_wrapper):
    
    pass
    
class adaptive_avgpool2d_wrapper(torch.nn.AdaptiveAvgPool2d, no_param_wrapper):
    
    pass

class flatten_wrapper(torch.nn.Flatten, no_param_wrapper):
    
    pass
    
class dropout_wrapper(torch.nn.Dropout, no_param_wrapper):
    
    pass
    
class dropout_wrapper(torch.nn.Dropout, no_param_wrapper):
    
    pass

class maxpool2d_wrapper(torch.nn.MaxPool2d, no_param_wrapper):
    
    pass

class localresponsenorm_wrapper(torch.nn.LocalResponseNorm, no_param_wrapper):
    
    pass