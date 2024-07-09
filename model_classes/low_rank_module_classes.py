from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torchvision

from pytorch_module_wrapper_classes import batchnorm2d_wrapper, module_list_over_time

class low_rank_module(torch.nn.Module, ABC):
    
    @abstractmethod
    def forward(self,
                x,
                t):
        '''
        Compute output from low-rank module at a particular time step
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        pass
    
    def freeze_params_up_to_time(self,
                                 t):
        '''
        Set require grad to False for parameters at all time steps up to and including t
        @param self:
        @param t: int, time step to freeze up to
        @return: None
        '''
        assert t >= 0 and t <= self.num_low_rank_adapters
        self.weight.requires_grad = False
        for adapter_idx in range(t):
            self.input_adapters[adapter_idx].requires_grad  = False
            self.output_adapters[adapter_idx].requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            for adapter_idx in range(t):
                self.bias_adapters[adapter_idx].requires_grad = False
                
    def ablate_adapters(self):
        '''
        Set adapters to have no effect
        For multiply mode:
        - Set input adapters to all 1
        - Set output adapters to all 1/rank
        For add mode:
        - Set input adapters to all 0
        For both modes:
        - Set bias adapters to all 0
        - Set require grad to False for all adapters
        @param self:
        @return: None
        '''
        for adapter_idx in range(self.num_low_rank_adapters):
            if self.adapter_mode == 'multiply':
                input_adapter_ones  = torch.ones(self.input_adapters[adapter_idx].shape)
                output_adapter_ones = torch.ones(self.output_adapters[adapter_idx].shape)/self.adapter_rank
                if torch.cuda.is_available():
                    input_adapter_ones  = input_adapter_ones.cuda()
                    output_adapter_ones = output_adapter_ones.cuda()
                self.input_adapters[adapter_idx]  = torch.nn.Parameter(input_adapter_ones,
                                                                       requires_grad = False)
                self.output_adapters[adapter_idx] = torch.nn.Parameter(output_adapter_ones,
                                                                       requires_grad = False)
            else:
                input_adapter_zeros = torch.zeros(self.input_adapters[adapter_idx].shape)
                if torch.cuda.is_available():
                    input_adapter_zeros = input_adapter_zeros.cuda()
                self.input_adapters[adapter_idx]  = torch.nn.Parameter(input_adapter_zeros,
                                                                       requires_grad = False)
                self.output_adapters[adapter_idx].requires_grad = False
            if self.bias is not None:
                bias_zeros = torch.zeros(self.bias_adapters[adapter_idx].shape)
                if torch.cuda.is_available():
                    bias_zeros = bias_zeros.cuda()
                self.bias_adapters[adapter_idx]   = torch.nn.Parameter(bias_zeros,
                                                                       requires_grad = False)
    
    def load_partial_state_dict(self,
                                pretrained_module):
        '''
        Load state into initial modules of this list.
        May load from a low rank module with at most the same number of adapters as this low rank module
        or from a single pytorch module into only the initial module
        @param self:
        @param pretrained_module: low_rank_module or torch.nn.Module to load weights from
        @return: None
        '''
        with torch.no_grad():
            self.weight = deepcopy(pretrained_module.weight)
            if self.bias is not None:
                self.bias = deepcopy(pretrained_module.bias)
            else:
                assert pretrained_module.bias is None
            if isinstance(pretrained_module, low_rank_module):
                assert pretrained_module.num_low_rank_adapters <= self.num_low_rank_adapters
                assert pretrained_module.adapter_rank == self.adapter_rank
                assert pretrained_module.adapter_mode == self.adapter_mode
                for adapter_idx in range(pretrained_module.num_low_rank_adapters):
                    self.input_adapters[adapter_idx]    = deepcopy(pretrained_module.input_adapters[adapter_idx])
                    self.output_adapters[adapter_idx]   = deepcopy(pretrained_module.output_adapters[adapter_idx])
                    if self.bias is not None:
                        self.bias_adapters[adapter_idx] = deepcopy(pretrained_module.bias_adapters[adapter_idx])
            
    def compute_param_norm_at_each_time_step(self,
                                             norm):
        '''
        Compute norm of parameters at each time step
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = # time steps
        '''
        assert norm in {'l1', 'l2'}
        param_norms = torch.zeros(self.num_low_rank_adapters + 1)
        if torch.cuda.is_available():
            param_norms = param_norms.cuda()
        if norm == 'l2':
            norm_func = torch.square
        else:
            norm_func = torch.abs
        param_norms[0] = torch.sum(norm_func(self.weight))
        if self.bias is not None:
            param_norms[0] += torch.sum(norm_func(self.bias))
        for adapter_idx in range(self.num_low_rank_adapters):
            param_norms[adapter_idx + 1] += torch.sum(norm_func(self.input_adapters[adapter_idx]))
            param_norms[adapter_idx + 1] += torch.sum(norm_func(self.output_adapters[adapter_idx]))
            if self.bias is not None:
                param_norms[adapter_idx + 1] += torch.sum(norm_func(self.bias_adapters[adapter_idx]))
        return param_norms
    
    def _get_adapted_weight_and_bias(self,
                                     t):
        '''
        Compute adapted parameter vector at a particular time step
        @param self:
        @param t: int, time step to compute
        @return: 1. torch FloatTensor, weight at time step
                 2. torch FloatTensor, bias as time step, None if does not exist
        '''
        assert t >= 0 and t <= self.num_low_rank_adapters
        if t == 0:
            return self.weight, self.bias
        if self.adapter_mode == 'multiply':
            adapter_weight = 1
        else:
            adapter_weight = 0
        for adapter_idx in range(t):
            this_adapter_weight = self.output_adapters[adapter_idx] @ self.input_adapters[adapter_idx]
            if len(this_adapter_weight.shape) == 4:
                this_adapter_weight = this_adapter_weight.permute((2, 3, 0, 1))
            if self.adapter_mode == 'multiply':
                adapter_weight *= this_adapter_weight
            else:
                adapter_weight += this_adapter_weight
        if self.adapter_mode == 'multiply':
            adapted_weight = adapter_weight * self.weight
        else:
            adapted_weight = adapter_weight + self.weight
        if self.bias is not None:
            adapted_bias = self.bias_adapters[t - 1]
        else:
            adapted_bias = None
        return adapted_weight, adapted_bias
    
    def get_param_vec(self,
                      t):
        '''
        Compute adapted parameter vector at a particular time step
        Flatten and concatenate all parameters
        @param self:
        @param t: int, time step to compute
        @return: np array
        '''
        adapted_weight, adapted_bias = self._get_adapted_weight_and_bias(t)
        adapted_weight = torch.flatten(adapted_weight)
        if adapted_bias is not None:
            adapted_weight = torch.cat((adapted_weight, adapted_bias))
        if torch.cuda.is_available():
            return adapted_weight.cpu().numpy()
        return adapted_weight.numpy()

class low_rank_linear(low_rank_module, torch.nn.Linear):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 in_features, 
                 out_features,
                 bias   = True):
        '''
        Initialize a linear layer with low-rank adapters A_i (adapter_rank x in_features) and A_o (out_features x adapter_rank)
        At each time step, (A_o x A_i) is multiplied or added elementwise to weight.
        A_i and A_o are initialized from U(-1.5, 1.5).
        New bias is added. Biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/(input_dim).
        @param self:
        @param num_low_rank_adapters: int, number of adapters, each corresponds to a new time step
        @param adapter_rank: int, number of adapter pairs at each time step
        @param adapter_mode: str, multiply or add
        @param in_features: int, number of input features
        @param out_features: int, number of output features
        @param bias: bool, whether to include a bias
        @return: None
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters:
            assert adapter_rank > 0
            assert adapter_rank * (in_features + out_features) < in_features * out_features, \
                'adapters should be more parameter efficient'
            assert adapter_mode in {'multiply', 'add'}
        assert in_features > 0
        assert out_features > 0
        super(low_rank_linear, self).__init__(in_features,
                                              out_features,
                                              bias)
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        self.input_adapters        = torch.nn.ParameterList([])
        self.output_adapters       = torch.nn.ParameterList([])
        if bias:
            self.bias_adapters     = torch.nn.ParameterList([])
        with torch.no_grad():
            for adapter_idx in range(num_low_rank_adapters):
                input_factor  = torch.rand((adapter_rank, in_features), dtype = torch.float)  * 3 - 1.5
                output_factor = torch.rand((out_features, adapter_rank), dtype = torch.float) * 3 - 1.5
                self.input_adapters.append(torch.nn.Parameter(input_factor))
                self.output_adapters.append(torch.nn.Parameter(output_factor))
                if bias:
                    k = 1./(in_features)
                    k_sqrt = np.sqrt(k)
                    new_bias  = torch.rand(out_features, dtype = torch.float) * 2 * k_sqrt - k_sqrt
                    self.bias_adapters.append(torch.nn.Parameter(new_bias))
    
    def forward(self,
                x,
                t):
        '''
        Compute output from low-rank linear layer at a particular time step
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        if isinstance(t, int):
            adapted_weight, adapted_bias = self._get_adapted_weight_and_bias(t)
            return torch.nn.functional.linear(x,
                                              adapted_weight,
                                              adapted_bias)
        
        assert torch.all(t >= 0) and torch.all(t <= self.num_low_rank_adapters)
        assert x.shape[0] == len(t)
        out = torch.empty((len(t), self.out_features))
        
        time0_sample_idxs = torch.nonzero(t == 0, as_tuple = True)[0]
        if len(time0_sample_idxs) > 0:
            out[time0_sample_idxs] = torch.nn.functional.linear(x[time0_sample_idxs], self.weight, self.bias)
        
        if self.adapter_mode == 'multiply':
            adapter_weight = 1
        else:
            adapter_weight = 0
        for adapter_idx in range(self.num_low_rank_adapters):
            this_adapter_weight = self.output_adapters[adapter_idx] @ self.input_adapters[adapter_idx]
            if self.adapter_mode == 'multiply':
                adapter_weight *= this_adapter_weight
            else:
                adapter_weight += this_adapter_weight
            
            time_sample_idxs = torch.nonzero(t == adapter_idx + 1, as_tuple = True)[0]
            if len(time_sample_idxs) > 0:
                if self.bias is not None:
                    adapted_bias      = self.bias_adapters[adapter_idx]
                else:
                    adapted_bias      = None
                if self.adapter_mode == 'multiply':
                    adapted_weight    = adapter_weight * self.weight
                else:
                    adapted_weight    = adapter_weight + self.weight
                out[time_sample_idxs] = torch.nn.functional.linear(x[time_sample_idxs],
                                                                   adapted_weight,
                                                                   adapted_bias)
        return out
    
class low_rank_conv2d(low_rank_module, torch.nn.Conv2d):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride       = 1,
                 padding      = 0,
                 dilation     = 1,
                 groups       = 1,
                 bias         = True,
                 padding_mode = 'zeros'):
        '''
        Initialize conv2d layer with low-rank A_i (kernel_size x kernel_size x adapter_rank x in_channels) 
        and A_o (kernel_size x kernel_size x out_channels x adapter_rank) adapters.
        At each time step, (A_o x A_i) is multiplied or added element-wise to weight.
        A_i and A_o are initialized from U(-1.5, 1.5).
        New bias is added. Biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/(3 * kernel_size * kernel_size)
        @param self:
        @param num_low_rank_adapters: int, number of adapters, each corresponds to a new time step
        @param adapter_rank: int, rank of adapters at each time step
        @param adapter_mode: str, multiply or add
        @param in_channels: int, number of input channels
        @param out_channels: int, number of output channels
        @param kernel_size: int or tuple of 2 ints, size of kernels
        @param stride: int or tuple of ints, # units to move kernel between each convolution
        @param padding: int, # units to pad
        @param groups: int, # blocked connections from input to output
        @param bias: bool, whether to include bias
        @param padding_mode: str, how to pad
        @return: None
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters:
            assert adapter_rank > 0
            assert adapter_rank * (in_channels + out_channels) < in_channels * out_channels, \
                'adapters should be more parameter efficient' 
            assert adapter_mode in {'multiply', 'add'}
        assert in_channels > 0
        assert out_channels > 0
        super(low_rank_conv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              dilation,
                                              groups,
                                              bias,
                                              padding_mode)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert len(kernel_size) == 2
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        
        self.input_adapters    = torch.nn.ParameterList([])
        self.output_adapters   = torch.nn.ParameterList([])
        if bias:
            self.bias_adapters = torch.nn.ParameterList([])

        with torch.no_grad():
            for adapter_idx in range(num_low_rank_adapters):
                input_factor  = torch.rand((kernel_size[0], kernel_size[1], adapter_rank, in_channels), 
                                            dtype = torch.float) * 3 - 1.5
                output_factor = torch.rand((kernel_size[0], kernel_size[1], out_channels, adapter_rank),
                                           dtype = torch.float) * 3 - 1.5
                self.input_adapters.append(torch.nn.Parameter(input_factor))
                self.output_adapters.append(torch.nn.Parameter(output_factor))
                if bias:
                    k = 1./(3 * kernel_size[0] * kernel_size[1])
                    k_sqrt = np.sqrt(k)
                    new_bias  = torch.rand(out_channels, dtype = torch.float) * 2 * k_sqrt - k_sqrt
                    self.bias_adapters.append(torch.nn.Parameter(new_bias))

    def _conv_forward(self,
                      x,
                      weight,
                      bias = None):
        '''
        Default implementation taken from PyTorch v2.1
        Overloading for earlier versions of PyTorch that do not take bias argument
        Apply padding if needed. Then apply convolution with specified weight and bias.
        @param self:
        @param x: torch FloatTensor, input features
        @param weight: torch FloatTensor, convolutional kernels
        @param bias: torch FloatTensor, biases for output channels
        @return: torch FloatTensor
        '''
        if self.padding_mode != 'zeros':
            padded_x = torch.nn.functional.pad(x, 
                                               self._reversed_padding_repeated_twice, 
                                               mode=self.padding_mode)
            return torch.nn.functional.conv2d(padded_x,
                                              weight, 
                                              bias, 
                                              self.stride,
                                              torch.modules.utils_pair(0),
                                              self.dilation,
                                              self.groups)
        return torch.nn.functional.conv2d(x,
                                          weight,
                                          bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups)
    
    def forward(self,
                x,
                t):
        '''
        Compute output from low-rank conv2d at particular time steps
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        if isinstance(t, int):
            adapted_weight, adapted_bias = self._get_adapted_weight_and_bias(t)
            return self._conv_forward(x,
                                      adapted_weight,
                                      adapted_bias)
        
        assert torch.all(t >= 0) and torch.all(t <= self.num_low_rank_adapters)
        assert x.shape[0] == len(t)
        out_sample_shape = self._conv_forward(x[0], self.weight, self.bias).shape
        out = torch.empty((len(t), out_sample_shape[0], out_sample_shape[1], out_sample_shape[2]))
        
        time0_sample_idxs = torch.nonzero(t == 0, as_tuple = True)[0]
        if len(time0_sample_idxs) > 0:
            out[time0_sample_idxs] = self._conv_forward(x[time0_sample_idxs], self.weight, self.bias)
        
        if self.adapter_mode == 'multiply':
            adapter_weight = 1
        else:
            adapter_weight = 0
        for adapter_idx in range(self.num_low_rank_adapters):
            this_adapter_weight = self.output_adapters[adapter_idx] @ self.input_adapters[adapter_idx]
            this_adapter_weight = this_adapter_weight.permute((2, 3, 0, 1))
            if self.adapter_mode == 'multiply':
                adapter_weight *= this_adapter_weight
            else:
                adapter_weight += this_adapter_weight
            
            time_sample_idxs = torch.nonzero(t == adapter_idx + 1, as_tuple = True)[0]
            if len(time_sample_idxs) > 0:
                if self.adapter_mode == 'multiply':
                    adapted_weight = adapter_weight * self.weight
                else:
                    adapted_weight = adapter_weight + self.weight
                if self.bias is not None:
                    adapted_bias = self.bias_adapters[adapter_idx]
                else:
                    adapted_bias = None
                out[time_sample_idxs] = self._conv_forward(x[time_sample_idxs],
                                                           adapted_weight,
                                                           adapted_bias)
        return out
    
class low_rank_module_list(torch.nn.ModuleList, ABC):
    
    def __init__(self,
                 modules):
        '''
        Initialize a list of low-rank modules
        @param self:
        @param modules: list of low-rank modules with the same # adapters, adapter rank, and adapter mode
        @return: None
        '''
        super(low_rank_module_list, self).__init__(modules)
        self.num_low_rank_adapters = self[0].num_low_rank_adapters
        self.adapter_rank          = self[0].adapter_rank
        self.adapter_mode          = self[0].adapter_mode
        for i in range(1, len(self)):
            assert self[i].num_low_rank_adapters == self.num_low_rank_adapters
            assert self[i].adapter_rank          == self.adapter_rank
            assert self[i].adapter_mode          == self.adapter_mode
    
    def freeze_params_up_to_time(self,
                                 t):
        '''
        Set require grad to False for parameters at all time steps up to and including t
        @param self:
        @param t: int, time step to freeze up to
        @return: None
        '''
        assert t >= 0 and t <= self.num_low_rank_adapters
        for module in self:
            module.freeze_params_up_to_time(t)
    
    def ablate_adapters(self):
        '''
        Set adapters to have no effect
        For multiply mode:
        - Set input adapters to all 1
        - Set output adapters to all 1/rank
        For add mode:
        - Set input adapters to all 0
        For both modes:
        - Set bias adapters to all 0
        - Set require grad to False for all adapters
        - Set batch norm modules to all be the first batch norm module
        @param self:
        @return: None
        '''
        for module in self:
            module.ablate_adapters()
    
    def load_partial_state_dict(self,
                                pretrained_module):
        '''
        Load state into initial modules of this list.
        May load from a low rank module list with at most the same number of adapters as this low rank module list
        or from a wrapper class that has the same architecture and can be iterated like a list
        into only the initial modules
        @param self:
        @param pretrained_module: low_rank_module_list or module_wrapper to load weights from
        @return: None
        '''
        with torch.no_grad():
            if isinstance(pretrained_module, low_rank_module_list):
                assert pretrained_module.num_low_rank_adapters <= self.num_low_rank_adapters
                assert pretrained_module.adapter_rank == self.adapter_rank
                assert pretrained_module.adapter_mode == self.adapter_mode
            for idx in range(len(self)):
                self[idx].load_partial_state_dict(pretrained_module[idx])
            
    def compute_param_norm_at_each_time_step(self,
                                             norm):
        '''
        Compute norm of parameters at each time step
        @param self:
        @param norm: str, l1 or l2 norm
        @return: torch FloatTensor, length = # time steps
        '''
        assert norm in {'l1', 'l2'}
        return torch.sum(torch.stack([module.compute_param_norm_at_each_time_step(norm)
                                      for module in self]),
                         dim = 0)
        
class low_rank_sequential(low_rank_module_list, torch.nn.Sequential):
    
    def __init__(self, *args):
        '''
        Initialize a low-rank sequential module
        @param self:
        @param *args: modules to put in sequential module
        @return: None
        '''
        torch.nn.Sequential.__init__(self, *args)
        low_rank_module_list.__init__(self, list(args))
    
    def forward(self,
                x,
                t):
        '''
        Compute output from low-rank sequential module at particular time steps
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        for module in self:
            x = module(x, t)
        return x