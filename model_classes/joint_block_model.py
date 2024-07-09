import torch
import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

from pytorch_module_wrapper_classes import (
    module_wrapper,
    module_list_over_time,
    conv2d_wrapper,
    batchnorm2d_wrapper,
    sequential_wrapper
)
from block_model import block_model
from convnet_with_dropout import convnet_with_dropout
from densenet_with_dropout import densenet_with_dropout
from resnet_with_dropout import resnet_with_dropout

class joint_block_model(block_model, ABC):
    
    @abstractmethod
    def __init__(self,
                 num_time_steps,
                 separate_layers,
                 model_properties = {'num_blocks': 4},
                 num_classes      = 10,
                 mode             = 'separate',
                 adapter_ranks    = [0,0,0,0,0,0],
                 adapter_mode     = None,
                 side_layers      = ['separate',[],[],[],[],'separate']):
        '''
        Initialize a joint block model that has some shared layers across time steps
        @param self:
        @param num_time_steps: int, number of time steps in model
        @param separate_layers: list of list of str, specify which layers will be new modules at each time step,
                                outer list over time steps (first entry is 2nd time step 
                                since all modules are new at 1st time step), second list), 
                                outer list has length num_time_steps - 1
                                inner list over layers: subset of conv1, layer1, layer2, layer3, layer4, fc,
                                example: [[conv1, layer1], [fc]] means 3 time step model has 
                                - one set of conv1, layer1 at time 0 and another set shared at times 1 and 2
                                - one fc module at times 0 and 1, another fc module at time 2
                                - all other modules shared across all 3 time steps
        @param model_properties: dict mapping str to list of int over layers, properties must include num_blocks
        @param num_classes: int, number of output classes
        @param mode: str, separate to create new layers at separate time steps,
                     side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                     low_rank_adapt to multiply or add to weights with low-rank adapters,
                     parameter-efficient modes only apply to residual blocks 
                     because no smaller side module can be constructed 
                     and rank >= 3 or rank >= num_classes would not be parameter-efficient
        @param adapter_ranks: list of int, rank of adapters in each layer
        @param adapter_mode: str, multiply or add
        @param side_layers: list of (str or lists of int), number of output channels 
                            for each of the intermediate convolution layers in side modules,
                            outer list over layers in model
                            str option is "block" for side module to be a residual block,
                            str option is "separate" for conv1 and fc layers
        @return: None
        '''
        assert num_time_steps > 1
        assert len(separate_layers) == num_time_steps - 1
        assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
        if mode == 'low_rank_adapt':
            assert adapter_mode in {'multiply', 'add'}
        
        block_model.__init__(self,
                             model_properties,
                             num_classes)
        self.num_time_steps   = num_time_steps
        self.separate_layers  = separate_layers
        self.mode             = mode
        self.adapter_ranks    = adapter_ranks
        self.adapter_mode     = adapter_mode
        self.side_layers      = side_layers
        self.num_classes      = num_classes
        self.model_properties = model_properties
        self.num_blocks       = model_properties['num_blocks']
        
        # for each time step, specify which version of each layer should be used
        self.block_module_path = np.zeros((num_time_steps, self.num_blocks), dtype=int)
        if mode in {'separate', 'side_tune'}:
            block_lists = []
            for block_idx in range(self.num_blocks):
                block_lists.append([self._make_block(block_idx)])
        for time_idx in range(1, num_time_steps):
            for block_idx in range(self.num_blocks):
                if 'layer' + str(block_idx + 1) in separate_layers[time_idx - 1]:
                    if mode == 'side_tune':
                        block_lists[block_idx].append(self._make_side_module(block_idx))
                    elif mode == 'separate':
                        block_lists[block_idx].append(self._make_block(block_idx))
                self.block_module_path[time_idx:, block_idx] += 1
        self.blocks = torch.nn.ModuleList()
        if mode == 'low_rank_adapt':
            for block_idx in range(self.num_blocks):
                self.blocks.append(self._make_low_rank_block(block_idx))
        else:
            for block_idx in range(self.num_blocks):
                self.blocks.append(module_list_over_time(block_lists[block_idx]))
        self.block_names = ['layer' + str(block_idx + 1) for block_idx in range(self.num_blocks)]
        
    @abstractmethod
    def _compute_side_module_properties(self,
                                        block_idx):
        '''
        Compute number of input channels, number of output channels, kernel size, stride, and padding
        for each layer in side module so that side module maps to the same output dimension as the original block
        Check if number of parameters is at most original number of parameters
        @param self:
        @param block_idx: int, index of block to compute side module properties for
        @return: dict mapping str to list of int, list over side module layers,
                 properties: input_dims, output_dims, kernel_size, stride, padding
        '''
        pass
    
    def _make_input_conv_and_fc_head(self):
        '''
        Make input convolution and fully connected head for standard joint block models that have blocks between the two
        Throws error for other architectures that don't have input conv and fc head, e.g. AlexNet
        @param self:
        @return: None
        '''
        assert self.model_type != 'AlexNet'
        self.module_path = np.zeros((self.num_time_steps, self.num_blocks + 2), dtype=int)
        self.module_path[:,1:-1] = self.block_module_path
        if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[0] == 0:
            conv1_list = [self._make_input_conv_layer()]
        if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[-1] == 0:
            fc_list    = [self._make_fc_layer()]
        for time_idx in range(1, self.num_time_steps):
            if 'conv1' in self.separate_layers[time_idx - 1]:
                if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[0] == 0:
                    conv1_list.append(self._make_input_conv_layer())
                self.module_path[time_idx:, 0] += 1
            if 'fc' in self.separate_layers[time_idx - 1]:
                if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[-1] == 0:
                    fc_list.append(self._make_fc_layer())
                self.module_path[time_idx:, -1] += 1
        
        if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[0] == 0:
            self.conv1 = module_list_over_time(conv1_list)
        else:
            self.conv1 = self._make_low_rank_input_conv_layer()
        
        if self.mode in {'separate', 'side_tune'} or self.adapter_ranks[-1] == 0:
            self.fc    = module_list_over_time(fc_list)
        else:
            self.fc    = self._make_low_rank_fc()
        
        # put layers together in order
        self.layers      = torch.nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.extend(self.blocks)
        self.layers.append(self.fc)
        self.layer_names = ['conv1'] + self.block_names + ['fc']
        
    def _make_side_module(self,
                          block_idx):
        '''
        Make convolutional layers that map to the same output dimension as the original block
        Number of output channels for intermediate layers specified by side_layers attribute
        Add a batch norm layer at the end
        @param self:
        @param block_idx: int, index of block to create side module for
        @return: sequential_wrapper
        '''
        assert self.mode == 'side_tune'
        assert block_idx >= 0 and block_idx < self.num_blocks
        if isinstance(self.side_layers[block_idx + self.num_layers_before_blocks], str):
            assert (self.side_layers[block_idx + self.num_layers_before_blocks] == 'block') \
                or (self.model_type == 'ConvNet' and self.side_layers[block_idx + self.num_layers_before_blocks] == 'separate')
            return self._make_block(block_idx)
        side_module_properties = self._compute_side_module_properties(block_idx)
        layers = []
        for layer_idx in range(len(side_module_properties['input_dims'])):
            layers.append(conv2d_wrapper(side_module_properties['input_dims'][layer_idx],
                                         side_module_properties['output_dims'][layer_idx],
                                         kernel_size = side_module_properties['kernel_size'][layer_idx],
                                         stride      = side_module_properties['stride'][layer_idx],
                                         padding     = side_module_properties['padding'][layer_idx],
                                         bias        = False))
        layers.append(batchnorm2d_wrapper(side_module_properties['output_dims'][layer_idx]))
        return sequential_wrapper(*layers)
    
    @abstractmethod
    def _make_low_rank_block(self,
                             block_idx):
        '''
        Make a low-rank block
        @param self:
        @param block_idx: int, index of residual block, determines dimensions, strides, and whether to include 
                          a downsample block
        @return: low_rank_sequential
        '''
        pass
    
    def _forward_layer(self,
                       x,
                       t,
                       layer_idx,
                       dropout = 0):
        '''
        Compute output from a layer
        @param self:
        @param x: torch FloatTensor, input features
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @param layer_idx: int, index of layer to compute output from
        @param dropout: float, probability each output entry after each layer/block is zero'd during training
        @return: torch FloatTensor, output
        '''
        assert layer_idx >= 0 and layer_idx < len(self.layers)
        assert dropout >= 0 and dropout < 1
        layer  = self.layers[layer_idx]
        
        if isinstance(t, int):
            module_t = int(self.module_path[t, layer_idx])
        else:
            module_t = torch.LongTensor([self.module_path[t[i], layer_idx]
                                         for i in range(len(t))])
            if torch.cuda.is_available():
                module_t = module_t.cuda()
        
        if self.mode == 'low_rank_adapt':
            out = layer(x, module_t)
        else:
            sum_up_to_t = (self.mode == 'side_tune') and ((not isinstance(self.side_layers[layer_idx], str))
                                                          or self.side_layers[layer_idx] != 'separate')
            out = layer(x, module_t, sum_up_to_t)
            
        if self.training and layer_idx != len(self.layers) - 1 and dropout > 0:
            out = torch.nn.functional.dropout(out, dropout)
        return out
        
    def forward(self,
                x,
                t,
                dropout = 0):
        '''
        Compute output from model
        @param self:
        @param x: torch FloatTensor, input features
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @param dropout: float, probability each output entry after each layer/block is zero'd during training
        @return: torch FloatTensor, output
        '''
        assert dropout >= 0 and dropout < 1
        out = x
        for layer_idx in range(len(self.layers)):
            out = self._forward_layer(out, t, layer_idx, dropout)
        return out

    def get_activations_from_all_layers(self,
                                        x,
                                        t):
        '''
        Return the output from each block of the model
        @param self:
        @param x: torch FloatTensor
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: list of torch FloatTensor
        '''
        outputs = []
        out = x
        for layer_idx in range(len(self.layers)):
            out = self._forward_layer(out, t, layer_idx)
            outputs.append(out)
        return outputs

    def freeze_params_up_to_time(self,
                                 t):
        '''
        Set require grad to False for parameters at all time steps up to and including t
        @param self:
        @param t: int, time step to freeze up to
        @return: None
        '''
        assert t >= 0 and t < self.num_time_steps
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].freeze_params_up_to_time(self.module_path[t, layer_idx])
        
    def ablate_side_modules(self):
        '''
        Set side modules to have no effect on output
        @param self:
        @return: None
        '''
        assert self.mode == 'side_tune'
        for block_idx in range(self.num_blocks):
            self.blocks[block_idx].ablate_modules_after_time0()
    
    def ablate_adapters(self):
        '''
        Set low-rank adapters to have no effect on output
        @param self:
        @return: None
        '''
        assert self.mode == 'low_rank_adapt'
        for block_idx in range(self.num_blocks):
            self.blocks[block_idx].ablate_adapters()
        
    def load_partial_state_dict(self,
                                pretrained_model          = None,
                                use_partial_to_init_final = False,
                                all_modules               = False):
        '''
        Load state from a model into initial modules of this joint model.
        May load from a joint model with at most the same number of modules as this model
        or from a resnet model into either the initial time step or all time steps of this model.
        Additional time steps in this model are unaffected.
        @param self:
        @param pretrained_model: joint_block_model
                                 to load weights from,
                                 if None will load ImageNet pre-trained resnet18 from PyTorch,
                                 must be same architecture as this model
        @param use_partial_to_init_final: bool, whether to use module at time T - 1 to initialize module at T
        @param all_modules: bool, load resnet18 weights into ALL separate modules
        @return: None
        '''
        if pretrained_model is not None:
            assert pretrained_model.num_classes      == self.num_classes
            assert pretrained_model.num_blocks       == self.num_blocks
            assert pretrained_model.model_properties == self.model_properties
        else:
            pretrained_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True)
            assert self.num_blocks                           == 4
            assert self.model_properties['layers_per_block'] == [2, 2, 2, 2]
            assert self.model_properties['input_dims']       == [64, 64, 128, 256]
            assert self.model_properties['output_dims']      == [64, 128, 256, 512]
            assert self.model_properties['strides']          == [1, 2, 2, 2]
        if isinstance(pretrained_model, joint_block_model):
            if use_partial_to_init_final:
                assert pretrained_model.num_time_steps == self.num_time_steps - 1
                assert self.mode == 'separate'
            else:
                assert pretrained_model.num_time_steps <= self.num_time_steps
            assert pretrained_model.mode == self.mode
            if self.mode == 'low_rank_adapt':
                assert pretrained_model.adapter_ranks == self.adapter_ranks
                assert pretrained_model.adapter_mode  == self.adapter_mode
            if self.mode == 'side_tune':
                assert pretrained_model.side_layers  == self.side_layers
        if all_modules:
            assert not isinstance(pretrained_model, joint_block_model)
            assert self.mode == 'separate'
        for layer_idx in range(len(self.layers)):
            if self.mode == 'low_rank_adapt' and self.adapter_ranks[layer_idx] > 0:
                # use_partial_to_init_final and all_modules arguments not available for low-rank modules
                self.layers[layer_idx].load_partial_state_dict(pretrained_model.layers[layer_idx])
            else:
                self.layers[layer_idx].load_partial_state_dict(pretrained_model.layers[layer_idx],
                                                               use_partial_to_init_final = use_partial_to_init_final,
                                                               all_modules               = all_modules)
    
    def compute_fisher_info(self):
        '''
        Compute Fisher info for each parameter
        @param self:
        @return: dict mapping str for layer name to list over time of dicts mapping str parameter name 
                 to torch FloatTensor containing Fisher info
        '''
        assert self.mode == 'separate'
        fisher_infos = {self.layer_names[layer_idx]: self.layers[layer_idx].compute_fisher_info()
                        for layer_idx in range(len(self.layers))}
        return fisher_infos
        
    def compute_adjacent_param_norm(self,
                                    norm                = 'l2',
                                    weight_by_time_step = False,
                                    fisher_info         = defaultdict(lambda: None)):
        '''
        Compute sum of squared L2 norm or sum of L1 norm
        of difference between parameters at adjacent time steps for all separate layers
        For side tuning or low-rank adaptation, only sum differences in input convolution and fully connected layers
        @param self:
        @param norm: str, l1 or l2
        @param weight_by_time_step: bool, if True weight adjacent reg by 1/(T-t)
        @param fisher_info: dict mapping str for layer name to list over time of dicts mapping str parameter name 
                            to torch FloatTensor containing Fisher info,
                            used to weight adjacent parameter norms when computing sum,
                            default: weight 1 for each parameter
        @return: torch FloatTensor
        '''
        assert norm in {'l1', 'l2'}
        if self.mode == 'separate':
            layers_to_compute = range(len(self.layers))
        else:
            layers_to_compute = [0, len(self.layers) - 1]
        adjacent_param_norms \
            = [self.layers[layer_idx].compute_adjacent_param_norm_at_each_time_step(norm,
                                                                                    fisher_info[self.layer_names[layer_idx]])
               for layer_idx in layers_to_compute]
        
        adj_norm = 0.
        for t in range(self.num_time_steps - 1):
            if weight_by_time_step:
                t_weight = 1./(self.num_time_steps - t)
            else:
                t_weight = 1.
            
            for layer_to_compute_idx in range(len(layers_to_compute)):
                layer_idx = layers_to_compute[layer_to_compute_idx]
                if self.module_path[t + 1, layer_idx] - self.module_path[t, layer_idx] == 1:
                    adj_norm += t_weight * adjacent_param_norms[layer_to_compute_idx][self.module_path[t, layer_idx]]
        return adj_norm

    def compute_param_norm(self,
                           norm                = 'l2',
                           weight_by_time_step = False,
                           weight_for_efficient_modules = 1):
        '''
        Compute sum of squared L2 norm or sum of L1 norm of parameters at all time steps
        @param self:
        @param norm: str, l1 or l2
        @param weight_by_time_step: bool, if True weight reg by 1/(t+1)
        @param weight_for_efficient_modules: float, multiply norm of parameters in side modules or low-rank adapters
                                             by this constant, only applied if not weighting by time step
        @return: torch FloatTensor
        '''
        param_norms_over_time = [layer.compute_param_norm_at_each_time_step(norm)
                                 for layer in self.layers]
        
        norm = 0.
        for t, layer_idx in product(range(self.num_time_steps), range(len(self.layers))):
            if weight_by_time_step:
                layer_t_weight = 1./(t+1)
            else:
                if self.module_path[t, layer_idx] != 0 and ((self.mode == 'low_rank_adapt' and self.adapter_ranks[layer_idx] > 0)
                                                             or (self.mode == 'side_tune' 
                                                                 and not isinstance(self.side_layers[layer_idx], str))):
                    layer_t_weight = weight_for_efficient_modules
                else:
                    layer_t_weight = 1.
            
            norm += layer_t_weight * param_norms_over_time[layer_idx][self.module_path[t, layer_idx]]

        return norm
    
    def get_param_vec(self,
                      layer_idx,
                      time_idx):
        '''
        Get all parameters in layer at a particular time step flattened and concatenated
        @param self:
        @param layer_idx: int, index of layer to get
        @param time_idx: int, index of time step to get
        @return: np array
        '''
        assert layer_idx >= 0 and layer_idx < len(self.layers)
        assert time_idx  >= 0 and time_idx  < self.num_time_steps
        assert self.mode != 'side_tune'
        return self.layers[layer_idx].get_param_vec(self.module_path[time_idx, layer_idx])

    def interpolate_weights(self,
                            t,
                            interpolation_val):
        '''
        Interpolate model between time t - 1 and time t
        If joint model has separate modules, return a block model with weights set to (1 - val) * (prev module) + val * (curr module)
        If joint model has side modules, return a joint model with the side modules at time t set to val * (side module)
        @param self:
        @param t: int, time step
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: block_model or joint_block_model
        '''
        assert self.model_type in {'ResNet', 'DenseNet', 'ConvNet'}
        assert self.mode in {'separate', 'side_tune'}
        assert t > 0 and t < self.num_time_steps
        assert interpolation_val > 0 and interpolation_val < 1

        if self.mode == 'separate':
            interpolated_layers = []
            for layer_idx, layer in enumerate(self.layers):
                if self.module_path[t - 1, layer_idx] != self.module_path[t, layer_idx]:
                    prev_layer = layer[self.module_path[t-1, layer_idx]]
                    curr_layer = layer[self.module_path[t, layer_idx]]
                    interpolated_layers.append(type(prev_layer).interpolate_weights(prev_layer,
                                                                                    curr_layer,
                                                                                    interpolation_val))
                else:
                    # no change so nothing to interpolate
                    interpolated_layers.append(layer[self.module_path[t, layer_idx]])

            if self.model_type == 'ResNet':
                return resnet_with_dropout.make_model_with_preset_blocks(self.model_properties,
                                                                         self.num_classes,
                                                                         interpolated_layers)
            if self.model_type == 'DenseNet':
                return densenet_with_dropout.make_model_with_preset_blocks(self.model_properties,
                                                                           self.num_classes,
                                                                           interpolated_layers)
            return convnet_with_dropout.make_model_with_preset_blocks(self.model_properties,
                                                                      self.num_classes,
                                                                      interpolated_layers)

        interpolated_model = deepcopy(self)
        for layer_idx, layer in enumerate(self.layers):
            if self.module_path[t - 1, layer_idx] != self.module_path[t, layer_idx]:
                interpolated_model.layers[layer_idx][self.module_path[t, layer_idx]].shrink_weights(interpolation_val)
        return interpolated_model