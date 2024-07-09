import torch
import torchvision

from joint_block_model import joint_block_model
from densenet_with_dropout import (
    densenet_with_dropout,
    denseblock_wrapper,
    transition_wrapper
)

from pytorch_module_wrapper_classes import (
    module_wrapper,
    linear_wrapper,
    batchnorm2d_wrapper,
    avgpool2d_wrapper,
    adaptive_avgpool2d_wrapper,
    flatten_wrapper,
    relu_wrapper,
    module_list_over_time
)
from low_rank_module_classes import (
    low_rank_conv2d,
    low_rank_linear,
    low_rank_sequential,
    low_rank_module_list
)

class low_rank_denselayer(module_wrapper, low_rank_sequential, torchvision.models.densenet._DenseLayer):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 num_input_features,
                 growth_rate,
                 bn_size):
        '''
        Create a low-rank dense layer: list of (batchnorm2d_wrappers -> relu_wrapper -> low_rank_conv2d) x 2
        @param self:
        @param num_low_rank_adapters: int, number of adapters, each corresponds to a new time step
        @param adapter_rank: int, rank of adapters at each time step
        @param adapter_mode: str, multiply or add
        @param num_input_features: int, number of input dimensions for first convolution layer 
        @param growth_rate: int, output size from second convolution layer
        @param bn_size: int, output size from first convolution layer is bn_size * growth_rate
        @return: None
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters > 0:
            assert adapter_rank > 0
            assert adapter_mode in {'multiply', 'add'}
        assert num_input_features > 0
        assert growth_rate > 0
        assert bn_size > 0
        module_wrapper.__init__(self)
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        self.num_input_features    = num_input_features
        self.growth_rate           = growth_rate
        self.bn_size               = bn_size
        
        self.norm1 = module_list_over_time([batchnorm2d_wrapper(num_input_features)
                                            for i in range(num_low_rank_adapters + 1)])
        self.relu1 = relu_wrapper(inplace = True)
        self.conv1 = low_rank_conv2d(num_low_rank_adapters,
                                     adapter_rank,
                                     adapter_mode,
                                     num_input_features, 
                                     bn_size * growth_rate,
                                     kernel_size = 1,
                                     stride      = 1,
                                     padding     = 1,
                                     bias        = False)
        self.norm2 = module_list_over_time([batchnorm2d_wrapper(bn_size * growth_rate)
                                            for i in range(num_low_rank_adapters + 1)])
        self.relu2 = relu_wrapper(inplace = True)
        self.conv2 = low_rank_conv2d(num_low_rank_adapters,
                                     adapter_rank,
                                     adapter_mode,
                                     bn_size * growth_rate,
                                     growth_rate,
                                     kernel_size = 3,
                                     stride      = 1,
                                     bias        = False)
        self.layers = [self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2]
        low_rank_sequential.__init__(self,
                                     *self.layers)

class low_rank_denseblock(low_rank_module_list, denseblock_wrapper, torch.nn.ModuleDict):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 num_layers,
                 num_input_features,
                 bn_size,
                 growth_rate):
        '''
        Create a low-rank dense block containing many dense layers
        @param self:
        @param num_low_rank_adapters: int, number of adapters, each corresponds to a new time step
        @param adapter_rank: int, rank of adapters at each time step
        @param adapter_mode: str, multiply or add
        @param num_layers: int, number of dense layers
        @param num_input_features: int, number of input dimensions for first convolution layer 
        @param growth_rate: int, output size from second convolution layer
        @param bn_size: int, output size from first convolution layer is bn_size * growth_rate
        @return: None
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters > 0:
            assert adapter_rank > 0
            assert adapter_mode in {'multiply', 'add'}
        assert num_layers > 0
        assert num_input_features > 0
        assert growth_rate > 0
        assert bn_size > 0
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        self.num_layers            = num_layers
        self.num_input_features    = num_input_features
        self.growth_rate           = growth_rate
        self.bn_size               = bn_size
        
        torch.nn.ModuleDict.__init__(self)
        self.layers = []
        for i in range(num_layers):
            layer = low_rank_denselayer(num_low_rank_adapters,
                                        adapter_rank,
                                        adapter_mode,
                                        num_input_features + i * growth_rate,
                                        growth_rate,
                                        bn_size)
            self.layers.append(layer)
            self.add_module('denselayer%d' % (i + 1), layer)
        low_rank_module_list.__init__(self,
                                      *self.layers)
    
    def forward(self,
                x,
                t):
        '''
        Compute the output at time t
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        features = [x]
        for name, layer in self.items():
            new_features = layer(features, t)
            features.append(new_features)
        return torch.cat(features, 1)
    
class low_rank_transition(low_rank_sequential, transition_wrapper):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 num_input_features,
                 num_output_features):
        '''
        Create a low-rank transition layer: list of batchnorm2d_wrappers -> relu_wrapper -> low_rank_conv2d -> avgpool2d_wrapper
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters > 0:
            assert adapter_rank > 0
            assert adapter_mode in {'multiply', 'add'}
        assert num_input_features > 0
        assert num_output_features > 0
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        self.num_input_features    = num_input_features
        self.num_output_features   = num_output_features
        
        self.norm = module_list_over_time([batchnorm2d_wrapper(num_input_features)
                                           for _ in range(num_low_rank_adapters + 1)])
        self.relu = relu_wrapper(inplace = True)
        self.conv = low_rank_conv2d(num_low_rank_adapters,
                                    adapter_rank,
                                    adapter_mode,
                                    num_input_features,
                                    num_output_features,
                                    kernel_size = 1,
                                    stride      = 1,
                                    bias        = False)
        self.pool = avgpool2d_wrapper(kernel_size = 2,
                                      stride      = 2)
        
        self.layers = [self.norm, self.relu, self.conv, self.pool]
        low_rank_sequential.__init__(self,
                                     *self.layers)

class joint_densenet_with_dropout(joint_block_model, densenet_with_dropout):
    
    def __init__(self,
                 num_time_steps,
                 separate_layers,
                 model_properties = {'num_blocks'          : 4,
                                     'num_layers_per_block': [6, 12, 24, 16]},
                 num_classes       = 10,
                 mode              = 'separate',
                 adapter_ranks     = [0, 0, 0, 0, 0, 0],
                 adapter_mode      = None,
                 side_layers       = ['separate',[],[],[],[],'separate']):
        '''
        Initialize a joint densenet model that has some shared layers across time steps
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
        @param model_properties: dict mapping str to int or list of int, must contain num_blocks
                                 and num_layers_per_block for number of dense layers (BatchNorm2d -> ReLU -> Conv2d) x 2 
                                 in each dense block
        @param num_classes: int, number of output classes
        @param mode: str, separate to create new layers at separate time steps,
                     side_tune to add convolution side modules instead of dense blocks at each time step,
                     low_rank_adapt to multiply or add to weights with low-rank adapters
        @param adapter_ranks: list of int, rank of adapters in each layer
        @param adapter_mode: str, multiply or add
        @param side_layers: list of (str or lists of int), number of output channels 
                            for each of the intermediate convolution layers in side modules,
                            outer list over layers in model
                            str option is "block" for side module to be a residual block,
                            str option is "separate" for conv1 and fc layers
        @return: None
        '''
        self.model_type = 'DenseNet'
        self.num_layers_before_blocks = 1
        
        # create blocks
        joint_block_model.__init__(self,
                                   num_time_steps,
                                   separate_layers,
                                   model_properties,
                                   num_classes,
                                   mode,
                                   adapter_ranks,
                                   adapter_mode,
                                   side_layers)
        
        # create other layers and put layers together in order
        self._make_input_conv_and_fc_head()
        
    def _make_low_rank_input_conv_layer(self):
        '''
        Make low-rank input convolution layer: low_rank_conv2d -> list of batchnorm2d_wrappers 
                                               -> relu_wrapper -> maxpool2d_wrapper
        @param self:
        @return: low_rank_sequential
        '''
        assert self.mode == 'low_rank_adapt'
        return low_rank_sequential(low_rank_conv2d(self.module_path[-1, 0],
                                                   self.adapter_ranks[0],
                                                   self.adapter_mode,
                                                   3,
                                                   64,
                                                   kernel_size = 7,
                                                   stride      = 2,
                                                   padding     = 3,
                                                   bias        = False),
                                    batchnorm2d_wrapper(64),
                                    relu_wrapper(inplace = True),
                                    maxpool2d_wrapper(kernel_size=3, stride=2, padding=1))
    
    def _make_low_rank_block(self,
                             block_idx):
        '''
        Make a low-rank dense block and low-rank transition layer
        @param self:
        @param block_idx: int, index of dense block, determines number of layers, input size, 
                          and whether to include a transition layer
        @return: low_rank_sequential or low_rank_denseblock
        '''
        assert self.mode == 'low_rank_adapt'
        
        block_num_features = 64
        for idx in range(block_idx):
            block_num_features += 32 * self.model_properties['num_layers_per_block'][idx]
            block_num_features  = block_num_features // 2
        
        denseblock = low_rank_denseblock(self.block_module_path[-1, block_idx],
                                         self.adapter_ranks[self.num_layers_before_blocks + block_idx],
                                         self.adapter_mode,
                                         num_layers         = self.model_properties['num_layers_per_block'][block_idx],
                                         num_input_features = block_num_features,
                                         bn_size            = 4,
                                         growth_rate        = 32,
                                         drop_rate          = 0)
        if block_idx != self.num_blocks - 1:
            transition_num_features = block_num_features + 32 * self.model_properties['num_layers_per_block'][block_idx]
            transition = low_rank_transition(self.block_module_path[-1, block_idx],
                                             self.adapter_ranks[self.num_layers_before_blocks + block_idx],
                                             self.adapter_mode,
                                             transition_num_features,
                                             transition_num_features // 2)
            return low_rank_sequential(denseblock, transition)
        return denseblock
    
    def _make_low_rank_fc(self):
        '''
        Make a low-rank fully connected head: list of batchnorm2d_wrappers -> relu_wrapper -> adaptive_avgpool2d_wrapper 
                                              -> flatten_wrapper -> low_rank_linear
        @param self:
        @return: low_rank_sequential
        '''
        assert self.mode == 'low_rank_adapt'
        # compute dimensions for final layers
        curr_num_features = 64
        for block_idx in range(self.num_blocks):
            curr_num_features += 32 * self.model_properties['num_layers_per_block'][block_idx]
            if block_idx != self.num_blocks - 1:
                curr_num_features = curr_num_features // 2
        
        # create final layers
        return low_rank_sequential(module_list_over_time([batchnorm2d_wrapper(curr_num_features)
                                                          for _ in range(self.module_path[-1, -1] + 1)]),
                                   relu_wrapper(inplace = True),
                                   adaptive_avgpool2d_wrapper((1, 1)),
                                   flatten_wrapper(),
                                   low_rank_linear(self.module_path[-1, -1],
                                                   self.adapter_ranks[-1],
                                                   self.adapter_mode,
                                                   curr_num_features,
                                                   self.num_classes))
    
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
        assert not isinstance(self.side_layers[block_idx + self.num_layers_before_blocks], str)
        block_num_features = 64
        for idx in range(block_idx):
            block_num_features += 32 * self.model_properties['num_layers_per_block'][idx]
            block_num_features  = block_num_features // 2
        next_block_num_features = block_num_features + 32 * self.model_properties['num_layers_per_block'][block_idx]
        if block_idx != self.num_blocks - 1:
            # account for transition layer
            next_block_num_features = next_block_num_features // 2
            starting_stride         = 2
        else:
            starting_stride         = 1
        side_module_properties  = {'input_dims' : [block_num_features],
                                   'kernel_size': [3],
                                   'stride'     : [starting_stride],
                                   'padding'    : [1]}
        if len(self.side_layers[block_idx + self.num_layers_before_blocks]) == 0:
            side_module_properties['output_dims'] = [next_block_num_features]
            return side_module_properties
        
        side_module_properties['output_dims'] = [self.side_layers[block_idx + self.num_layers_before_blocks][0]]
                                  
        num_orig_params = len(self._make_block(block_idx).get_param_vec())
        num_side_params = self.model_properties['input_dims'][block_idx] \
                        * self.side_layers[block_idx + self.num_layers_before_blocks][0] * 9
        for layer_idx in range(len(self.side_layers[block_idx + self.num_layers_before_blocks])):
            if layer_idx == len(self.side_layers[block_idx + self.num_layers_before_blocks]) - 1:
                num_output_channels = next_block_num_features
            else:
                num_output_channels = self.side_layers[block_idx + self.num_layers_before_blocks][layer_idx + 1]
            num_side_params += self.side_layers[block_idx + self.num_layers_before_blocks][layer_idx] * num_output_channels * 9
            side_module_properties['input_dims'].append(self.side_layers[block_idx + self.num_layers_before_blocks][layer_idx])
            side_module_properties['output_dims'].append(num_output_channels)
            side_module_properties['kernel_size'].append(3)
            side_module_properties['stride'].append(1)
            side_module_properties['padding'].append(1)
        assert num_side_params < num_orig_params
        return side_module_properties