import torchvision

from joint_block_model import joint_block_model
from resnet_with_dropout import resnet_with_dropout

from pytorch_module_wrapper_classes import (
    module_wrapper,
    linear_wrapper,
    relu_wrapper,
    batchnorm2d_wrapper,
    module_list_over_time
)
from low_rank_module_classes import (
    low_rank_conv2d,
    low_rank_linear,
    low_rank_sequential,
    low_rank_module_list
)

class low_rank_basic_block(low_rank_module_list, torchvision.models.resnet.BasicBlock):
    
    def __init__(self,
                 num_low_rank_adapters,
                 adapter_rank,
                 adapter_mode,
                 inplanes,
                 planes,
                 stride     = 1,
                 downsample = None):
        '''
        Initialize a residual BasicBlock with low_rank_conv2d layers
        @param self:
        @param num_low_rank_adapters: int, number of adapters, each corresponds to a new time step
        @param adapter_rank: int, rank of adapters at each time step
        @param adapter_mode: str, multiply or add
        @param inplanes: int, number of input dimensions for first convolution layer 
        @param planes: int, number of dimensions for all other convolution layers
        @param stride: int, stride for first convolution layer
        @param downsample: low_rank_sequential, downsample layer
        @return: None
        '''
        assert num_low_rank_adapters >= 0
        if num_low_rank_adapters > 0:
            assert adapter_rank > 0
            assert adapter_mode in {'multiply', 'add'}
        assert inplanes > 0
        assert planes > 0
        assert stride > 0
        low_rank_module_list.__init__(self, [])
        
        self.num_low_rank_adapters = num_low_rank_adapters
        self.adapter_rank          = adapter_rank
        self.adapter_mode          = adapter_mode
        self.conv1 = low_rank_conv2d(num_low_rank_adapters,
                                     adapter_rank,
                                     adapter_mode,
                                     inplanes, 
                                     planes,
                                     kernel_size = 3,
                                     stride      = stride,
                                     bias        = False,
                                     padding     = 1)
        self.bn1   = module_list_over_time([batchnorm2d_wrapper(planes) for i in range(num_low_rank_adapters + 1)])
        self.relu  = relu_wrapper(inplace = True)
        self.conv2 = low_rank_conv2d(num_low_rank_adapters,
                                     adapter_rank,
                                     adapter_mode,
                                     planes, 
                                     planes,
                                     kernel_size = 3,
                                     padding     = 1,
                                     bias        = False)
        self.bn2   = module_list_over_time([batchnorm2d_wrapper(planes) for i in range(num_low_rank_adapters + 1)])
        self.downsample = downsample
        
        self.append(self.conv1)
        self.append(self.bn1)
        self.append(self.relu)
        self.append(self.conv2)
        self.append(self.bn2)
        if downsample is not None:
            self.append(self.downsample)
        
    def forward(self,
                x,
                t):
        '''
        Compute output from low-rank basic block at particular time steps
        @param self:
        @param x: torch FloatTensor, input
        @param t: int or torch LongTensor, time step for all samples or sequence of time steps for each sample
        @return: torch FloatTensor, output
        '''
        identity = x
        
        out = self.conv1(x, t)
        out = self.bn1(out, t)
        out = self.relu(out)
        out = self.conv2(out, t)
        out = self.bn2(out, t)
        
        if self.downsample is not None:
            identity = self.downsample(identity, t)
        
        out += identity
        out  = self.relu(out)
        
        return out

class joint_resnet_with_dropout(joint_block_model, resnet_with_dropout):
    
    def __init__(self,
                 num_time_steps,
                 separate_layers,
                 model_properties = {'num_blocks'           : 4,
                                     'num_layers_per_block' : [2, 2, 2, 2],
                                     'input_dims'           : [64, 64, 128, 256],
                                     'output_dims'          : [64, 128, 256, 512],
                                     'strides'              : [1, 2, 2, 2]},
                 num_classes       = 10,
                 mode              = 'separate',
                 adapter_ranks     = [0, 0, 0, 0, 0, 0],
                 adapter_mode      = None,
                 side_layers       = ['separate', [],[],[],[], 'separate']):
        '''
        Initialize a joint resnet model that has some shared layers across time steps
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
        @param model_properties: dict mapping str to list of int over layers, properties must include num_layers_per_block,
                                 input_dims, output_dims, and strides
        @param num_classes: int, number of output classes
        @param mode: str, separate to create new layers at separate time steps,
                     side_tune to add convolution side modules instead of residual blocks at each time step,
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
        self.model_type = 'ResNet'
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
        Make low-rank input convolution layer: low_rank_conv2d -> batchnorm2d_wrapper
        @param self:
        @return: low_rank_sequential
        '''
        assert self.mode == 'low_rank_adapt'
        return low_rank_sequential(low_rank_conv2d(self.module_path[-1, 0],
                                                   self.adapter_ranks[0],
                                                   self.adapter_mode,
                                                   3,
                                                   self.model_properties['input_dims'][0],
                                                   kernel_size = 7, 
                                                   stride      = 2,
                                                   padding     = 3, 
                                                   bias        = False),
                                   batchnorm2d_wrapper(self.model_properties['input_dims'][0]))
    
    def _make_low_rank_block(self,
                             block_idx):
        '''
        Make a low-rank residual block
        @param self:
        @param block_idx: int, index of residual block, determines dimensions, strides, and whether to include 
                          a downsample block
        @return: low_rank_sequential composed of low_rank_basic_blocks
        '''
        assert self.mode == 'low_rank_adapt'
        assert block_idx >= 0 and block_idx < self.num_blocks
        num_adapters = self.block_module_path[-1, block_idx]
        
        if block_idx == 0:
            downsample = None
        else:
            downsample \
                = low_rank_sequential(low_rank_conv2d(num_adapters,
                                                      self.adapter_ranks[block_idx + self.num_layers_before_blocks],
                                                      self.adapter_mode,
                                                      self.model_properties['input_dims'][block_idx], 
                                                      self.model_properties['output_dims'][block_idx], 
                                                      kernel_size = 1, 
                                                      stride      = self.model_properties['strides'][block_idx], 
                                                      bias        = False),
                                      module_list_over_time([batchnorm2d_wrapper(self.model_properties['output_dims'][block_idx])
                                                             for i in range(num_adapters + 1)]))

        layers = [low_rank_basic_block(num_adapters,
                                       self.adapter_ranks[block_idx + self.num_layers_before_blocks],
                                       self.adapter_mode,
                                       self.model_properties['input_dims'][block_idx], 
                                       self.model_properties['output_dims'][block_idx], 
                                       self.model_properties['strides'][block_idx], 
                                       downsample)]
        
        for layer_idx in range(1, self.model_properties['num_layers_per_block'][block_idx]):
            layers.append(low_rank_basic_block(num_adapters,
                                               self.adapter_ranks[block_idx + self.num_layers_before_blocks],
                                               self.adapter_mode,
                                               self.model_properties['output_dims'][block_idx], 
                                               self.model_properties['output_dims'][block_idx]))
        
        return low_rank_sequential(*layers)
    
    def _make_low_rank_fc(self):
        '''
        Make a low-rank fully connected layer
        @param self:
        @return: low_rank_linear
        '''
        assert self.mode == 'low_rank_adapt'
        return low_rank_linear(self.module_path[-1, -1],
                               self.adapter_ranks[-1],
                               self.adapter_mode,
                               self.model_properties['output_dims'][-1],
                               self.num_classes)
    
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
        side_module_properties = {'input_dims' : [self.model_properties['input_dims'][block_idx]],
                                  'kernel_size': [3],
                                  'stride'     : [self.model_properties['strides'][block_idx]],
                                  'padding'    : [1]}
        if len(self.side_layers[block_idx + self.num_layers_before_blocks]) == 0:
            side_module_properties['output_dims'] = [self.model_properties['output_dims'][block_idx]]
            return side_module_properties
        side_module_properties['output_dims'] = [self.side_layers[block_idx + self.num_layers_before_blocks][0]]
                                  
        num_orig_params = self.model_properties['input_dims'][block_idx] \
                        * self.model_properties['output_dims'][block_idx] * (9 if block_idx == 0 else 10) \
                        + self.model_properties['output_dims'][block_idx] \
                        * self.model_properties['output_dims'][block_idx] * 3 * 9
        num_side_params = self.model_properties['input_dims'][block_idx] \
                        * self.side_layers[block_idx + self.num_layers_before_blocks][0] * 9
        for layer_idx in range(len(self.side_layers[block_idx + self.num_layers_before_blocks])):
            if layer_idx == len(self.side_layers[block_idx + self.num_layers_before_blocks]) - 1:
                num_output_channels = self.model_properties['output_dims'][block_idx]
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