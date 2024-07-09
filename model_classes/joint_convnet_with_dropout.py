from joint_block_model import joint_block_model
from convnet_with_dropout import convnet_with_dropout

from pytorch_module_wrapper_classes import (
    batchnorm2d_wrapper,
    module_list_over_time
)
from low_rank_module_classes import (
    low_rank_conv2d,
    low_rank_linear,
    low_rank_sequential
)

class joint_convnet_with_dropout(joint_block_model, convnet_with_dropout):
    
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
                     side_tune to add output from conv layer + batch norm side module at each time step,
                     low_rank_adapt to multiply or add to weights with low-rank adapters
        @param adapter_ranks: list of int, rank of adapters in each layer, must be more parameter-efficient than original module
        @param adapter_mode: str, multiply or add
        @param side_layers: list of str, "block" or "separate" for each layer,
                            side layers for conv net are same size as original modules,
                            "block" means add to original output,
                            "separate" means don't add to original output,
                            unlike other block_models, block layers can be separate rather than side modules
        @return: None
        '''
        self.model_type = 'ConvNet'
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
        
        return low_rank_sequential(low_rank_conv2d(num_adapters,
                                                    self.adapter_ranks[block_idx + self.num_layers_before_blocks],
                                                    self.adapter_mode,
                                                    self.model_properties['input_dims'][block_idx], 
                                                    self.model_properties['output_dims'][block_idx], 
                                                    kernel_size = 3, 
                                                    stride      = self.model_properties['strides'][block_idx], 
                                                    padding     = 1,
                                                    bias        = False),
                                   module_list_over_time([batchnorm2d_wrapper(self.model_properties['output_dims'][block_idx])
                                                          for i in range(num_adapters + 1)]))
    
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
        All convnet side modules are the same dimensions as the original modules, so this method should never be called
        @param self:
        @param block_idx: int, index of block to compute side module properties for
        '''
        assert False