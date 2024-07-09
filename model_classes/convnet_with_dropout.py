import torch
import torchvision
import numpy as np

from block_model import block_model
from pytorch_module_wrapper_classes import (
    module_wrapper,
    linear_wrapper, 
    conv2d_wrapper, 
    relu_wrapper,
    batchnorm2d_wrapper, 
    sequential_wrapper,
    flatten_wrapper,
    maxpool2d_wrapper,
    adaptive_avgpool2d_wrapper
)

class convnet_with_dropout(block_model):
    
    def __init__(self,
                 model_properties = {'input_dims'           : [64, 64, 128, 256],
                                     'output_dims'          : [64, 128, 256, 512],
                                     'strides'              : [1, 2, 2, 2],
                                     'num_blocks'           : 4},
                 num_classes       = 10):
        '''
        Initialize a resnet model with dropout added after input convolution and each residual block
        @param self:
        @param model_properties: dict mapping str to list of ints, list over blocks
                                 properties to include: input_dims, output_dims, strides, num_blocks
        @param num_classes: int, number of output classes
        @return: None
        '''
        # create blocks
        super(convnet_with_dropout, self).__init__(model_properties,
                                                   num_classes)
        
        # create the other layers
        self.conv1       = self._make_input_conv_layer()
        self.fc          = self._make_fc_layer()
        
        # put the layers together in order
        self.layers      = torch.nn.ModuleList([self.conv1])
        self.layers.extend(self.blocks)
        self.layers.append(self.fc)
        self.layer_names = ['conv1'] + self.block_names + ['fc']
        self.model_type  = 'ConvNet'
    
    def _make_input_conv_layer(self):
        '''
        Make input convolution layer + batch norm 2d layer
        @param self:
        @return: sequential_wrapper
        '''
        return sequential_wrapper(conv2d_wrapper(3,
                                                 self.model_properties['input_dims'][0],
                                                 kernel_size = 7,
                                                 stride      = 2,
                                                 padding     = 3,
                                                 bias        = False),
                                  batchnorm2d_wrapper(self.model_properties['input_dims'][0]),
                                  relu_wrapper(inplace = True),
                                  maxpool2d_wrapper(kernel_size=3, stride=2, padding=1))
    
    def _make_block(self,
                    block_idx):
        '''
        Make a block with only a convolution layer and a batch norm layer
        @param self:
        @param block_idx: index of block, determines dimensions and stride
        @return: sequential_wrapper
        '''
        assert block_idx >= 0 and block_idx < self.num_blocks
        return sequential_wrapper(conv2d_wrapper(self.model_properties['input_dims'][block_idx], 
                                                 self.model_properties['output_dims'][block_idx], 
                                                 kernel_size = 3, 
                                                 stride      = self.model_properties['strides'][block_idx], 
                                                 padding     = 1,
                                                 bias        = False),
                                  batchnorm2d_wrapper(self.model_properties['output_dims'][block_idx]))
    
    def _make_fc_layer(self):
        '''
        Make fully connected head: adaptive_avgpool2d_wrapper -> flatten_wrapper -> linear_wrapper
        @param: self
        @return: sequential_wrapper
        '''
        return sequential_wrapper(adaptive_avgpool2d_wrapper((1, 1)),
                                  flatten_wrapper(),
                                  linear_wrapper(self.model_properties['output_dims'][-1], self.num_classes))

    @classmethod
    def make_model_with_preset_blocks(cls,
                                      model_properties,
                                      num_classes,
                                      layers):
        '''
        Create a model with pre-set weights
        @param cls:
        @param model_properties: dict mapping str to list of ints, list over blocks
                                 properties to include: input_dims, output_dims, strides, num_blocks
        @param num_classes: int, number of output classes
        @param layers: list of layers that match the model specification
        @return: convnet_with_dropout
        '''
        new_model = cls(model_properties,
                        num_classes)
        new_model.layers = torch.nn.ModuleList(layers)
        new_model.conv1  = layers[0]
        new_model.fc     = layers[-1]
        new_model.blocks = torch.nn.ModuleList(layers[1:-1])
        return new_model