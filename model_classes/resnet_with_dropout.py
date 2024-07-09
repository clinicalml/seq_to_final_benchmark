import torch
import torchvision
import numpy as np
from copy import deepcopy

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

class basicblock_wrapper(torchvision.models.resnet.BasicBlock, module_wrapper):
        
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
            param_norm = torch.sum(torch.square(self.conv1.weight)) + torch.sum(torch.square(self.conv2.weight))
        else:
            param_norm = torch.sum(torch.abs(self.conv1.weight)) + torch.sum(torch.abs(self.conv2.weight))
        if self.downsample is not None:
            param_norm += self.downsample.compute_param_norm(norm)
        return param_norm
    
    def __getitem__(self, idx):
        '''
        Allow basicblock_wrapper to function like a module list
        '''
        modules = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]
        if self.downsample is not None:
            modules.append(self.downsample)
        return modules[idx]
        
    def __len__(self):
        if self.downsample is None:
            return 5
        return 6
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        if torch.cuda.is_available():
            weights = [self.conv1.weight.cpu(), self.conv2.weight.cpu()]
        else:
            weights = [self.conv1.weight, self.conv2.weight]
        return np.concatenate([weight.detach().numpy().flatten() for weight in weights])

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: basicblock_wrapper
        @param layer2: basicblock_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: basicblock_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        assert interpolation_val > 0 and interpolation_val < 1
        interpolated_layer = deepcopy(layer1)
        interpolated_layer.conv1.weight = torch.nn.Parameter((1. - interpolation_val) * layer1.conv1.weight
                                                             + interpolation_val * layer2.conv1.weight)
        interpolated_layer.conv2.weight = torch.nn.Parameter((1. - interpolation_val) * layer1.conv2.weight
                                                             + interpolation_val * layer2.conv2.weight)
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
        self.conv1.weight = torch.nn.Parameter(self.conv1.weight * shrink_factor)
        self.conv2.weight = torch.nn.Parameter(self.conv2.weight * shrink_factor)

class resnet_with_dropout(block_model):
    
    def __init__(self,
                 model_properties = {'num_layers_per_block' : [2, 2, 2, 2],
                                     'input_dims'           : [64, 64, 128, 256],
                                     'output_dims'          : [64, 128, 256, 512],
                                     'strides'              : [1, 2, 2, 2],
                                     'num_blocks'           : 4},
                 num_classes       = 10):
        '''
        Initialize a resnet model with dropout added after input convolution and each residual block
        @param self:
        @param model_properties: dict mapping str to list of ints, list over blocks
                                 properties to include: num_layers_per_block, input_dims, output_dims, strides, num_blocks
        @param num_classes: int, number of output classes
        @return: None
        '''
        # create blocks
        super(resnet_with_dropout, self).__init__(model_properties,
                                                  num_classes)
        
        # create the other layers
        self.conv1       = self._make_input_conv_layer()
        self.fc          = self._make_fc_layer()
        
        # put the layers together in order
        self.layers      = torch.nn.ModuleList([self.conv1])
        self.layers.extend(self.blocks)
        self.layers.append(self.fc)
        self.layer_names = ['conv1'] + self.block_names + ['fc']
        self.model_type  = 'ResNet'
    
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
        Make a residual block
        @param self:
        @param block_idx: index of residual block, determines dimensions, strides, and whether to include a downsample block
        @return: residual block
        '''
        assert block_idx >= 0 and block_idx < self.num_blocks
        if self.model_properties['strides'][block_idx] == 1 \
        and self.model_properties['input_dims'][block_idx] == self.model_properties['output_dims'][block_idx]:
            downsample = None
        else:
            downsample = sequential_wrapper(conv2d_wrapper(self.model_properties['input_dims'][block_idx], 
                                                           self.model_properties['output_dims'][block_idx], 
                                                           kernel_size = 1, 
                                                           stride      = self.model_properties['strides'][block_idx], 
                                                           bias        = False),
                                            batchnorm2d_wrapper(self.model_properties['output_dims'][block_idx]))

        layers = [basicblock_wrapper(self.model_properties['input_dims'][block_idx], 
                                     self.model_properties['output_dims'][block_idx], 
                                     self.model_properties['strides'][block_idx], 
                                     downsample)]

        for layer_idx in range(1, self.model_properties['num_layers_per_block'][block_idx]):
            layers.append(basicblock_wrapper(self.model_properties['output_dims'][block_idx], 
                                             self.model_properties['output_dims'][block_idx]))
        
        return sequential_wrapper(*layers)
    
    def _make_fc_layer(self):
        '''
        Make fully connected head: adaptive_avgpool2d_wrapper -> flatten_wrapper -> linear_wrapper
        @param: self
        @return: sequential_wrapper
        '''
        return sequential_wrapper(adaptive_avgpool2d_wrapper((1, 1)),
                                  flatten_wrapper(),
                                  linear_wrapper(self.model_properties['output_dims'][-1], self.num_classes))
    
    def forward(self,
                x,
                dropout = 0):
        '''
        Compute output from model
        @param self:
        @param x: torch FloatTensor, input
        @param dropout: float, probability each output entry after each layer/block is zero'd during training
        @return: torch FloatTensor, output
        '''
        assert dropout >= 0 and dropout < 1
        out = self.conv1(x)
        if self.training and dropout > 0:
            out = torch.nn.functional.dropout(out, dropout)
        for block in self.blocks:
            out = block(out)
            if self.training and dropout > 0:
                out = torch.nn.functional.dropout(out, dropout)
        out = self.fc(out)
        return out

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
        @return: resnet_with_dropout
        '''
        new_model = cls(model_properties,
                        num_classes)
        new_model.layers = torch.nn.ModuleList(layers)
        new_model.conv1  = layers[0]
        new_model.fc     = layers[-1]
        new_model.blocks = torch.nn.ModuleList(layers[1:-1])
        return new_model