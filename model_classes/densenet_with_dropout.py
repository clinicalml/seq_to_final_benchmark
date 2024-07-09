import torch
import torchvision
import numpy as np
from copy import deepcopy

from block_model import block_model
from pytorch_module_wrapper_classes import (
    module_wrapper,
    linear_wrapper, 
    conv2d_wrapper,
    maxpool2d_wrapper,
    batchnorm2d_wrapper,
    sequential_wrapper,
    flatten_wrapper,
    adaptive_avgpool2d_wrapper,
    relu_wrapper
)

class denseblock_wrapper(torchvision.models.densenet._DenseBlock, module_wrapper):
    
    def compute_param_norm(self,
                           norm):
        '''
        Compute sum of squared L2 or sum of L1 norm of parameters in all dense layers in block
        Batch norm parameters are not included
        @param self
        @param norm: str, l1 or l2
        @return: torch FloatTensor
        '''
        assert norm in {'l1', 'l2'}
        param_norm = 0.
        for dense_layer_name in self._modules:
            dense_layer = self._modules[dense_layer_name]
            if norm == 'l2':
                param_norm += (torch.sum(torch.square(dense_layer.conv1.weight))
                               + torch.sum(torch.square(dense_layer.conv2.weight)))
            else:
                param_norm += (torch.sum(torch.abs(dense_layer.conv1.weight))
                               + torch.sum(torch.abs(dense_layer.conv2.weight)))
        return param_norm
    
    def __getitem__(self, idx):
        '''
        Allow basicblock_wrapper to function like a module list
        '''
        return self._modules['denselayer' + str(idx + 1)]
        
    def __len__(self):
        return len(self._modules)
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        weights = []
        for dense_layer_name in self._modules:
            dense_layer = self._modules[dense_layer_name]
            if torch.cuda.is_available():
                weights.extend([dense_layer.conv1.weight.cpu(), dense_layer.conv2.weight.cpu()])
            else:
                weights.extend([dense_layer.conv1.weight, dense_layer.conv2.weight])
        return np.concatenate([weight.detach().numpy().flatten() for weight in weights])

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: denseblock_wrapper
        @param layer2: denseblock_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: denseblock_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        interpolated_layer = deepcopy(layer1)
        for dense_layer_name in layer1._modules:
            dense_layer1 = layer1._modules[dense_layer_name]
            dense_layer2 = layer2._modules[dense_layer_name]
            conv1_weight1 = dense_layer1.conv1.weight
            conv1_weight2 = dense_layer2.conv1.weight
            interpolated_layer._modules[dense_layer_name].conv1.weight \
                = torch.nn.Parameter((1. - interpolation_val) * conv1_weight1 + interpolation_val * conv1_weight2)

            conv2_weight1 = dense_layer1.conv2.weight
            conv2_weight2 = dense_layer2.conv2.weight
            interpolated_layer._modules[dense_layer_name].conv2.weight \
                = torch.nn.Parameter((1. - interpolation_val) * conv2_weight1 + interpolation_val * conv2_weight2)
            
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
        for dense_layer_name in self._modules:
            self._modules[dense_layer_name].conv1.weight = torch.nn.Parameter(self._modules[dense_layer_name].conv1.weight * shrink_factor)
            self._modules[dense_layer_name].conv2.weight = torch.nn.Parameter(self._modules[dense_layer_name].conv2.weight * shrink_factor)
    
class transition_wrapper(torchvision.models.densenet._Transition, module_wrapper):
    
    def compute_param_norm(self,
                           norm):
        '''
        Compute sum of squared L2 or sum of L1 norm of parameters
        Batch norm parameters are not included
        @param self
        @param norm: str, l1 or l2
        @return: torch FloatTensor
        '''
        if norm == 'l2':
            return torch.sum(torch.square(self.conv.weight))
        return torch.sum(torch.abs(self.conv.weight))
    
    def __getitem__(self, idx):
        '''
        Allow basicblock_wrapper to function like a module list
        '''
        modules = [self.norm, self.relu, self.conv, self.pool]
        return modules[idx]
        
    def __len__(self):
        return 4
    
    def get_param_vec(self):
        '''
        Create a flat vector of all parameters flattened and concatenated together
        @param self:
        @return: np array
        '''
        if torch.cuda.is_available():
            return self.conv.weight.cpu().detach().numpy().flatten()
        return self.conv.weight.detach().numpy().flatten()

    @classmethod
    def interpolate_weights(cls,
                            layer1,
                            layer2,
                            interpolation_val):
        '''
        Set parameters to (1 - val) * layer1 + val * layer2
        @param cls:
        @param layer1: transition_wrapper
        @param layer2: transition_wrapper
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: transition_wrapper
        '''
        assert isinstance(layer1, cls)
        assert isinstance(layer2, cls)
        interpolated_layer = deepcopy(layer1)
        interpolated_layer.conv.weight = torch.nn.Parameter((1. - interpolation_val) * layer1.conv.weight + interpolation_val * layer2.conv.weight)
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
        self.conv.weight = torch.nn.Parameter(self.conv.weight * shrink_factor)

class densenet_with_dropout(block_model):
    
    def __init__(self,
                 model_properties = {'num_blocks': 4,
                                     'num_layers_per_block': [6, 12, 24, 16]},
                 num_classes      = 10):
        '''
        Initialize a densenet
        @param self:
        @param model_properties: dict mapping str to int or list of int, must contain num_blocks
                                 and num_layers_per_block for number of dense layers (BatchNorm2d -> ReLU -> Conv2d) x 2 
                                 in each dense block
        @param num_classes: int, number of label classes
        @return: None
        '''
        # create blocks
        super(densenet_with_dropout, self).__init__(model_properties,
                                                    num_classes)
        
        # create other layers
        self.conv1 = self._make_input_conv_layer()
        self.fc    = self._make_fc_layer()
        
        # put layers together in order
        self.layers = torch.nn.ModuleList([self.conv1])
        self.layers.extend(self.blocks)
        self.layers.append(self.fc)
        self.layer_names = ['conv1'] + self.block_names + ['fc']
        self.model_type = 'DenseNet'
                
    def _make_input_conv_layer(self):
        '''
        Make input conv 2d -> batch norm 2d -> relu -> max pool 2d
        @param self:
        @return: sequential_wrapper
        '''
        return sequential_wrapper(conv2d_wrapper(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                  batchnorm2d_wrapper(64),
                                  relu_wrapper(inplace = True),
                                  maxpool2d_wrapper(kernel_size=3, stride=2, padding=1))
    
    def _make_block(self,
                    block_idx):
        '''
        Make dense block_wrapper with number of dense layers for specified block
        If this is not the last dense block, add a transition_wrapper
        @param self:
        @param block_idx: int, index of block to create
        @return: sequential_wrapper or denseblock_wrapper
        '''
        block_num_features = 64
        for idx in range(block_idx):
            block_num_features += 32 * self.model_properties['num_layers_per_block'][idx]
            block_num_features  = block_num_features // 2
        denseblock = denseblock_wrapper(num_layers         = self.model_properties['num_layers_per_block'][block_idx],
                                        num_input_features = block_num_features,
                                        bn_size            = 4,
                                        growth_rate        = 32,
                                        drop_rate          = 0)
        if block_idx != self.num_blocks - 1:
            transition_num_features = block_num_features + 32 * self.model_properties['num_layers_per_block'][block_idx]
            transition = transition_wrapper(transition_num_features,
                                            transition_num_features // 2)
            return sequential_wrapper(denseblock, transition)
        return denseblock
    
    def _make_fc_layer(self):
        '''
        Make batchnorm2d_wrapper -> relu_wrapper -> adaptive_avgpool2d_wrapper -> flatten_wrapper -> linear_wrapper
        with input size based on blocks in model
        @param self:
        @return: sequential_wrapper
        '''
        # compute dimensions for final layers
        curr_num_features = 64
        for block_idx in range(self.num_blocks):
            curr_num_features += 32 * self.model_properties['num_layers_per_block'][block_idx]
            if block_idx != self.num_blocks - 1:
                curr_num_features = curr_num_features // 2
        
        # create final layers
        return sequential_wrapper(batchnorm2d_wrapper(curr_num_features),
                                  relu_wrapper(inplace = True),
                                  adaptive_avgpool2d_wrapper((1, 1)),
                                  flatten_wrapper(),
                                  linear_wrapper(curr_num_features, self.num_classes))

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
        @return: densenet_with_dropout
        '''
        new_model = cls(model_properties,
                        num_classes)
        new_model.layers = torch.nn.ModuleList(layers)
        new_model.conv1  = layers[0]
        new_model.fc     = layers[-1]
        new_model.blocks = torch.nn.ModuleList(layers[1:-1])
        return new_model