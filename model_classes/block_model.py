import torch
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

from pytorch_module_wrapper_classes import module_wrapper

class block_model(module_wrapper, ABC):
    
    @abstractmethod
    def __init__(self,
                 model_properties,
                 num_classes):
        '''
        Initialize a block model
        '''
        module_wrapper.__init__(self)
        assert num_classes >= 2
        self.num_classes = num_classes
        assert model_properties['num_blocks'] >= 1
        self.num_blocks = model_properties['num_blocks']
        self.model_properties = model_properties
        
        self.blocks = torch.nn.ModuleList([self._make_block(block_idx) for block_idx in range(self.num_blocks)])
        self.block_names = ['layer' + str(block_idx + 1) for block_idx in range(self.num_blocks)]
    
    def is_layer_batch_norm(self,
                            layer_name):
        return 'norm' in layer_name or 'bn' in layer_name or 'downsample.1' in layer_name
    
    @abstractmethod
    def _make_block(self,
                    block_idx):
        '''
        Make a block component of this model
        '''
        pass
    
    def forward(self,
                x,
                dropout = 0):
        '''
        Compute output from model
        @param self:
        @param x: torch FloatTensor
        @param dropout: float, probability of setting an entry to 0 where dropout is applied
        @return: torch FloatTensor
        '''
        out = x
        for layer_idx in range(len(self.layers)):
            out = self.layers[layer_idx](out)
            if layer_idx != len(self.layers) - 1 and self.training and dropout > 0:
                out = torch.nn.functional.dropout(out, dropout)
        return out

    def get_activations_from_all_layers(self,
                                        x):
        '''
        Return the output from each block of the model
        @param self:
        @param x: torch FloatTensor
        @return: list of torch FloatTensor
        '''
        outputs = []
        out = x
        for layer_idx in range(len(self.layers)):
            out = self.layers[layer_idx](out)
            outputs.append(out)
        return outputs
    
    def compute_param_norm(self,
                           reg_type = 'l2'):
        '''
        Compute sum of squared L2 norm or sum of L1 norm of parameters in convolutional or fully connected layers
        Batch norm parameters are not included in sum
        @param self
        @param reg_type: str, l2 or l1
        @return: torch FloatTensor
        '''
        reg = 0.
        for layer in self.layers:
            reg += layer.compute_param_norm(reg_type)
        '''
        for name, param in self.named_parameters():
            if not self.is_layer_batch_norm(name):
                if reg_type == 'l2':
                    reg += torch.sum(torch.square(param))
                else:
                    reg += torch.sum(torch.abs(param))
        '''
        return reg
    
    def compute_fisher_info(self,
                            train_loader):
        '''
        Compute Fisher information for each parameter (average gradient squared)
        @param self:
        @param train_loader: torch DataLoader, contains training features and outcomes in batches
        @return: dict mapping str for parameter name to torch FloatTensor containing Fisher information for each parameter
        '''
        self.eval()
        adam_optimizer = torch.optim.Adam(self.parameters())
        fisher_infos   = defaultdict(lambda: 0)
        loss_fn        = torch.nn.CrossEntropyLoss(reduction = 'none')
        num_samples    = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            if len(batch_samples) == 3:
                batch_x, batch_y, batch_weight = batch_samples
                if torch.cuda.is_available():
                    batch_weight = batch_weight.cuda()
            else:
                batch_x, batch_y = batch_samples
            if torch.cuda.is_available():
                batch_x  = batch_x.cuda()
                batch_y  = batch_y.cuda()

            batch_pred       = self(batch_x)
            if len(batch_samples) == 3:
                batch_train_loss = batch_weight * loss_fn(batch_pred, batch_y)
            else:
                batch_train_loss = loss_fn(batch_pred, batch_y)
            batch_train_loss = torch.sum(batch_train_loss)
            if torch.cuda.is_available():
                batch_train_loss = batch_train_loss.cuda()
            batch_train_loss.backward(retain_graph = True)

            for name, param in self.named_parameters():
                if not self.is_layer_batch_norm(name):
                    fisher_infos[name] += torch.square(param.grad.detach())
            num_samples += len(batch_y)

            adam_optimizer.zero_grad()
        self.train()
        for name in fisher_infos:
            fisher_infos[name] /= num_samples
        return fisher_infos
    
    def compute_param_norm_to_another_model(self,
                                            other_model,
                                            norm        = 'l2',
                                            fisher_info = defaultdict(lambda: None)):
        '''
        Compute sum of squared L2 norm or sum of L1 norm of difference between this model and other model
        Differences weighted by fisher infos if provided
        @param other_model: resnet_with_dropout model, must have same architecture
        @param norm: str, l1 or l2, must be l2 if using Fisher info
        @param fisher_info: dict mapping str to FloatTensor, layer name to Fisher info weight for each parameter
        @return: torch FloatTensor
        '''
        assert other_model.num_blocks       == self.num_blocks
        assert other_model.num_classes      == self.num_classes
        assert other_model.model_properties == self.model_properties
        assert norm in {'l1', 'l2'}
        if len(fisher_info) > 0:
            assert norm == 'l2'
        param_norm = 0.
        for name, param in self.named_parameters():
            if not self.is_layer_batch_norm(name):
                if norm == 'l2':
                    param_norm += torch.sum(fisher_info[name]
                                            * torch.square(param - other_model.get_parameter(name)))
                else:
                    param_norm += torch.sum(torch.abs(param - other_model.get_parameter(name)))
        return param_norm
    
    def get_param_vec(self,
                      layer_idx):
        '''
        Get all parameters in layer flattened and concatenated
        @param self:
        @param layer_idx: int, index of layer to get
        @return: np array
        '''
        assert layer_idx >= 0 and layer_idx < len(self.layers)
        return self.layers[layer_idx].get_param_vec()

    @classmethod
    def interpolate_weights(cls,
                            model1,
                            model2,
                            interpolation_val):
        '''
        Create a model with weights set to (1. - interpolation_val) * model1 + interpolation_val * model2
        @param model1: block_model
        @param model2: block_model
        @param interpolation_val: float, between 0 and 1 exclusive
        @return: block_model
        '''
        assert type(model1) is type(model2)
        assert model1.num_blocks       == model2.num_blocks
        assert model1.num_classes      == model2.num_classes
        assert model1.model_properties == model2.model_properties

        interpolated_model = deepcopy(model1)
        for layer_idx in range(len(model1.layers)):
            layer1 = model1.layers[layer_idx]
            layer2 = model2.layers[layer_idx]
            interpolated_model.layers[layer_idx] = type(layer1).interpolate_weights(layer1,
                                                                                    layer2,
                                                                                    interpolation_val)
        return interpolated_model