import math

from resnet_with_dropout import resnet_with_dropout
from densenet_with_dropout import densenet_with_dropout
from convnet_with_dropout import convnet_with_dropout
from joint_resnet_with_dropout import joint_resnet_with_dropout
from joint_densenet_with_dropout import joint_densenet_with_dropout
from joint_convnet_with_dropout import joint_convnet_with_dropout

def get_joint_model_class_and_properties(model_type,
                                         num_blocks,
                                         logger = None):
    '''
    Get model class based on name
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param logger: logger, for INFO messages
    @return: 1. joint model class
             2. model class
             3. dict mapping str to int or list of int, properties to define that model class
    '''
    assert model_type in {'resnet', 'densenet', 'convnet'}
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    if model_type == 'resnet':
        joint_model_class = joint_resnet_with_dropout
    elif model_type == 'densenet':
        joint_model_class = joint_densenet_with_dropout
    else:
        joint_model_class = joint_convnet_with_dropout
    return joint_model_class, model_class, model_properties

def get_model_class_and_properties(model_type,
                                   num_blocks,
                                   logger = None):
    '''
    Get model class based on name
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param logger: logger, for INFO messages
    @return: 1. model class
             2. dict mapping str to int or list of int, properties to define that model class
    '''
    assert model_type in {'resnet', 'densenet', 'convnet'}
    if model_type == 'resnet':
        model_class      = resnet_with_dropout
        model_properties = get_resnet_model_properties(num_blocks)
    elif model_type == 'densenet':
        model_class      = densenet_with_dropout
        model_properties = get_densenet_model_properties(num_blocks)
    else:
        model_class      = convnet_with_dropout
        model_properties = get_convnet_model_properties(num_blocks)
    if logger is not None:
        for property_name in model_properties:
            if property_name != 'num_blocks':
                logger.info(property_name + ': ' + ', '.join(map(str, model_properties[property_name])))
    return model_class, model_properties

def get_resnet_model_properties(num_blocks):
    '''
    Define block sizes for different sized resnet models
    @param num_blocks: int, number of blocks
    @return: dict mapping str to int or list of int, properties for constructing resnet blocks
    '''
    num_layers_per_block = [2 for _ in range(num_blocks)]
    if num_blocks == 1:
        input_dims       = [64]
        output_dims      = [512]
    elif num_blocks == 2:
        input_dims       = [64, 128]
        output_dims      = [128, 512]
    elif num_blocks == 3:
        input_dims       = [64, 128, 256]
        output_dims      = [128, 256, 512]
    else:
        input_dims       = [64, 64, 128, 256] + [512 for _ in range(num_blocks - 4)]
        output_dims      = [64, 128, 256] + [512 for _ in range(num_blocks - 3)]
    strides = [1 for _ in range(max(num_blocks - 3, 0))] + [2 for _ in range(min(num_blocks, 3))]
    
    return {'num_layers_per_block': num_layers_per_block,
            'input_dims'          : input_dims,
            'output_dims'         : output_dims,
            'strides'             : strides,
            'num_blocks'          : num_blocks}

def get_convnet_model_properties(num_blocks):
    '''
    Define block sizes for different sized resnet models
    @param num_blocks: int, number of blocks
    @return: dict mapping str to int or list of int, properties for constructing resnet blocks
    '''
    if num_blocks == 1:
        input_dims       = [64]
        output_dims      = [512]
    elif num_blocks == 2:
        input_dims       = [64, 128]
        output_dims      = [128, 512]
    elif num_blocks == 3:
        input_dims       = [64, 128, 256]
        output_dims      = [128, 256, 512]
    else:
        input_dims       = [64, 64, 128, 256] + [512 for _ in range(num_blocks - 4)]
        output_dims      = [64, 128, 256] + [512 for _ in range(num_blocks - 3)]
    strides = [1 for _ in range(max(num_blocks - 3, 0))] + [2 for _ in range(min(num_blocks, 3))]
    
    return {'input_dims'          : input_dims,
            'output_dims'         : output_dims,
            'strides'             : strides,
            'num_blocks'          : num_blocks}

def get_densenet_model_properties(num_blocks):
    '''
    Define block sizes for different sized densenet models
    @param num_blocks: int, number of blocks
    @return: dict mapping str to int or list of int, properties for constructing densenet blocks
    '''
    if num_blocks == 1:
        num_layers_per_block = [16]
    elif num_blocks == 2:
        num_layers_per_block = [6, 16]
    else:
        num_layers_per_block = [6, 12] + [24 for _ in range(num_blocks - 3)] + [16]
    return {'num_blocks': num_blocks,
            'num_layers_per_block': num_layers_per_block}

def compute_max_fc_adapter_rank(model_type,
                                num_blocks,
                                num_classes):
    '''
    Compute maximum possible rank for fully connected layer that is more parameter-efficient than original module
    @param model_type: str, densenet, convnet, or resnet
    @param num_blocks: int, number of blocks
    @param num_classes: int, number of output classes
    @return: int, rank
    '''
    assert model_type in {'resnet', 'densenet', 'convnet'}
    assert num_blocks > 0
    assert num_classes > 1
    if model_type in {'resnet', 'convnet'}:
        fc_input_size = 512
    else:
        if num_blocks == 1:
            num_layers_per_block = [16]
        elif num_blocks == 2:
            num_layers_per_block = [6, 16]
        else:
            num_layers_per_block = [6, 12] + [24 for _ in range(num_blocks - 3)] + [16]
        block_num_features = 64
        for idx in range(num_blocks):
            block_num_features += 32 * self.model_properties['num_layers_per_block'][idx]
            block_num_features  = block_num_features // 2
        fc_input_size = block_num_features
    
    fc_num_params = fc_input_size * num_classes
    fc_max_rank   = int(math.floor((fc_num_params - 1)/(fc_input_size + num_classes)))
    return fc_max_rank
    
def compute_conv_output_size(input_size,
                             padding,
                             kernel_size,
                             stride):
    '''
    Compute convolution output size (width / height)
    @param input_size, size of layer input in 1 dimension
    @param padding: int, as defined in conv2d
    @param kernel_size: int, as defined in conv2d
    @param stride: int, as defined in conv2d
    @return: int, output size
    '''
    return math.floor((input_size + 2 * padding - kernel_size)/stride + 1)

def get_layer_names(model_type,
                    num_blocks):
    '''
    Get layer names for a particular architecture
    @param model_type: str, convnet, densenet, or resnet
    @param num_blocks: int, number of blocks
    @return: list of str
    '''
    assert model_type in {'densenet', 'resnet', 'convnet'}
    layer_names = ['conv1']
    for i in range(num_blocks):
        layer_names.append('layer' + str(i + 1))
    layer_names.append('fc')
    return layer_names

def get_combos(layer_names):
    '''
    Create a list of comma-separated layer combinations
    Include all combinations with and without each layer
    @param layer_names: list of str
    @return: list of str, combinations
    '''
    if len(layer_names) == 1:
        return layer_names
    combos_without_first = get_combos(layer_names[1:])
    return [layer_names[0]] + [layer_names[0] + ',' + c for c in combos_without_first] + [c for c in combos_without_first]

def define_layer_combos(model_type,
                        num_blocks):
    '''
    Create a list of comma-separated combinations of layers to tune for a particular model architecture
    @param model_type: str, convnet, densenet, or resnet
    @param num_blocks: int, number of blocks
    @return: list of str
    '''
    layer_names = get_layer_names(model_type,
                                  num_blocks)
    combos = get_combos(layer_names)
    all_idx = combos.index(','.join(layer_names))
    combos[all_idx] = 'all'
    return combos

def get_plot_layer_names(model_type,
                         num_blocks):
    '''
    Get readable layer names for plotting
    @param model_type: str, convnet, densenet, or resnet
    @param num_blocks: int, number of blocks
    @return: list of str
    '''
    layers = []
    if model_type in {'densenet', 'resnet', 'convnet'}:
        layers.append('input convolution')
    model_type_map = {'densenet': 'Dense',
                      'resnet': 'Residual',
                      'convnet': 'Conv'}
    layers.extend(['block ' + str(block_idx)
                   for block_idx in range(num_blocks)])
    layers.append('output fully connected')
    return layers