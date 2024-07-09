import sys
from os.path import dirname, abspath, join
from datetime import datetime

import config
from model_property_helper_functions import get_model_class_and_properties

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def compute_model_num_params(model_type,
                             num_blocks,
                             logger):
    '''
    Compute and log number of parameters in model
    @param model_type: str, resnet, alexnet, or densenet
    @param num_blocks: int, number of blocks
    @param logger: logger, for INFO messages
    '''
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    model = model_class(model_properties = model_properties,
                        num_classes      = 10)
    num_params = 0
    for layer_idx in range(len(model.layers)):
        num_params += len(model.get_param_vec(layer_idx))
    logger.info(model_type + ' with ' + str(num_blocks) + ' blocks has ' + str(num_params) + ' parameters')

if __name__ == '__main__':
    
    logging_filename = config.output_dir + 'compute_num_params_logs/compute_num_params_in_different_model_sizes_' \
                     + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    
    model_sizes = {'resnet'  : [1, 2, 3, 4],
                   'densenet': [1, 2, 3, 4],
                   'convnet' : [1, 2, 3, 4]}
    for model_type in model_sizes:
        for num_blocks in model_sizes[model_type]:
            compute_model_num_params(model_type,
                                     num_blocks,
                                     logger)