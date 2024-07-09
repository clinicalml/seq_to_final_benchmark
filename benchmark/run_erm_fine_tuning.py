import os
import sys
import time
import json
import argparse

from itertools import product
from functools import partial
from os.path import dirname, abspath, join
from datetime import datetime
from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from fit_single_model import (
    load_imagenet_pretrained_resnet18, 
    tune_hyperparameters_for_model,
    compute_accuracy
)
from run_joint_model_learning import (
    tune_joint_model_hyperparameters_for_a_single_combo,
    compute_joint_model_accuracy_for_a_time_step
)

sys.path.append(join(dirname(dirname(abspath(__file__))), 'model_classes'))
from model_property_helper_functions import (
    get_layer_names,
    define_layer_combos,
    get_plot_layer_names,
    get_model_class_and_properties,
    compute_max_fc_adapter_rank
)

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

sys.path.append(join(dirname(dirname(abspath(__file__))), 'data_loaders'))
from load_image_data_with_shifts import load_datasets_over_time, combine_data_loaders_over_time
from load_yearbook_data import load_yearbook_data

sys.path.append(dirname(dirname(abspath(__file__))))
import config

'''
Methods:
1. ERM on data from final time period
2. ERM on data from all time periods (with option to weight losses at each time t by 1/(T - t + 1))
4. ERM on data from all previous time periods and fine tuning at final time step
    Layer options:
    a. Specified set of layers to tune, e.g. all layers
    b. Adding layers ordered by surgical fine tuning metrics: relative gradient norm or signal-to-noise ratio
    c. Searching over all sets of layers to tune
    d. Linear probing then fine tuning all layers
    e. Gradual unfreezing: first to last or last to first
    f. Any specified combination of layers
    g. Add side modules or low-rank adapters instead of separate layers
    Regularization options:
    a. Standard: L2 regularization towards 0
    b. Previous: L2 regularization towards previous parameters
    c. Fisher: L2 regularization towards previous parameters weighted by Fisher info
    d. Previous L1: L1 regularization towards previous parameters
5. Sequential fine tuning on each time period:
    Same layer options as 4
    Same regularization options plus:
    e. EWC (elastic weight consolidation): L2 regularization towards previous parameters from all time steps weighted by Fisher info
    Learning rate options:
    a. Tuned between 1e-3 and 1e-4
    b. Decayed starting at 1e-4 and subtracting 1e-5 at each time step or starting at 1e-3 and subtracting 1e-4
    c. Decayed starting at 2e-4 and multiplying .5 at each time step, starting at 2e-3 and multiplying .5, 
       or starting at 2e-4 and multiplying .75
6. IRM or DRO on data from all time periods
7. IRM or DRO on data from all historic time periods, then fine-tune at the final time step
'''
    
def run_erm(dataset_name,
            model_type,
            num_blocks,
            shift_sequence,
            source_sample_size,
            target_sample_sizes,
            final_target_test_size,
            seed                = 1007,
            loss_weights        = 'equal',
            start_with_imagenet = False,
            method              = 'erm'):
    '''
    Fit a single model on data from all time steps via ERM
    Compute test accuracy on test set at final time step
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         next 4 arguments disregarded for portraits
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param seed: int, for np random generator
    @param loss_weights: str, how to weight loss for each time point, options: equal, final, increasing
    @param start_with_imagenet: bool, whether to initialize first model with pre-trained model from ImageNet
    @param method: str, erm, irm, or dro
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'resnet', 'convnet', 'densenet'}
    assert num_blocks > 0
    assert loss_weights in {'equal', 'final', 'increasing'}
    assert method in {'erm', 'irm', 'dro'}
    if method != 'erm':
        assert loss_weights == 'equal'
    real_world_datasets = {'portraits'}
    if dataset_name in real_world_datasets:
        num_target_steps = 7
        shift_sequence_with_sample_size = dataset_name
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    if loss_weights == 'equal':
        method_name = method + '_all'
    elif loss_weights == 'final':
        method_name = 'erm_final'
    else:
        method_name = 'erm_weighted_loss'
    if start_with_imagenet:
        method_name += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir         = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + method_name + '/' \
                       + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    if os.path.exists(method_dir + 'metrics.csv'):
        return # this experiment has already been run
    logging_dir        = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_' + method_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Running ' + method)
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model type: '             + model_type)
    logger.info('# blocks in model: '      + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: '    
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    logger.info('Loss weights: '           + loss_weights)
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fileheader = model_dir + method_name + '_'
    
    # even if only using final samples, load all time steps to get the same data based on random seeds
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger,
                                                    model_dir + 'portraits_examples.pdf')
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = True,
                                                         output_dir       = model_dir)
    
    if loss_weights == 'final':
        combined_data_loaders = data_loaders_over_time[-1]
    else:
        num_time_steps = len(data_loaders_over_time)
        if loss_weights == 'equal':
            time_weights = np.ones(num_time_steps)
        else:
            time_weights = np.empty(num_time_steps)
            for t in range(len(time_weights)):
                time_weights[t] = 1./(num_time_steps - t)
        if method == 'erm':
            combined_data_loaders = combine_data_loaders_over_time([{'train': loaders['train']}
                                                                    for loaders in data_loaders_over_time],
                                                                   time_weights)
    
    learning_rates = [1e-3, 1e-4]
    weight_decays  = [0.0001, 0.01]
    dropouts       = [0, 0.5]
    n_epochs       = 100
    
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    erm_model   = model_class(model_properties = model_properties,
                              num_classes      = num_classes)
    if start_with_imagenet:
        assert model_type == 'resnet'
        assert num_blocks == 4
        erm_model   = load_imagenet_pretrained_resnet18(erm_model)
    if method == 'erm':
        train_data = combined_data_loaders['train']
    else:
        train_data = [data_loaders['train'] for data_loaders in data_loaders_over_time]
    erm_model, erm_val_acc = tune_hyperparameters_for_model('all',
                                                            erm_model,
                                                            train_data,
                                                            data_loaders_over_time[-1]['valid'],
                                                            learning_rates,
                                                            weight_decays,
                                                            dropouts,
                                                            n_epochs,
                                                            fileheader,
                                                            logger,
                                                            method = method)
    
    final_test_acc = compute_accuracy(erm_model,
                                      data_loaders_over_time[-1]['test'])
    logger.info('Test accuracy on final time step from ERM on all data: ' + str(final_test_acc))
    
    model_name    = model_type + ' ' + str(num_blocks) + ' blocks'
    if method_name == 'erm_final' and dataset_name not in real_world_datasets:
        method_name += '_' + str(target_sample_sizes[-1]) + '_samples'
    df = pd.DataFrame(data    = {'Model'         : [model_name],
                                 'Method'        : [method_name],
                                 'Shift sequence': [shift_sequence_with_sample_size],
                                 'Seed'          : [seed],
                                 'Test accuracy' : [final_test_acc]},
                      columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy'])
    df.to_csv(method_dir + 'metrics.csv',
              index = False)
    
def order_layers_to_tune_surgical_metrics(model,
                                          data_loader,
                                          surgical_metric,
                                          fileheader):
    '''
    Compute surgical fine tuning metric for each layer
    Save to json, loads from json if exists
    Return layers ordered by metric from largest to smallest
    @param model: block_model, model to compute metrics for
    @param data_loader: torch DataLoader, used to compute metrics
    @param surgical_metric: str, rgn (relative gradient norm) or snr (signal-to-noise ratio)
    @param fileheader: str, start of path to save metrics
    @return: list of str, layer names ordered by surgical fine tuning metric
    '''
    assert surgical_metric in {'rgn', 'snr'}
    assert fileheader[-1] == '_'
    
    surgical_metrics_filename = fileheader + 'surgical_' + surgical_metric + '_metrics_per_layer.json'
    if os.path.exists(surgical_metrics_filename):
        with open(surgical_metrics_filename, 'r') as f:
            layer_total_metrics = json.load(f)
    else:
        model.eval()
        loss_fn     = torch.nn.CrossEntropyLoss(reduction = 'mean')
        num_samples = len(data_loader.dataset.tensors[1])
        single_sample_data_loader = torch.utils.data.DataLoader(data_loader.dataset,
                                                                batch_size = 1,
                                                                shuffle    = False)
        layer_metrics = {layer: defaultdict(lambda: 0)
                         for layer in model.layer_names}
        for sample_idx, sample in enumerate(data_loader):
            if len(sample) == 3:
                sample_x, sample_y, _ = sample
            else:
                sample_x, sample_y    = sample
            if torch.cuda.is_available():
                sample_x = sample_x.cuda()
                sample_y = sample_y.cuda()

            sample_pred  = model(sample_x)
            sample_loss  = loss_fn(sample_pred, sample_y)
            for layer, layer_name in zip(model.layers, model.layer_names):
                sample_grads = torch.autograd.grad(outputs      = sample_loss, 
                                                   inputs       = layer.parameters(), 
                                                   retain_graph = True)
                for (name, param), grad in zip(layer.named_parameters(), sample_grads):
                    if surgical_metric == 'rgn':
                        layer_metrics[layer_name][name] += torch.norm(grad)
                    else:
                        layer_metrics[layer_name][name] += torch.mean(torch.square(grad)/(torch.var(grad, 
                                                                                                    dim=0, 
                                                                                                    keepdim=True) + 1e-8))
        for layer, layer_name in zip(model.layers, model.layer_names):
            for name, param in layer.named_parameters():
                layer_metrics[layer_name][name] /= num_samples
                if surgical_metric == 'rgn':
                    layer_metrics[layer_name][name] /= torch.norm(param)

        layer_total_metrics = defaultdict(lambda: 0)
        for layer_name in layer_metrics:
            layer_total_metric = 0
            for param_name in layer_metrics[layer_name]:
                layer_total_metric += layer_metrics[layer_name][param_name]
            layer_total_metrics[layer_name] = float(layer_total_metric) / len(layer_metrics[layer_name])

        model.train()

        surgical_metrics_filename = fileheader + 'surgical_' + surgical_metric + '_metrics_per_layer.json'
        with open(surgical_metrics_filename, 'w') as f:
            json.dump(layer_total_metrics, f)
    
    layer_metrics = np.array([layer_total_metrics[layer_name] for layer_name in layer_names])
    ordered_layer_idxs = np.argsort(layer_metrics)[::-1]
    return [layer_names[i] for i in ordered_layer_idxs]

def check_layer_option_is_lp_ft(layers,
                                model_type,
                                num_blocks):
    '''
    Check if order of layers to tune is linear probing then fine tuning
    @param layers: list of str, comma-separated combinations of layers to tune
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @return: bool, True if is LP-FT
    '''
    if len(layers) != 2:
        return False
    if layers[0] != 'fc':
        return False
    layer_names = get_layer_names(model_type,
                                  num_blocks)
    if layers[1] != 'all' and set(layers[1].split(',')) != set(layer_names):
        return False
    return True

def check_layer_option_is_first_to_last(layers,
                                        model_type,
                                        num_blocks):
    '''
    Check if order of layers to tune is gradual unfreezing first to last
    @param layers: list of str, comma-separated combinations of layers to tune
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @return: bool, True if is first to last
    '''
    layer_order = get_layer_names(model_type,
                                  num_blocks)
    if len(layers) != len(layer_order):
        return False
    for i in range(len(layer_order)):
        if i == len(layer_order) - 1:
            if layers[i] != 'all' and set(layers[i].split(',')) != set(layer_order):
                return False
        elif set(layers[i].split(',')) != set(layer_order[:i+1]):
            return False
    return True
    
def check_layer_option_is_last_to_first(layers,
                                        model_type,
                                        num_blocks):
    '''
    Check if order of layers to tune is gradual unfreezing last to first
    @param layers: list of str, comma-separated combinations of layers to tune
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @return: bool, True if is last to first
    '''
    layer_order = get_layer_names(model_type,
                                  num_blocks)
    if len(layers) != len(layer_order):
        return False
    for i in range(len(layer_order)):
        if i == len(layer_order) - 1:
            if layers[i] != 'all' and set(layers[i].split(',')) != set(layer_order):
                return False
        elif set(layers[i].split(',')) != set(layer_order[:i+1]):
            return False
    return True
    
def run_erm_fine_tune(dataset_name,
                      model_type,
                      num_blocks,
                      shift_sequence,
                      source_sample_size,
                      target_sample_sizes,
                      final_target_test_size,
                      layers_to_tune,
                      seed                  = 1007,
                      regularization        = 'standard',
                      start_with_imagenet   = False,
                      ablate_fine_tune_init = False,
                      initial_method        = 'erm'):
    '''
    Fit a single block model on data from all time steps before the final time step via ERM
    Then fine tune on data from last time step
    Compute test accuracy on test set at final time step
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         for portraits, shift_sequence, source_sample_size, target_sample_sizes, 
                         and final_target_test_size arguments are disregarded
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param layers_to_tune: list of str, 
                           options for length 1 list:
                           - search to find the best combination of layers to tune
                           - surgical_rgn to use relative gradient norm to determine which blocks to tune
                           - surgical_snr to use signal-to-noise ratio to determine which blocks to tune
                           options for each entry of length 1+ list:
                           - all to tune all layers
                           - comma-separated combination of layers: conv1,layer1,layer2,layer3,layer4,fc
                           examples:
                           - ['all'] is standard fine tuning
                           - ['fc', 'all'] is linear probe then fine tune
                           - ['fc', 'fc,layer4', 'fc,layer4,layer3', 'fc,layer4,layer3,layer2', 
                              'fc,layer4,layer3,layer2,layer1', 'fc,layer4,layer3,layer2,layer1,conv1'] 
                             is gradual unfreezing last -> first
    @param seed: int, for np random generator
    @param regularization: str, options: standard, previous, fisher, fisher_all, previous_l1
                           specify whether L2 regularization is towards 0, towards weights at t-1, 
                           towards weights at t-1 weighted by Fisher, towards weights at all previous t weighted by Fisher,
                           or towards weights at t-1 as L1 instead of L2 regularization
    @param start_with_imagenet: bool, whether to initialize first model with pre-trained model from ImageNet
    @param ablate_fine_tune_init: bool, ablate to initialize "fine tuned" model from scratch,
                                  generally use in conjunction with some regularization towards previous model,
                                  otherwise becomes erm_final
    @param initial_method: str, erm, irm, or dro
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'resnet', 'convnet', 'densenet'}
    assert initial_method in {'erm', 'irm', 'dro'}
    assert num_blocks > 0
    real_world_datasets = {'portraits'}
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
        num_target_steps = 7
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    is_lp_ft         = False
    is_last_to_first = False
    is_first_to_last = False
    if len({'search', 'surgical_rgn', 'surgical_snr'}.intersection(set(layers_to_tune))) > 0:
        assert len(layers_to_tune) == 1
    if layers_to_tune == ['all']:
        method_name = initial_method + '_fine_tune'
    elif layers_to_tune[0] in {'search', 'surgical_rgn', 'surgical_snr'}:
        method_name = initial_method + '_fine_tune_' + layers_to_tune[0]
    elif check_layer_option_is_lp_ft(layers_to_tune,
                                     model_type,
                                     num_blocks):
        is_lp_ft    = True
        method_name = initial_method + '_linear_probe_then_fine_tune'
    elif check_layer_option_is_last_to_first(layers_to_tune,
                                             model_type,
                                             num_blocks):
        is_last_to_first = True
        method_name      = initial_method + '_gradual_unfreeze_last_to_first_fine_tune'
    elif check_layer_option_is_first_to_last(layers_to_tune,
                                             model_type,
                                             num_blocks):
        is_first_to_last = True
        method_name      = initial_method + '_gradual_unfreeze_first_to_last_fine_tune'
    else:
        method_name = initial_method + '_fine_tune_' + '-'.join(layers_to_tune)
    if ablate_fine_tune_init:
        method_name += '_ablate_init'
    if regularization == 'previous':
        method_name += '_reg_previous'
    elif regularization == 'fisher':
        method_name += '_reg_previous_fisher'
    elif regularization == 'previous_l1':
        method_name += '_l1_reg_previous'
    erm_name = initial_method + '_all_prev'
    if start_with_imagenet:
        method_name += '_start_with_imagenet'
        erm_name    += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir  = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + method_name + '/' \
                + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    if os.path.exists(method_dir + 'metrics.csv'):
        return # this experiment has already been run
    logging_dir = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_' + method_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Running ' + initial_method + ' then fine tune on last time step')
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model architecture: '     + model_type)
    logger.info('# blocks: '               + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: ' 
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    logger.info('Layers to tune: '         + ' - '.join(layers_to_tune))
    logger.info('Regularization: '         + regularization)
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    logger.info('Ablate fine-tune init: '  + str(ablate_fine_tune_init))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    erm_dir   = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + erm_name + '/' \
              + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(erm_dir):
        os.makedirs(erm_dir)
    fileheader     = model_dir + method_name + '_'
    erm_fileheader = erm_dir + erm_name + '_'
    
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger,
                                                    model_dir + 'portraits_examples.pdf')
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = True,
                                                         output_dir       = model_dir)
    
    combined_data_loaders = combine_data_loaders_over_time([{'train': loaders['train'],
                                                             'valid': loaders['valid']}
                                                            for loaders in data_loaders_over_time[:-1]])
    
    learning_rates = [1e-3, 1e-4]
    weight_decays  = [0.0001, 0.01]
    dropouts       = [0, 0.5]
    n_epochs       = 100
    
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    erm_model   = model_class(model_properties = model_properties,
                              num_classes = num_classes)
    if start_with_imagenet:
        assert model_type == 'resnet'
        assert num_blocks == 4
        erm_model    = load_pretrained_weights(erm_model)
    if initial_method == 'erm':
        train_data = combined_data_loaders['train']
    else:
        train_data = [data_loaders['train'] for data_loaders in data_loaders_over_time]
    erm_model, erm_val_acc = tune_hyperparameters_for_model('all',
                                                            erm_model,
                                                            train_data,
                                                            combined_data_loaders['valid'],
                                                            learning_rates,
                                                            weight_decays,
                                                            dropouts,
                                                            n_epochs,
                                                            erm_fileheader,
                                                            logger,
                                                            remove_uncompressed_model = False,
                                                            method = initial_method)
    
    prev_erm_test_acc    = compute_accuracy(erm_model, data_loaders_over_time[-1]['test'])
    prev_reg_model       = []
    prev_reg_fisher_info = []
    if regularization in {'previous', 'fisher', 'previous_l1'}:
        prev_reg_model.append(erm_model)
    if regularization == 'fisher':
        prev_reg_fisher_info.append(erm_model.compute_fisher_info(combined_data_loaders['train']))
        for name in prev_reg_fisher_info[0]:
            logger.info('Average Fisher info for ' + name + ' from ERM on all previous time steps: ' 
                        + str(float(torch.mean(prev_reg_fisher_info[0][name]))))
    
    if layers_to_tune[0] in {'search', 'surgical_rgn', 'surgical_snr'}:
        if layers_to_tune == 'search':
            combos         = define_layer_combos()
        else:
            ordered_layers = order_layers_to_tune_surgical_metrics(deepcopy(erm_model),
                                                                   data_loaders_over_time[-1]['train'],
                                                                   layers_to_tune[0][-3:],
                                                                   fileheader)
            combos = [','.join(ordered_layers[:layer_idx+1]) for layer_idx in range(len(ordered_layers))]
        best_fine_tune_val_acc = -1
        combo_search_test_accs = dict()
        for combo in combos:
            if ablate_fine_tune_init:
                init_fine_tune_model = model_class(model_properties = model_properties,
                                                   num_classes      = num_classes)
            else:
                init_fine_tune_model = deepcopy(erm_model)
            fine_tune_model, fine_tune_val_acc \
                = tune_hyperparameters_for_model(combo,
                                                 init_fine_tune_model,
                                                 data_loaders_over_time[-1]['train'],
                                                 data_loaders_over_time[-1]['valid'],
                                                 learning_rates,
                                                 weight_decays,
                                                 dropouts,
                                                 n_epochs,
                                                 fileheader + 'fine_tune_',
                                                 logger,
                                                 regularization              = regularization,
                                                 regularization_prev_models  = prev_reg_model,
                                                 regularization_fisher_infos = prev_reg_fisher_info)
            if fine_tune_val_acc > best_fine_tune_val_acc:
                best_combo              = combo
                best_fine_tune_val_acc  = fine_tune_val_acc
                best_fine_tune_model    = fine_tune_model
                
            combo_search_test_accs[combo] = compute_accuracy(fine_tune_model, data_loaders_over_time[-1]['test'])
    
        best_combo_filename = fileheader + 'fine_tune_best_layers_to_tune.json'
        with open(best_combo_filename, 'w') as f:
            json.dump({'Best layers to tune': best_combo}, f)
        combo_test_acc_filename = fileheader + 'fine_tune_search_layers_to_tune_test_accs.json'
        with open(combo_test_acc_filename, 'w') as f:
            json.dump(combo_search_test_accs, f)
        logger.info('Best combination of layers for fine tuning: ' + best_combo)
        
        test_acc = compute_accuracy(best_fine_tune_model, data_loaders_over_time[-1]['test'])
        logger.info('Test accuracy from ERM on previous time steps: ' + str(prev_erm_test_acc))
        logger.info('Test accuracy from ERM and fine tuning on final time step: ' + str(test_acc))
        
    else:
        if ablate_fine_tune_init:
            prev_model = model_class(model_properties = model_properties,
                                     num_classes      = num_classes)
        else:
            prev_model = deepcopy(erm_model)
        test_accs = []
        for combo_idx, combo in enumerate(layers_to_tune):
            if layers_to_tune == ['all']:
                model_fileheader = fileheader + 'fine_tune_'
            elif is_lp_ft:
                if combo_idx == 0:
                    model_fileheader = fileheader + 'linear_probe_'
                else:
                    model_fileheader = fileheader + 'linear_probe_then_fine_tune_'
            elif is_last_to_first:
                model_fileheader = fileheader + 'gradual_unfreeze_last_to_first_fine_tune_' + str(combo_idx + 1) + 'layers_'
            elif is_first_to_last:
                model_fileheader = fileheader + 'gradual_unfreeze_first_to_last_fine_tune_' + str(combo_idx + 1) + 'layers_'
            else:
                model_fileheader = fileheader + 'fine_tune_' + '-'.join(layers_to_tune[:combo_idx+1]) + '_'
            best_fine_tune_model, _ = tune_hyperparameters_for_model(combo,
                                                                     prev_model,
                                                                     data_loaders_over_time[-1]['train'],
                                                                     data_loaders_over_time[-1]['valid'],
                                                                     learning_rates,
                                                                     weight_decays,
                                                                     dropouts,
                                                                     n_epochs,
                                                                     model_fileheader,
                                                                     logger,
                                                                     regularization              = regularization,
                                                                     regularization_prev_models  = prev_reg_model,
                                                                     regularization_fisher_infos = prev_reg_fisher_info)
            prev_model = deepcopy(best_fine_tune_model)
            test_accs.append(compute_accuracy(best_fine_tune_model, data_loaders_over_time[-1]['test']))
    
        logger.info('Test accuracy from ERM on previous time steps: ' + str(prev_erm_test_acc))
        for combo_idx, test_acc in enumerate(test_accs):
            logger.info('Test accuracy from ERM + fine-tuning ' + ' & '.join(layers_to_tune[:combo_idx+1]) 
                        + ': ' + str(test_acc))
        test_acc = test_accs[-1]
    
    model_name    = model_type + ' ' + str(num_blocks) + ' blocks'
    df = pd.DataFrame(data    = {'Model'         : [model_name],
                                 'Method'        : [method_name],
                                 'Shift sequence': [shift_sequence_with_sample_size],
                                 'Seed'          : [seed],
                                 'Test accuracy' : [test_acc]},
                      columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy'])
    df.to_csv(method_dir + 'metrics.csv',
              index = False)
            
def run_erm_side_tune_or_low_rank_adapt(dataset_name,
                                        model_type,
                                        num_blocks,
                                        shift_sequence,
                                        source_sample_size,
                                        target_sample_sizes,
                                        final_target_test_size,
                                        mode,
                                        adapter_rank         = 0,
                                        adapter_mode         = None,
                                        side_layer_sizes     = [],
                                        seed                 = 1007,
                                        start_with_imagenet  = False,
                                        all_layers_efficient = False):
    '''
    Fit a single model on data from all time steps before the final time step via ERM
    Then fit a joint model with parameters for t = 0 frozen at the parameters from ERM on the previous steps
    and parameters for t = 1 fit on data from the last time step
    Compute test accuracy on test set at final time step
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         for portraits, shift_sequence, source_sample_size, target_sample_sizes, 
                         and final_target_test_size arguments are disregarded
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param mode: str, side_tune, low_rank_adapt, separate (match erm final),
                 separate_with_init (initialize t = 1 with t = 0 parameters to match fine tune),
                 separate_with_prev_reg (match fine tune ablate init with regularization towards previous),
                 or separate_with_init_and_prev_reg (match fine tune with regularization towards previous)
    @param adapter_rank: int, rank of adapters
    @param adapter_mode: str, add or multiply
    @param side_layer_sizes: str or list of int, number of output channels for each of the intermediate convolution layers 
                             in side module, same for all block layers,
                             str option is "block" for side module to match original block
    @param seed: int, for np random generator
    @param start_with_imagenet: bool, whether to initialize first model with pre-trained model from ImageNet
    @param all_layers_efficient: bool, whether to make non-block modules parameter-efficient if mode is not separate,
                                 side modules for non-block modules are single layer modules,
                                 low-rank adapters for non-block modules are fixed at rank 10,
                                 other than the input conv layer in resnets that can only be rank 1 to be more efficient,
                                 when False, non-block modules are just separate
    @return: None
    '''
    assert mode in {'side_tune', 'low_rank_adapt', 'separate', 'separate_with_init', 'separate_with_prev_reg', 
                    'separate_with_init_and_prev_reg'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
        assert adapter_rank > 0
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'resnet', 'densenet', 'convnet'}
    assert num_blocks > 0
    real_world_datasets = {'portraits'}
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
        num_target_steps = 7
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    use_partial_to_init_final = False
    joint_mode                = mode
    if mode.startswith('separate'):
        joint_mode  = 'separate'
        if mode == 'separate':
            method_name = 'erm_final_using_frozen_joint'
        elif mode == 'separate_with_prev_reg':
            method_name = 'erm_fine_tune_using_frozen_joint_adjacent_reg_l2_ablate_init'
        elif mode == 'separate_with_init':
            method_name = 'erm_fine_tune_using_frozen_joint_adjacent_reg_none'
            use_partial_to_init_final = True
        else:
            method_name = 'erm_fine_tune_using_frozen_joint_adjacent_reg_l2'
            use_partial_to_init_final = True
    elif mode == 'low_rank_adapt':
        method_name = 'erm_low_rank_adapt_' + adapter_mode + '_rank' + str(adapter_rank)
    else:
        method_name = 'erm_side_tune'
        if len(side_layer_sizes) > 0:
            if model_type == 'convnet':
                assert isinstance(side_layer_sizes, str)
            if isinstance(side_layer_sizes, str):
                assert side_layer_sizes == 'block'
                method_name += '_block'
            else:
                method_name += '_' + str(len(side_layer_sizes) + 1) + 'layers_size' \
                             + ', '.join([str(size) for size in side_layer_sizes])
    erm_name = 'erm_all_prev'
    if start_with_imagenet:
        method_name += '_start_with_imagenet'
        erm_name    += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir  = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + method_name + '/' \
                + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    if os.path.exists(method_dir + 'metrics.csv'):
        return # this experiment has already been run
    logging_dir = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_' + method_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Running ERM then learning a joint model with t = 0 frozen at ERM and t = 1 fit on last time step')
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model architecture: '     + model_type)
    logger.info('# blocks: '               + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: ' 
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    logger.info('Mode: '                   + str(mode))
    if mode == 'low_rank_adapt':
        logger.info('Adapter mode: '       + str(adapter_mode))
        logger.info('Adapter rank: '       + str(adapter_rank))
    if mode == 'side_tune':
        if isinstance(side_layer_sizes, str):
            logger.info('Side modules are blocks.')
        else:
            logger.info('Side modules: '       + ('No intermediate layers' if len(side_layer_sizes) == 0
                                                  else 'intermediate output sizes are ' 
                                                  + ', '.join([str(size) for size in side_layer_sizes])))
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    logger.info('Making non-block layers parameter-efficient: ' + str(all_layers_efficient))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    erm_dir   = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + erm_name + '/' \
              + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(erm_dir):
        os.makedirs(erm_dir)
    fileheader     = model_dir + method_name + '_'
    erm_fileheader = erm_dir + erm_name + '_'
    
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger,
                                                    model_dir + 'portraits_examples.pdf')
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = True,
                                                         output_dir       = model_dir)
    
    combined_data_loaders = combine_data_loaders_over_time([{'train': loaders['train'],
                                                             'valid': loaders['valid']}
                                                            for loaders in data_loaders_over_time[:-1]])
    
    learning_rates = [1e-3, 1e-4]
    all_prev_weight_decays = [0.0001, 0.01]
    if mode in {'separate_with_prev_reg', 'separate_with_init_and_prev_reg'}:
        weight_decays  = [0]            # match fine-tuning with regularization only towards previous
        adjacent_regs  = [0.0001, 0.01]
    elif mode in {'separate', 'separate_with_init'}:
        weight_decays  = [0.0001, 0.01] # match erm final and fine-tuning
        adjacent_regs  = [0]
    else:
        weight_decays  = [0.0001, 0.01]
        adjacent_regs  = [0.01, 0.1, 1]
    dropouts       = [0, 0.5]
    n_epochs       = 100
    last_time_step_loss_wts = [2] # after dividing by # time steps this should match fine-tuning
    regularization_params = {'weight_decay_type' : 'l2', 
                             'weight_decays'     : weight_decays,
                             'adjacent_reg_type' : 'l2', 
                             'adjacent_regs'     : adjacent_regs,
                             'weight_reg_by_time': False,
                             'dropouts'          : dropouts}
    
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
    side_layers     = [side_layer_sizes for _ in range(num_blocks)]
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    erm_model    = model_class(model_properties = model_properties,
                               num_classes      = num_classes)
    if start_with_imagenet:
        assert args.num_blocks == 4
        erm_model    = load_pretrained_weights(erm_model)
    erm_model, erm_val_acc = tune_hyperparameters_for_model('all',
                                                            erm_model,
                                                            combined_data_loaders['train'],
                                                            combined_data_loaders['valid'],
                                                            learning_rates,
                                                            all_prev_weight_decays,
                                                            dropouts,
                                                            n_epochs,
                                                            erm_fileheader,
                                                            logger,
                                                            remove_uncompressed_model = False)
    prev_erm_test_acc  = compute_accuracy(erm_model, data_loaders_over_time[-1]['test'])
    prev_erm_file_name = erm_fileheader + 'all_best_model.pt.gz'
    
    layer_names = get_layer_names(model_type, num_blocks)
    
    if mode == 'side_tune':
        side_layers     = [side_layer_sizes for _ in range(num_blocks)]
        if all_layers_efficient:
            if model_type == 'convnet':
                side_layers = ['block'] + side_layers + ['block']
            else:
                side_layers = [[]] + side_layers + [[]]
        else:
            side_layers = ['separate'] + side_layers + ['separate']
    else:
        side_layers   = [[] for _ in range(num_blocks + 2)]
    
    if mode == 'low_rank_adapt':
        adapter_ranks   = [adapter_rank for _ in range(num_blocks)]
        if all_layers_efficient:
            fc_adapter_rank = min(adapter_rank,
                                  compute_max_fc_adapter_rank(model_type,
                                                              num_blocks,
                                                              num_classes))
            logger.info('Adapter rank for final fully connected layer: ' + str(fc_adapter_rank))
            adapter_ranks = [1] + adapter_ranks + [fc_adapter_rank]
        else:
            adapter_ranks = [0] + adapter_ranks + [0]
    else:
        adapter_ranks = [0  for _ in range(num_blocks + 2)]
    
    fine_tune_joint_model, _, _ \
        = tune_joint_model_hyperparameters_for_a_single_combo(model_type,
                                                              num_blocks,
                                                              [layer_names],
                                                              [None, data_loaders_over_time[-1]],
                                                              num_classes,
                                                              learning_rates,
                                                              regularization_params,
                                                              last_time_step_loss_wts,
                                                              n_epochs,
                                                              fileheader,
                                                              logger,
                                                              mode                              = joint_mode,
                                                              adapter_ranks                     = adapter_ranks,
                                                              adapter_mode                      = adapter_mode,
                                                              side_layers                       = side_layers,
                                                              partial_model_file_name           = prev_erm_file_name,
                                                              use_partial_to_init_final         = use_partial_to_init_final,
                                                              remove_uncompressed_partial_model = False)
    test_acc, _ = compute_joint_model_accuracy_for_a_time_step(fine_tune_joint_model,
                                                               data_loaders_over_time[-1]['test'],
                                                               1)
    if mode not in {'separate', 'separate_with_prev_reg'}:
        logger.info('Test accuracy from ERM on previous time steps: ' + str(prev_erm_test_acc))
    logger.info('Test accuracy from ERM then learning a joint model with t = 0 frozen at ERM and t = 1 with ' 
                + mode + ' modules fit on last time step: ' + str(test_acc))
    
    model_name    = model_type + ' ' + str(num_blocks) + ' blocks'
    df = pd.DataFrame(data    = {'Model'         : [model_name],
                                 'Method'        : [method_name],
                                 'Shift sequence': [shift_sequence_with_sample_size],
                                 'Seed'          : [seed],
                                 'Test accuracy' : [test_acc]},
                      columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy'])
    df.to_csv(method_dir + 'metrics.csv',
              index = False)
    
def visualize_weight_differences(models_over_time,
                                 filename):
    '''
    Plot L2 norm of difference between weights at consecutive time steps for each layer
    @param models_over_time: list of block_models
    @param filename: str, path to save plot
    @return: None
    '''
    layers             = get_plot_layer_names(models_over_time[0].model_type,
                                              models_over_time[0].num_blocks)
    num_time_steps     = len(models_over_time)
    weight_differences = {layer: np.zeros(num_time_steps - 1)
                          for layer in layers}
    for time_idx in range(1, len(models_over_time)):
        for layer_idx in range(len(layers)):
            weight_differences[layers[layer_idx]][time_idx - 1] \
                = np.sum(np.square(models_over_time[time_idx - 1].layers[layer_idx].get_param_vec()
                                   - models_over_time[time_idx].layers[layer_idx].get_param_vec()))
        
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    fig, ax = plt.subplots(nrows   = 2,
                           ncols   = 3,
                           figsize = (6.4*3, 4.8*2),
                           sharex  = True)
    for layer_idx in range(len(layers)):
        ax[int(layer_idx / 3), int(layer_idx % 3)].plot(np.arange(1, num_time_steps),
                                                        weight_differences[layers[layer_idx]],
                                                        linewidth = 3)
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_title(layers[layer_idx])
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_ylim(bottom = 0)
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_xlim([1, num_time_steps - 1])
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_xticks(range(1, num_time_steps))
        if int(layer_idx / 3) == 1:
            ax[int(layer_idx / 3), int(layer_idx % 3)].set_xlabel('Time step')
        if int(layer_idx % 3) == 0:
            ax[int(layer_idx / 3), int(layer_idx % 3)].set_ylabel('L2 norm of weight change')
    
    fig.tight_layout()
    fig.savefig(filename)
    
def run_sequential_fine_tune(dataset_name,
                             model_type,
                             num_blocks,
                             shift_sequence,
                             source_sample_size,
                             target_sample_sizes,
                             final_target_test_size,
                             layers_to_tune,
                             seed                  = 1007,
                             regularization        = 'standard',
                             start_with_imagenet   = False,
                             learning_rate_decay   = 'none',
                             ablate_fine_tune_init = False,
                             test_all_time_steps   = False):
    '''
    Fine tune a single block model on each time step
    Compute test accuracy on test set at final time step
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         for portraits, shift_sequence, source_sample_size, target_sample_sizes, 
                         and final_target_test_size arguments are disregarded
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param layers_to_tune: list of list of str, outer list over time steps, inner list defines combination of layers to tune
                           until convergence before starting next combination of layers at each time step,
                           if outer list is length 1, same layers tuned at all time steps,
                           options for length 1 inner list:
                           - search to find the best combination of layers to tune
                           - surgical_rgn to use relative gradient norm to determine which blocks to tune
                           - surgical_snr to use signal-to-noise ratio to determine which blocks to tune
                           options for each entry of length 1+ inner list:
                           - all to tune all layers
                           - comma-separated combination of layers: conv1,layer1,layer2,layer3,layer4,fc
                           examples of inner lists:
                           - ['all'] is standard fine tuning
                           - ['fc', 'all'] is linear probe then fine tune
                           - ['fc', 'fc,layer4', 'fc,layer4,layer3', 'fc,layer4,layer3,layer2', 
                              'fc,layer4,layer3,layer2,layer1', 'fc,layer4,layer3,layer2,layer1,conv1'] 
                             is gradual unfreezing last -> first
    @param seed: int, for np random generator
    @param regularization: str, options: standard, previous, fisher, fisher_all, previous_l1
                           specify whether L2 regularization is towards 0, towards weights at t-1, 
                           towards weights at t-1 weighted by Fisher, towards weights at all previous t weighted by Fisher,
                           or towards weights at t-1 as L1 instead of L2 regularization
    @param start_with_imagenet: bool, whether to initialize first model with pre-trained model from ImageNet
    @param learning_rate_decay: str, if None tunes learning rate between 1e-3 and 1e-4, 
                                if 'linear' decays by constant factor, i.e. start with 1e-4 and subtract 1e-5 each time
                                or start with 1e-3 and subtract 1e-4 each time,
                                hyperparameter tuning based on val acc at last time step after fitting entire sequence
                                if 'exponential' decays by multiplicative factor,
                                i.e. start with 2e-4 and multiply 0.5 each time
                                or start with 2e-3 and multiply 0.5 each time,
                                or start with 2e-4 and multiply .75 each time
    @param ablate_fine_tune_init: bool, ablate to initialize "fine tuned" model from scratch,
                                  generally use in conjunction with some regularization towards previous model,
                                  otherwise becomes erm_final
    @param test_all_time_steps: bool, whether to evaluate test accuracy at each target step 
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'resnet', 'densenet', 'convnet'}
    assert num_blocks >= 1
    assert regularization in {'standard', 'previous', 'fisher', 'fisher_all', 'previous_l1'}
    assert learning_rate_decay in {'none', 'linear', 'exponential'}
    real_world_datasets = {'portraits'}
    if test_all_time_steps:
        assert dataset_name not in real_world_datasets
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
        num_target_steps = 7
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    if len(layers_to_tune) > 1:
        assert len(layers_to_tune) == num_target_steps
        all_same = True
        for i in range(1, num_target_steps):
            if layers_to_tune[i] != layers_to_tune[0]:
                all_same = False
                break
    else:
        all_same = True
        layers_to_tune = [layers_to_tune[0] for _ in range(num_target_steps)]
    for layers in layers_to_tune:
        if len({'search', 'surgical_rgn', 'surgical_snr'}.intersection(set(layers))) > 0:
            assert len(layers) == 1
    if all_same:
        if layers_to_tune[0] == ['all']:
            method_name = 'sequential_fine_tune'
        elif layers_to_tune[0][0] in {'search', 'surgical_rgn', 'surgical_snr'}:
            method_name = 'sequential_fine_tune_' + layers_to_tune[0][0]
        elif check_layer_option_is_lp_ft(layers_to_tune[0]):
            method_name = 'sequential_linear_probe_then_fine_tune'
        elif check_layer_option_is_last_to_first(layers_to_tune[0]):
            method_name = 'sequential_gradual_unfreeze_last_to_first_fine_tune'
        elif check_layer_option_is_first_to_last(layers_to_tune[0]):
            method_name = 'sequential_gradual_unfreeze_first_to_last_fine_tune'
        else:
            method_name = 'sequential_fine_tune_' + '-'.join(layers_to_tune[0])
    else:
        method_name     = 'sequential_fine_tune_' + ':'.join(['-'.join(layers) for layers in layers_to_tune])
    if ablate_fine_tune_init:
        method_name += '_ablate_init'
    time0_name   = 'erm_time0'
    if regularization == 'previous':
        method_name += '_reg_previous'
    elif regularization == 'fisher':
        method_name += '_reg_previous_fisher'
    elif regularization == 'fisher_all':
        method_name += '_reg_all_previous_fisher'
    elif regularization == 'previous_l1':
        method_name += '_l1_reg_previous'
    if learning_rate_decay != 'none':
        method_name += '_' + learning_rate_decay + '_lr_decay'
    if start_with_imagenet:
        method_name += '_start_with_imagenet'
        time0_name  += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir  = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + method_name + '/' \
                + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    if os.path.exists(method_dir + 'metrics.csv'):
        return # this experiment has already been run
    logging_dir = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_' + method_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Running sequential fine tuning at each time step')
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model architecture: '     + model_type)
    logger.info('# blocks: '               + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: ' 
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    logger.info('Layers to tune: '         + ' : '.join([' - '.join(layers) for layers in layers_to_tune]))
    logger.info('Regularization: '         + regularization)
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    logger.info('Learning rate decay: '    + str(learning_rate_decay))
    logger.info('Ablate fine-tune init: '  + str(ablate_fine_tune_init))
    logger.info('Test at all time steps: ' + str(test_all_time_steps))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    time0_dir = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + time0_name \
              + '/seed' + str(seed) + '/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(time0_dir):
        os.makedirs(time0_dir)
    fileheader       = model_dir + method_name + '_'
    time0_fileheader = time0_dir + time0_name + '_'
    
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger,
                                                    model_dir + 'portraits_examples.pdf')
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = True,
                                                         output_dir       = model_dir,
                                                         test_set_for_all = test_all_time_steps)
    
    if learning_rate_decay == 'none':
        learning_rates                = [1e-3, 1e-4]
        num_learning_rate_sequences   = 1
    elif learning_rate_decay == 'linear':
        starting_learning_rates       = [1e-3, 1e-4]
        learning_rate_decay_constants = [1e-4, 1e-5]
        num_learning_rate_sequences   = 2
    else:
        starting_learning_rates       = [2e-3, 2e-4, 2e-4]
        learning_rate_decay_constants = [.5, .5, .75]
        num_learning_rate_sequences   = 3
    weight_decays  = [0.0001, 0.01]
    dropouts       = [0, 0.5]
    n_epochs       = 100
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    prev_time_model = model_class(model_properties = model_properties,
                                  num_classes      = num_classes)
    if start_with_imagenet:
        prev_time_model = load_pretrained_weights(prev_time_model)
    models_over_time = []
    if regularization.startswith('fisher'):
        prev_time_fisher_infos = []
    
    best_sequence_final_val_acc = -1
    if test_all_time_steps:
        learning_rate_sequence_val_accs  = -1 * np.ones((num_learning_rate_sequences, num_target_steps))
        learning_rate_sequence_test_accs = -1 * np.ones((num_learning_rate_sequences, num_target_steps))
    for learning_rate_sequence_idx in range(num_learning_rate_sequences):
        if learning_rate_decay in {'linear', 'exponential'}:
            learning_rates = [starting_learning_rates[learning_rate_sequence_idx]]
            schedule_fileheader = fileheader + 'init_lr' + str(starting_learning_rates[learning_rate_sequence_idx]) \
                                + '_decay' + str(learning_rate_decay_constants[learning_rate_sequence_idx])
        else:
            schedule_fileheader = fileheader
        for time_step in range(len(data_loaders_over_time)):
            if time_step == 0:
                time_reg_name             = 'standard'
                time_reg_prev_models      = []
                time_reg_fisher_infos     = []
                remove_uncompressed_model = False
                time_step_fileheader      = time0_fileheader
            else:
                remove_uncompressed_model = True
                time_step_fileheader      = schedule_fileheader + 'time' + str(time_step) + '_'
                if regularization == 'standard':
                    time_reg_name         = 'standard'
                    time_reg_prev_models  = []
                    time_reg_fisher_infos = []
                elif regularization == 'previous':
                    time_reg_name         = 'previous'
                    time_reg_prev_models  = [models_over_time[-1]]
                    time_reg_fisher_infos = []
                elif regularization == 'fisher':
                    time_reg_name         = 'fisher'
                    time_reg_prev_models  = [models_over_time[-1]]
                    time_reg_fisher_infos = [prev_time_fisher_infos[-1]]
                elif regularization == 'fisher_all':
                    time_reg_name         = 'fisher'
                    time_reg_prev_models  = models_over_time
                    time_reg_fisher_infos = prev_time_fisher_infos
                else:
                    time_reg_name         = 'previous_l1'
                    time_reg_prev_models  = [models_over_time[-1]]
                    time_reg_fisher_infos = []

            if time_step != 0 and layers_to_tune[time_step - 1][0] in {'search', 'surgical_rgn', 'surgical_snr'}:
                if layers_to_tune[time_step - 1][0] == 'search':
                    combos = define_layer_combos(model_type,
                                                 num_blocks)
                else:
                    ordered_layers = order_layers_to_tune_surgical_metrics(deepcopy(prev_time_model),
                                                                           data_loaders_over_time[time_step]['train'],
                                                                           layers_to_tune[time_step - 1][0][-3:],
                                                                           time_step_fileheader)
                    combos = [','.join(ordered_layers[:layer_idx+1]) for layer_idx in range(len(ordered_layers))]

                best_fine_tune_val_acc = -1
                for combo in combos:
                    fine_tune_model, fine_tune_val_acc \
                        = tune_hyperparameters_for_model(combo,
                                                         deepcopy(prev_time_model),
                                                         data_loaders_over_time[time_step]['train'],
                                                         data_loaders_over_time[time_step]['valid'],
                                                         learning_rates,
                                                         weight_decays,
                                                         dropouts,
                                                         n_epochs,
                                                         time_step_fileheader,
                                                         logger,
                                                         regularization              = time_reg_name,
                                                         regularization_prev_models  = time_reg_prev_models,
                                                         regularization_fisher_infos = time_reg_fisher_infos)
                    if fine_tune_val_acc > best_fine_tune_val_acc:
                        best_combo              = combo
                        best_fine_tune_val_acc  = fine_tune_val_acc
                        best_fine_tune_model    = fine_tune_model

                best_combo_filename = fileheader + 'time' + str(time_step) + '_best_layers_to_tune.json'
                with open(best_combo_filename, 'w') as f:
                    json.dump({'Best layers to tune': best_combo}, f)
                logger.info('Best combination of layers for fine tuning at time ' + str(time_step) + ': ' + best_combo)

            else:
                if time_step == 0:
                    # at time step 0, tune all layers
                    time_step_layers = ['all']
                else:
                    time_step_layers = layers_to_tune[time_step - 1]
                prev_combo_model     = deepcopy(prev_time_model)
                for combo in time_step_layers:
                    best_fine_tune_model, best_fine_tune_val_acc \
                        = tune_hyperparameters_for_model(combo,
                                                         prev_combo_model,
                                                         data_loaders_over_time[time_step]['train'],
                                                         data_loaders_over_time[time_step]['valid'],
                                                         learning_rates,
                                                         weight_decays,
                                                         dropouts,
                                                         n_epochs,
                                                         time_step_fileheader,
                                                         logger,
                                                         regularization              = time_reg_name,
                                                         regularization_prev_models  = time_reg_prev_models,
                                                         regularization_fisher_infos = time_reg_fisher_infos,
                                                         remove_uncompressed_model   = remove_uncompressed_model)
                    prev_combo_model = deepcopy(best_fine_tune_model)
            if ablate_fine_tune_init:
                prev_time_model = model_class(model_properties = model_properties,
                                              num_classes      = num_classes)
            else:
                prev_time_model = best_fine_tune_model
            models_over_time.append(best_fine_tune_model)
            if regularization.startswith('fisher'):
                prev_time_fisher_infos.append(
                    best_fine_tune_model.compute_fisher_info(data_loaders_over_time[time_step]['train']))
                for name in prev_time_fisher_infos[-1]:
                    logger.info('Average Fisher info for ' + name + ' at time ' + str(time_step) + ': ' 
                                + str(float(torch.mean(prev_time_fisher_infos[-1][name]))))

            if learning_rate_decay == 'linear':
                learning_rates = [learning_rates[0] - learning_rate_decay_constants[learning_rate_sequence_idx]]
            elif learning_rate_decay == 'exponential':
                learning_rates = [learning_rates[0] * learning_rate_decay_constants[learning_rate_sequence_idx]]
                
            if test_all_time_steps and time_step != 0:
                time_val_acc  = compute_accuracy(best_fine_tune_model, data_loaders_over_time[time_step]['valid'])
                time_test_acc = compute_accuracy(best_fine_tune_model, data_loaders_over_time[time_step]['test'])
                learning_rate_sequence_val_accs[time_step - 1,  learning_rate_sequence_idx] = time_val_acc
                learning_rate_sequence_test_accs[time_step - 1, learning_rate_sequence_idx] = time_test_acc
        
        if best_fine_tune_val_acc > best_sequence_final_val_acc:
            best_sequence_final_val_acc = best_fine_tune_val_acc
            best_sequence_fine_tune_model = best_fine_tune_model
            if learning_rate_decay in {'linear', 'exponential'}:
                best_starting_learning_rate       = starting_learning_rates[learning_rate_sequence_idx]
                best_learning_rate_decay_constant = learning_rate_decay_constants[learning_rate_sequence_idx]
        
    if learning_rate_decay in {'linear', 'exponential'}:
        logger.info('Best '+ learning_rate_decay + ' learning rate decay sequence starts at '
                     + str(best_starting_learning_rate) + ' with decay factor ' + str(best_learning_rate_decay_constant))
    test_acc = compute_accuracy(best_sequence_fine_tune_model, data_loaders_over_time[-1]['test'])
    logger.info('Test accuracy from sequential fine tuning on final time step: ' + str(test_acc))
    
    visualize_weight_differences(models_over_time,
                                 fileheader + 'weight_differences.pdf')
    
    model_name = model_type + ' ' + str(num_blocks) + ' blocks'
    if test_all_time_steps:
        best_learning_rate_sequence_idxs = np.argmax(learning_rate_sequence_val_accs, axis = 1)
        best_test_accs = [learning_rate_sequence_test_accs[time_idx, best_learning_rate_sequence_idxs[time_idx]]
                          for time_idx in range(num_target_steps)]
        df = pd.DataFrame(data    = {'Model'              : [model_name  for _ in range(num_target_steps)],
                                     'Method'             : [method_name for _ in range(num_target_steps)],
                                     'Shift sequence'     : [shift_sequence_with_sample_size],
                                     'Seed'               : [seed  for _ in range(num_target_steps)],
                                     'Time step'          : [i + 1 for i in range(num_target_steps)],
                                     'Test accuracy'      : best_test_accs,
                                     'Learning rate decay': [learning_rate_decay for _ in range(num_target_steps)]},
                          columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy', 'Learning rate decay'])
        if learning_rate_decay in {'linear', 'exponential'}:
            df['Starting learning rate']     = [starting_learning_rates[best_learning_rate_sequence_idxs[time_idx]]
                                                for time_idx in range(num_target_steps)]
            df['Learning rate decay factor'] = [learning_rate_decay_constants[best_learning_rate_sequence_idxs[time_idx]]
                                                for time_idx in range(num_target_steps)]
    else:
        df = pd.DataFrame(data    = {'Model'         : [model_name],
                                     'Method'        : [method_name],
                                     'Shift sequence': [shift_sequence_with_sample_size],
                                     'Seed'          : [seed],
                                     'Test accuracy' : [test_acc],
                                     'Learning rate decay': [learning_rate_decay]},
                          columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy', 'Learning rate decay'])
        if learning_rate_decay in {'linear', 'exponential'}:
            df['Starting learning rate']     = [best_starting_learning_rate]
            df['Learning rate decay factor'] = [best_learning_rate_decay_constant]
    df.to_csv(method_dir + 'metrics.csv',
              index = False)
    
def run_sequential_side_tune_or_low_rank_adapt(dataset_name,
                                               model_type,
                                               num_blocks,
                                               shift_sequence,
                                               source_sample_size,
                                               target_sample_sizes,
                                               final_target_test_size,
                                               mode,
                                               adapter_rank         = 0,
                                               adapter_mode         = None,
                                               side_layer_sizes     = [],
                                               seed                 = 1007,
                                               start_with_imagenet  = False,
                                               all_layers_efficient = False,
                                               test_all_time_steps  = False):
    '''
    Learn a new side module or low-rank adapter for block model at each time step
    Compute test accuracy on test set at final time step
    Learning rates are tuned among 1e-3 and 1e-4 at each time step
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         for portraits, shift_sequence, source_sample_size, target_sample_sizes, 
                         and final_target_test_size arguments are disregarded
    @param model_type: str, densenet, resnet, or convnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param mode: str, side_tune, low_rank_adapt, 
                 separate_with_init (initialize t = 1 with t = 0 parameters to match fine tune),
                 separate_with_prev_reg (match fine tune ablate init with regularization towards previous),
                 or separate_with_init_and_prev_reg (match fine tune with regularization towards previous)
    @param adapter_rank: int, rank of adapters
    @param adapter_mode: str, add or multiply
    @param side_layer_sizes: str or list of int, number of output channels for each of the intermediate convolution layers 
                             in side module, same for all block layers,
                             str option is "block" for side module to match original block
    @param seed: int, for np random generator
    @param start_with_imagenet: bool, whether to initialize first model with pre-trained model from ImageNet
    @param all_layers_efficient: bool, whether to make non-block modules parameter-efficient if mode is not separate,
                                 side modules for non-block modules are single layer modules,
                                 low-rank adapters for non-block modules are fixed at rank 10,
                                 other than the input conv layer in resnets that can only be rank 1 to be more efficient,
                                 when False, non-block modules are just separate
    @param test_all_time_steps: bool, whether to evaluate test accuracy at each target step 
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'densenet', 'resnet', 'convnet'}
    assert num_blocks > 0
    assert mode in {'side_tune', 'low_rank_adapt', 'separate_with_init', 'separate_with_prev_reg', 
                    'separate_with_init_and_prev_reg'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
        assert adapter_rank > 0
    real_world_datasets = {'portraits'}
    if test_all_time_steps:
        assert dataset_name not in real_world_datasets
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
        num_target_steps = 7
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    joint_mode = mode
    use_partial_to_init_final = False
    if mode.startswith('separate'):
        joint_mode = 'separate'
        if mode == 'separate_with_init':
            method_name = 'sequential_fine_tune_using_frozen_joint'
            use_partial_to_init_final = True
        elif mode == 'separate_with_prev_reg':
            method_name = 'sequential_fine_tune_reg_previous_using_frozen_joint_ablate_init'
        else:
            method_name = 'sequential_fine_tune_reg_previous_using_frozen_joint'
            use_partial_to_init_final = True
    else:
        if mode == 'low_rank_adapt':
            method_name = 'sequential_low_rank_adapt_' + adapter_mode + '_rank' + str(adapter_rank)
        if mode == 'side_tune':
            method_name = 'sequential_side_tune'
            if model_type == 'convnet':
                assert isinstance(side_layer_sizes, str) and side_layer_sizes == 'block'
            if len(side_layer_sizes) > 0:
                if isinstance(side_layer_sizes, str):
                    assert side_layer_sizes == 'block'
                    method_name += '_block'
                else:
                    method_name += '_' + str(len(side_layer_sizes) + 1) + 'layers_size' \
                                 + ', '.join([str(size) for size in side_layer_sizes])
    time0_name   = 'erm_time0'
    if start_with_imagenet:
        method_name += '_start_with_imagenet'
        time0_name  += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir  = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + method_name + '/' \
                + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    if os.path.exists(method_dir + 'metrics.csv'):
        return # this experiment has already been run
    logging_dir = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_' + method_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Running sequential fine tuning with a joint model where parameters for previous time steps are frozen '
                + 'and parameters for new time step are fit at each time step')
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model architecture: '     + model_type)
    logger.info('# blocks: '               + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: '
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Mode: '                   + str(mode))
    if mode == 'low_rank_adapt':
        logger.info('Adapter mode: '       + str(adapter_mode))
        logger.info('Adapter rank: '       + str(adapter_rank))
    if mode == 'side_tune':
        if isinstance(side_layer_sizes, str):
            logger.info('Side modules are blocks')
        else:
            logger.info('Side modules: '       + ('No intermediate layers' if len(side_layer_sizes) == 0
                                                  else 'intermediate output sizes are ' 
                                                  + ', '.join([str(size) for size in side_layer_sizes])))
    logger.info('Seed: '                   + str(seed))
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    logger.info('Making non-block layers parameter-efficient: ' + str(all_layers_efficient))
    logger.info('Test at all time steps: ' + str(test_all_time_steps))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    time0_dir = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + time0_name \
              + '/seed' + str(seed) + '/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(time0_dir):
        os.makedirs(time0_dir)
    fileheader       = model_dir + method_name + '_'
    time0_fileheader = time0_dir + time0_name + '_'
    
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger,
                                                    model_dir + 'portraits_examples.pdf')
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = True,
                                                         output_dir       = model_dir,
                                                         test_set_for_all = test_all_time_steps)
    
    learning_rates = [1e-3, 1e-4]
    time0_weight_decays = [0.0001, 0.01]
    weight_decays  = [0.0001, 0.01]
    adjacent_regs  = [0.01, 0.1, 1]
    dropouts       = [0, 0.5]
    n_epochs       = 100
    regularization_params = {'weight_decay_type' : 'l2', 
                             'weight_decays'     : weight_decays,
                             'adjacent_reg_type' : 'l2', 
                             'adjacent_regs'     : adjacent_regs,
                             'weight_reg_by_time': False,
                             'dropouts'          : dropouts}
    last_time_step_loss_wts = [1]
    
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
        
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks,
                                                                   logger)
    erm_model    = model_class(model_properties = model_properties,
                               num_classes      = num_classes)
    if start_with_imagenet:
        assert num_blocks == 4
        erm_model    = load_pretrained_weights(erm_model)
    erm_model, erm_val_acc = tune_hyperparameters_for_model('all',
                                                            erm_model,
                                                            data_loaders_over_time[0]['train'],
                                                            data_loaders_over_time[0]['valid'],
                                                            learning_rates,
                                                            time0_weight_decays,
                                                            dropouts,
                                                            n_epochs,
                                                            time0_fileheader,
                                                            logger,
                                                            remove_uncompressed_model = False)
    prev_erm_test_acc  = compute_accuracy(erm_model, data_loaders_over_time[-1]['test'])
    prev_file_name     = time0_fileheader + 'all_best_model.pt.gz'
    
    layer_names = get_layer_names(model_type, num_blocks)
    if mode == 'side_tune':
        side_layers     = [side_layer_sizes for _ in range(num_blocks)]
        if all_layers_efficient:
            if model_type == 'convnet':
                side_layers = ['block'] + side_layers + ['block']
            else:
                side_layers = [[]] + side_layers + [[]]
        else:
            side_layers = ['separate'] + side_layers + ['separate']
    else:
        side_layers   = [[] for _ in range(num_blocks + 2)]
    
    if mode == 'low_rank_adapt':
        adapter_ranks   = [adapter_rank for _ in range(num_blocks)]
        if all_layers_efficient:
            fc_adapter_rank = min(adapter_rank,
                                  compute_max_fc_adapter_rank(model_type,
                                                              num_blocks,
                                                              num_classes))
            logger.info('Adapter rank for final fully connected layer: ' + str(fc_adapter_rank))
            adapter_ranks = [1] + adapter_ranks + [fc_adapter_rank]
        else:
            adapter_ranks = [0] + adapter_ranks + [0]
    else:
        adapter_ranks = [0  for _ in range(num_blocks + 2)]
    
    if test_all_time_steps:
        time_step_test_accs = []
    for time_step in range(1, len(data_loaders_over_time)):
        last_time_step_loss_wts = [time_step + 1] # after dividing by number of time steps, this is equivalent to fine-tuning
        time_step_empty_data_loaders = [None for _ in range(time_step)] + [data_loaders_over_time[time_step]]
        time_step_fileheader         = fileheader + 'time' + str(time_step) + '_'
        time_joint_resnet18, _, _ \
            = tune_joint_model_hyperparameters_for_a_single_combo(model_type,
                                                                  num_blocks,
                                                                  [layer_names for _ in range(time_step)],
                                                                  time_step_empty_data_loaders,
                                                                  num_classes,
                                                                  learning_rates,
                                                                  regularization_params,
                                                                  last_time_step_loss_wts,
                                                                  n_epochs,
                                                                  time_step_fileheader,
                                                                  logger,
                                                                  mode                              = joint_mode,
                                                                  adapter_ranks                     = adapter_ranks,
                                                                  adapter_mode                      = adapter_mode,
                                                                  side_layers                       = side_layers,
                                                                  partial_model_file_name           = prev_file_name,
                                                                  use_partial_to_init_final         = use_partial_to_init_final,
                                                                  remove_uncompressed_partial_model = (time_step == 1))
        prev_file_name = time_step_fileheader + 'all_separate_best_model.pt.gz'
        if test_all_time_steps:
            test_acc, _ = compute_joint_model_accuracy_for_a_time_step(time_joint_resnet18,
                                                                       data_loaders_over_time[time_step]['test'],
                                                                       time_step)
            time_step_test_accs.append(test_acc)
            logger.info('Test accuracy from sequential fine tuning a joint model '
                        + 'where parameters for previous time steps are frozen and '
                        + mode + ' module parameters for new time step are fit at time ' + str(time_step) 
                        + ': ' + str(test_acc))
    
    model_name = model_type + ' ' + str(num_blocks) + ' blocks'
    if not test_all_time_steps:
        test_acc, _ = compute_joint_model_accuracy_for_a_time_step(time_joint_resnet18,
                                                                   data_loaders_over_time[-1]['test'],
                                                                   len(data_loaders_over_time) - 1)
        logger.info('Test accuracy from sequential fine tuning a joint model '
                    + 'where parameters for previous time steps are frozen and ' 
                    + mode + ' module parameters for new time step are fit at each time step: ' + str(test_acc))
    
        df = pd.DataFrame(data    = {'Model'         : [model_name],
                                     'Method'        : [method_name],
                                     'Shift sequence': [shift_sequence_with_sample_size],
                                     'Seed'          : [seed],
                                     'Test accuracy' : [test_acc]},
                          columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy'])
    else:
        df = pd.DataFrame(data    = {'Model'         : [model_name  for _ in range(num_target_steps)],
                                     'Method'        : [method_name for _ in range(num_target_steps)],
                                     'Shift sequence': [shift_sequence_with_sample_size for _ in range(num_target_steps)],
                                     'Seed'          : [seed  for _ in range(num_target_steps)],
                                     'Time step'     : [i + 1 for i in range(num_target_steps)],
                                     'Test accuracy' : time_step_test_accs},
                          columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Time step', 'Test accuracy'])
        
    df.to_csv(method_dir + 'metrics.csv',
              index = False)
    
def create_parser():
    '''
    Create argument parser
    @return: ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Fit a model: ERM / IRM / DRO, '
                                                    'ERM / IRM / DRO then some variant of fine-tuning, '
                                                    'or sequential fine-tuning at each time step.'))
    parser.add_argument('--dataset',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify cifar10, cifar100, or portraits.')
    parser.add_argument('--model_type',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify whether to use resnet, convnet, or densenet architecture.')
    parser.add_argument('--num_blocks',
                        action  = 'store',
                        type    = int,
                        help    = 'Specify number of blocks to include in model.')
    parser.add_argument('--shift_sequence',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = ('Specify colon-separated sequence of shifts at each time step. '
                                   'Each step is a comma-separated combination of corruption, rotation, '
                                   'label_flip, label_shift, recolor, recolor_cond, rotation_cond, subpop.'))
    parser.add_argument('--source_sample_size',
                        action  = 'store',
                        type    = int,
                        default = 10000,
                        help    = 'Specify total number of training/validation samples source domain.')
    parser.add_argument('--target_sample_size_seq',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = 'Specify colon-separated sequence of sample sizes for target domains.')
    parser.add_argument('--target_test_size',
                        action  = 'store',
                        type    = int,
                        default = 1000,
                        help    = 'Specify number of test samples for final target domain.')
    parser.add_argument('--gpu_num',
                        action  = 'store',
                        type    = int,
                        default = 1,
                        help    = 'Specify which GPUs to use.')
    parser.add_argument('--method',
                        action  = 'store',
                        type    = str,
                        help    = ('Specify among erm_final, erm_all, erm_all_weighted, erm_fine_tune, '
                                   'sequential_fine_tune, dro_all, irm_all, dro_fine_tune, or irm_fine_tune.'))
    parser.add_argument('--fine_tune_layers',
                        action  = 'store',
                        type    = str,
                        default = 'all',
                        help    = ('Specify which layers to fine tune for each time step. '
                                   '- all (default): tune all layers. '
                                   '- search: try all combinations of layers at each step. '
                                   '- surgical_rgn and surgical_snr will order layers by relative gradient norm '
                                   'and signal-to-noise ratio, respectively, and try tuning different numbers of layers. '
                                   '- Can also specify dash-separated list of comma-separated list of layers. '
                                   'Comma-separated list of layers are tuned together. '
                                   'Dash-separated list corresponds to tuning first layer set in list until convergence, '
                                   'then tuning second layer set etc. '
                                   'Above options do the same procedure at each time step. '
                                   '- Can also specify colon-separated list of different options at each time step. '
                                   'Available shortcuts: '
                                   '- linear_probe_fine_tune = fc-all. '
                                   '- gradual_unfreeze_first_to_last = conv1-conv1,layer1-conv1,layer1,layer2-'
                                   'conv1,layer1,layer2,layer3-conv1,layer1,layer2,layer3,layer4-all. '
                                   '- gradual_unfreeze_last_to_first = fc-fc,layer4-fc,layer4,layer3-fc,layer4,layer3,layer2-'
                                   'fc,layer4,layer3,layer2,layer1-all.'))
    parser.add_argument('--regularization',
                        action  = 'store',
                        type    = str,
                        default = 'standard',
                        help    = ('Specify among standard (L2 reg towards 0), previous (L2 reg towards prev weights), '
                                   'fisher (L2 reg towards prev weights weighted by Fisher info), '
                                   'fisher_all (EWC: L2 reg towards prev weights at all time steps weighted by Fisher info), '
                                   'or previous_l1 (L1 reg towards prev weights).'))
    parser.add_argument('--start_with_imagenet',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to initialize model with ImageNet weights.')
    parser.add_argument('--learning_rate_decay',
                        action  = 'store',
                        type    = str,
                        default = 'exponential',
                        help    = ('Specify none, linear, or exponential for how to decay learning rate '
                                   'in sequential fine-tuning.'))
    parser.add_argument('--seed',
                        action  = 'store',
                        type    = int,
                        default = 1007,
                        help    = 'Specify random seed.')
    parser.add_argument('--fine_tune_mode',
                        action  = 'store',
                        type    = str,
                        default = 'standard',
                        help    = ('Options: '
                                   '- standard to fine-tune the layers in fine_tune_layers. '
                                   '- standard_using_joint to fine-tune all layers by freezing previous time steps '
                                     'and fine tuning the last time step in a joint model with no adjacent regularization, '
                                     'use only for debugging to see if matches ERM final. '
                                   '- standard_using_joint_with_init to also initialize last time step '
                                      'with previous parameters, use only for debugging to see if matches standard. '
                                   '- standard_using_joint_with_prev_reg to see if matches standard with previous '
                                      'regularization and ablated initialization. '
                                   '- standard_using_joint_with_init_and_prev_reg to see if matches standard with '
                                      'previous regularization.'
                                   '- side_tune to use side modules for blocks. '
                                   '- low_rank_adapt to use low-rank adapters for blocks.'))
    parser.add_argument('--adapter_mode',
                        action  = 'store',
                        type    = str,
                        default = None,
                        help    = 'Specify add or multiply for how to implement low-rank adapt modules.')
    parser.add_argument('--adapter_rank',
                        action  = 'store',
                        type    = int,
                        default = 0,
                        help    = 'Specify rank of input and output adapters.')
    parser.add_argument('--side_layer_sizes',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = ('Specify comma-separated list of number of output channels in intermediate convolution '
                                   'layers of side modules. Default is 1 layer that maps input size to output size. '
                                   'Another option is block for each side module to match original block.'))
    parser.add_argument('--ablate_init',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify to initialize fine-tuned model from scratch. '
                                   'Only supported for non-standard regularization and standard fine-tuning'))
    parser.add_argument('--all_layers_efficient',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to make non-block layers also parameter-efficient.')
    parser.add_argument('--test_all_steps',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to record test accuracy for sequential fine-tuning at each time step.')
    return parser

if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    torch.cuda.device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    assert(torch.cuda.is_available())
    
    assert args.fine_tune_mode in {'standard', 'standard_using_joint', 'standard_using_joint_with_init',
                                   'standard_using_joint_with_prev_reg', 'standard_using_joint_with_init_and_prev_reg',
                                   'side_tune', 'low_rank_adapt'}
    mode = args.fine_tune_mode.replace('standard_using_joint', 'separate')
    if args.fine_tune_mode == 'low_rank_adapt':
        assert args.adapter_mode in {'add', 'multiply'}
        assert args.adapter_rank > 0
    if args.fine_tune_mode == 'side_tune' and len(args.side_layer_sizes) > 0:
        if args.side_layer_sizes == 'block':
            side_layer_sizes = args.side_layer_sizes
        else:
            side_layer_sizes = [int(size) for size in args.side_layer_sizes.split(',')]
    else:
        side_layer_sizes = []
    if args.ablate_init:
        assert args.method in {'erm_fine_tune', 'sequential_fine_tune'}
        assert args.regularization != 'standard'
        assert args.fine_tune_mode == 'standard'
    assert args.model_type in {'resnet', 'densenet', 'convnet'}
    assert args.num_blocks >= 1
    
    lp_ft_layers                 = ['fc', 'all']
    layer_names                  = get_layer_names(args.model_type,
                                                   args.num_blocks)
    gradual_first_to_last_layers = [','.join(layer_names[:i]) for i in range(1, len(layer_names) - 1)] + ['all']
    gradual_last_to_first_layers = [','.join(layer_names[len(layer_names) - 1:len(layer_names) - 1 - i:-1])
                                    for i in range(1, len(layer_names) - 1)] + ['all']
    
    real_world_datasets = {'portraits'}
    if args.test_all_steps:
        assert args.method == 'sequential_fine_tune'
        assert args.dataset not in real_world_datasets
    if args.dataset not in real_world_datasets:
        target_sample_sizes = [int(i) for i in args.target_sample_size_seq.split(':')]
    else:
        target_sample_sizes = []
    if args.method in {'erm_final', 'erm_all', 'erm_all_weighted', 'dro_all', 'irm_all'}:
        method = args.method[:3]
        if args.method == 'erm_final':
            loss_weights = 'final'
        elif args.method.endswith('_all'):
            loss_weights = 'equal'
        else:
            loss_weights = 'increasing'
        run_erm(args.dataset,
                args.model_type,
                args.num_blocks,
                args.shift_sequence,
                args.source_sample_size,
                target_sample_sizes,
                args.target_test_size,
                seed                = args.seed,
                loss_weights        = loss_weights,
                start_with_imagenet = args.start_with_imagenet,
                method              = method)
    elif args.method in {'erm_fine_tune', 'dro_fine_tune', 'irm_fine_tune'}:
        method = args.method[:3]
        if args.fine_tune_mode == 'standard':
            layers_to_tune = args.fine_tune_layers.split('-')
            if len(layers_to_tune) == 1:
                if layers_to_tune[0] == 'linear_probe_fine_tune':
                    layers_to_tune = lp_ft_layers
                elif layers_to_tune[0] == 'gradual_unfreeze_first_to_last':
                    layers_to_tune = gradual_first_to_last_layers
                elif layers_to_tune[0] == 'gradual_unfreeze_last_to_first':
                    layers_to_tune = gradual_last_to_first_layers
            run_erm_fine_tune(args.dataset,
                              args.model_type,
                              args.num_blocks,
                              args.shift_sequence,
                              args.source_sample_size,
                              target_sample_sizes,
                              args.target_test_size,
                              layers_to_tune,
                              seed                  = args.seed,
                              regularization        = args.regularization,
                              start_with_imagenet   = args.start_with_imagenet,
                              ablate_fine_tune_init = args.ablate_init,
                              initial_method        = method)
        else:
            assert args.method == 'erm_fine_tune'
            run_erm_side_tune_or_low_rank_adapt(args.dataset,
                                                args.model_type,
                                                args.num_blocks,
                                                args.shift_sequence,
                                                args.source_sample_size,
                                                target_sample_sizes,
                                                args.target_test_size,
                                                mode,
                                                adapter_mode         = args.adapter_mode,
                                                adapter_rank         = args.adapter_rank,
                                                side_layer_sizes     = side_layer_sizes,
                                                seed                 = args.seed,
                                                start_with_imagenet  = args.start_with_imagenet,
                                                all_layers_efficient = args.all_layers_efficient)
    else:
        assert args.method == 'sequential_fine_tune'
        if args.fine_tune_mode == 'standard':
            layers_split_by_time = args.fine_tune_layers.split(':')
            layers_to_tune       = [layers.split('-') for layers in layers_split_by_time]
            for time_idx in range(len(layers_to_tune)):
                if len(layers_to_tune[time_idx]) == 1:
                    if layers_to_tune[time_idx][0] == 'linear_probe_fine_tune':
                        layers_to_tune[time_idx] = lp_ft_layers
                    elif layers_to_tune[time_idx][0] == 'gradual_unfreeze_first_to_last':
                        layers_to_tune[time_idx] = gradual_first_to_last_layers
                    elif layers_to_tune[time_idx][0] == 'gradual_unfreeze_last_to_first':
                        layers_to_tune[time_idx] = gradual_last_to_first_layers
            run_sequential_fine_tune(args.dataset,
                                     args.model_type,
                                     args.num_blocks,
                                     args.shift_sequence,
                                     args.source_sample_size,
                                     target_sample_sizes,
                                     args.target_test_size,
                                     layers_to_tune,
                                     seed                  = args.seed,
                                     regularization        = args.regularization,
                                     start_with_imagenet   = args.start_with_imagenet,
                                     learning_rate_decay   = args.learning_rate_decay,
                                     ablate_fine_tune_init = args.ablate_init,
                                     test_all_time_steps  = args.test_all_steps)
        else:
            run_sequential_side_tune_or_low_rank_adapt(args.dataset,
                                                       args.model_type,
                                                       args.num_blocks,
                                                       args.shift_sequence,
                                                       args.source_sample_size,
                                                       target_sample_sizes,
                                                       args.target_test_size,
                                                       mode,
                                                       adapter_mode         = args.adapter_mode,
                                                       adapter_rank         = args.adapter_rank,
                                                       side_layer_sizes     = side_layer_sizes,
                                                       seed                 = args.seed,
                                                       start_with_imagenet  = args.start_with_imagenet,
                                                       all_layers_efficient = args.all_layers_efficient,
                                                       test_all_time_steps  = args.test_all_steps)