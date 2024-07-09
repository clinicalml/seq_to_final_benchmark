'''
Goal for visualization is to compare the historical and final models learned by each method
Three types of plots:
1. Linear interpolation paths:
    a. Linearly interpolate weights from initial to final model at last time step
    b. Plot interpolation factor against accuracy of interpolated model evaluated on test samples at last time step
    c. Option to merge plots from multiple architectures
2. Projection of weights onto directions of historical shift and limited final sample size:
    a. Define origin with oracle weights
    b. Define x-axis with direction from oracle weights towards model weights from running ERM on historical data
    c. Define y-axis with direction from oracle weights towards model weights from running ERM on limited data from the final step (baseline)
    d. Project weights from each method onto the two directions
    e. Plot separately for each layer and for all layers concatenated
    f. Plot shows point for method that learns a single model on all time periods
    g. Plot shows path from historical to final for methods that learn from all historical data and fine-tune to the final step
    h. Plot shows path through all time steps from t = 0 to t = T for sequential/joint methods
3. Mean of top SVCCA coefficients:
    a. Run SVCCA to obtain correlation coefficients between each method and the oracle
    b. SVCCA is run on the intermediate outputs for each layer separately and on the concatenated outputs from all layers
    c. x-axis is mean of coefficients explaining 50% of variance
    d. y-axis is mean of coefficients explaining 90% of variance
    e. Plots also show paths over time as for projection of weights
Because last two visualizations require an oracle, they do not work for real-world datasets
'''
import os
import sys
import glob
import json
import argparse
import gzip
import shutil

from datetime import datetime
from os.path import dirname, abspath, join

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fit_single_model import compute_accuracy
from run_joint_model_learning import compute_joint_model_accuracy_for_a_time_step

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'data_loaders'))
from load_image_data_with_shifts import load_datasets_over_time
from load_yearbook_data import load_yearbook_data

sys.path.append(join(dirname(dirname(abspath(__file__))), 'model_classes'))
from model_property_helper_functions import (
    get_model_class_and_properties,
    get_joint_model_class_and_properties,
    get_plot_layer_names
)

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from pytorch_model_utils import load_state_dict_from_gz

sys.path.append(join(dirname(dirname(dirname(abspath(__file__)))), 'svcca'))
import cca_core

def get_singular_vecs(activations,
                      explained_variance_levels,
                      logger):
    '''
    Get singular vectors that explain the desired variance levels in the activations
    @param activations: np array, # samples x # hidden dimensions
    @param explained_variance_levels: list of floats, between 0 and 1, take average of first n CCA coefficients,
                                      where n is minimum # required to explain each amount of variance
    @param logger: logger, for INFO messages
    @return: list of np arrays, # singular vectors x # samples, list over variance levels
    '''
    assert np.all(np.array(explained_variance_levels) > 0)
    assert np.all(np.array(explained_variance_levels) <= 1)
    activations    = activations.T
    activations   -= np.mean(activations, axis = 1, keepdims = True)
    U, s, V        = np.linalg.svd(activations, full_matrices = False)
    total_variance = np.sum(s)
    frac_variance  = np.cumsum(s)/total_variance
    singular_vecs  = []
    for variance_level in explained_variance_levels:
        if variance_level == 1:
            num_vectors = s.shape[0]
        else:
            num_vectors = np.searchsorted(frac_variance, variance_level, side = 'right') + 1
        logger.info(str(num_vectors) + ' of ' + str(s.shape[0]) + ' singular vectors needed to explain ' 
                    + str(variance_level) + ' of variance')
        singular_vecs.append(np.dot(s[:num_vectors] * np.eye(num_vectors), V[:num_vectors]))
    return singular_vecs

def compute_mean_svcca_coefs(singular_vecs_1,
                             singular_vecs_2):
    '''
    Compute mean of correlation coefficients between two sets of singular vectors
    @param singular_vecs_1: np array, # singular vectors x # samples
    @param singular_vecs_2: np array, # singular vectors x # samples
    @return float
    '''
    svcca_results = cca_core.get_cca_similarity(singular_vecs_1, singular_vecs_2, epsilon = 1e-10, verbose = False)
    return float(svcca_results['cca_coef1'].mean())
    
def convert_output_to_numpy(out):
    '''
    If torch output is 4-dimensional, reshape to (# samples x height x width) x # output dimensions
    Remove from cuda and convert to numpy
    @param out: torch FloatTensor, # samples x # output dimensions or # samples x height x width x # channels
    @return: np array, # outputs x # output dimensions / channels
    '''
    if len(out.shape) == 4:
        out = torch.reshape(out, (out.shape[0], -1))
    out = out.detach()
    if torch.cuda.is_available():
        out = out.cpu()
    return out.numpy()

def get_joint_model_activations_per_layer_per_time_step(joint_model,
                                                        X):
    '''
    Compute output from each layer from joint model at each time step on the same samples X
    @param joint_model: joint_block_model
    @param X: torch FloatTensor, samples
    @return: list of list of np arrays, outer list over time steps, inner list over layers,
             contains outputs from each layer at each time step
    '''
    assert len(X.shape) == 4
    activations = []
    joint_model.eval()
    for time_step in range(joint_model.num_time_steps):
        time_step_activations = [convert_output_to_numpy(out) for out in joint_model.get_activations_from_all_layers(X, time_step)]
        activations.append(time_step_activations)
    joint_model.train()
    return activations

def compute_model_mean_cca_coefs(model,
                                 test_X,
                                 variance_levels,
                                 oracle_singular_vecs,
                                 logger):
    '''
    Compute mean SVCCA coefficients between a model and oracle at specified variance levels
    Take mean across all layers to represent entire model
    @param model: block_model, model to compare against oracle
    @param test_X: torch FloatTensor, # samples x # channels x H x W
    @param variance_levels: list of floats
    @param oracle_singular_vecs: list of lists of np arrays, outer list over layers, inner list over variance levels,
                                 np arrays are singular vectors for oracle model to compare against, # singular vecs x # samples
    @param logger: logger, for INFO messages
    @return: list of lists of floats, outer list over layers with last entry for entire model,
             inner list over variance levels, mean of SVCCA coefficients
    '''
    num_layers = len(model.layers)
    assert len(oracle_singular_vecs) == num_layers
    assert np.all(np.array([len(oracle_singular_vecs[layer_idx]) == len(variance_levels)
                            for layer_idx in range(num_layers)]))
    model_activations   = [convert_output_to_numpy(out) for out in model.get_activations_from_all_layers(test_X)]
    model_singular_vecs = [get_singular_vecs(model_activations[layer_idx],
                                             variance_levels,
                                             logger)
                           for layer_idx in range(num_layers)]
    model_cca_means     = [[compute_mean_svcca_coefs(model_singular_vecs[layer_idx][coord_idx],
                                                     oracle_singular_vecs[layer_idx][coord_idx])
                            for coord_idx in range(len(variance_levels))]
                           for layer_idx in range(num_layers)]
    model_cca_means.append([np.mean(np.array([model_cca_means[layer_idx][coord_idx]
                                              for layer_idx in range(num_layers)]))
                            for coord_idx in range(len(variance_levels))])
    return model_cca_means

def compute_joint_model_mean_cca_coefs(joint_model,
                                       test_X,
                                       variance_levels,
                                       oracle_singular_vecs,
                                       logger):
    '''
    Compute mean SVCCA coefficients between a model and oracle at specified variance levels
    Take mean across all layers to represent entire model at each time step
    @param joint_model: joint_block_model, model to compare against oracle at each time step
    @param test_X: torch FloatTensor, # samples x # channels x H x W
    @param variance_levels: list of floats
    @param oracle_singular_vecs: list of lists of np arrays, outer list over layers, inner list over variance levels,
                                 np arrays are singular vectors for oracle model to compare against, # singular vecs x # samples
    @param logger: logger, for INFO messages
    @return: list of lists of lists of floats, outer list over layers with last entry for entire model,
             middle list over time steps,
             inner list over variance levels, mean of SVCCA coefficients
    '''
    num_layers = len(joint_model.layers)
    assert len(oracle_singular_vecs) == num_layers
    assert np.all(np.array([len(oracle_singular_vecs[layer_idx]) == len(variance_levels)
                            for layer_idx in range(num_layers)]))
    joint_activations   = get_joint_model_activations_per_layer_per_time_step(joint_model, test_X)
    joint_singular_vecs = [[get_singular_vecs(joint_activations[time_idx][layer_idx],
                                              variance_levels,
                                              logger)
                            for layer_idx in range(num_layers)]
                           for time_idx in range(joint_model.num_time_steps)]
    joint_cca_means     = [[[compute_mean_svcca_coefs(joint_singular_vecs[time_idx][layer_idx][coord_idx],
                                                      oracle_singular_vecs[layer_idx][coord_idx])
                             for coord_idx in range(len(variance_levels))]
                            for time_idx in range(joint_model.num_time_steps)]
                           for layer_idx in range(num_layers)]
    # average over all layers
    joint_cca_means.append([[np.mean(np.array([joint_cca_means[layer_idx][time_idx][coord_idx]
                                               for layer_idx in range(num_layers)]))
                             for coord_idx in range(len(variance_levels))]
                            for time_idx in range(joint_model.num_time_steps)])
    return joint_cca_means

def plot_coord_time_sequences(model_coord_sequences,
                              readable_method_names,
                              plot_title,
                              xlabel,
                              ylabel,
                              output_filename = None,
                              contour_values  = None,
                              oracle_coords   = (1, 1),
                              ax              = None):
    '''
    For each method, plot coordinates representing model weights at each time step vs oracle
    Single models drawn as points in different colors, oracle denoted by x
    Fine-tuning variants drawn as arrow from previous time steps to final time step
    Sequential fine-tuning or joint model plotted as line with arrow towards final time step
    @param model_coord_sequences: dict mapping str to list of lists of floats, str for method name, outer list over time steps, 
                                  inner list contains pair of coordinates to point
    @param readable_method_names: dict mapping str to str, maps key in model_cca_sequences to name for legend,
                                  only methods in this dict are plotted, others in model_cca_sequences are ignored
    @param plot_title: str, plot title
    @param xlabel: str, x-axis label
    @param ylabel: str, y-axis label
    @param output_filename: str, path to save plot
    @param contour_values: np array, # points x 3, each row (x, y, z) contains coordinates (x, y) and value z,
                           no contour map created if values are not specified
    @param oracle_coords: tuple of floats, coordinates for oracle
    @param ax: matplotlib axes, will create plot on these axes, cannot be specified along with output_filename,
               use to create a subplot in a larger figure
    @return: None
    '''
    specify_output_or_ax = 0
    if output_filename is not None:
        specify_output_or_ax += 1
    if ax is not None:
        specify_output_or_ax += 1
    assert specify_output_or_ax == 1
    ax_is_preset = (ax is not None)
    colors         = ['maroon', 'coral', 'darkgoldenrod', 'olivedrab', 'mediumaquamarine', 'steelblue', 'mediumpurple', 'orchid']
    scatter_styles = ['o', '^', '*', 'D']
    line_styles    = ['solid', 'dashed', 'dotted', 'dashdot']
    scatter_idx    = 0
    line_idx       = 0
    if not ax_is_preset:
        fig, ax        = plt.subplots(nrows   = 1,
                                      ncols   = 1,
                                      figsize = (6.4, 6.4))
    
    ax.scatter([oracle_coords[0]],
               [oracle_coords[1]],
               label  = 'ERM final oracle',
               c      = 'black',
               marker = 'x')
    for method_name in readable_method_names:
        if len(model_coord_sequences[method_name]) == 1:
            ax.scatter([model_coord_sequences[method_name][0][0]],
                       [model_coord_sequences[method_name][0][1]],
                       label  = readable_method_names[method_name],
                       c      = colors[scatter_idx % len(colors)],
                       marker = scatter_styles[int(scatter_idx / len(colors))])
            scatter_idx += 1
        else:
            num_time_steps = len(model_coord_sequences[method_name])
            ax.plot([model_coord_sequences[method_name][time_idx][0] for time_idx in range(num_time_steps - 1)],
                    [model_coord_sequences[method_name][time_idx][1] for time_idx in range(num_time_steps - 1)],
                    c         = colors[line_idx % len(colors)],
                    linestyle = line_styles[int(line_idx / len(colors))],
                    label     = readable_method_names[method_name])
            ax.arrow(model_coord_sequences[method_name][num_time_steps - 2][0],
                     model_coord_sequences[method_name][num_time_steps - 2][1],
                     model_coord_sequences[method_name][num_time_steps - 1][0] 
                     - model_coord_sequences[method_name][num_time_steps - 2][0],
                     model_coord_sequences[method_name][num_time_steps - 1][1]
                     - model_coord_sequences[method_name][num_time_steps - 2][1],
                     length_includes_head = True,
                     color                = colors[line_idx % len(colors)],
                     linestyle            = line_styles[int(line_idx / len(colors))])
            line_idx += 1
    ax.legend(loc            = 'upper center',
              bbox_to_anchor = (0.5, -0.1),
              ncol           = 3)
    if contour_values is not None:
        ax.contour(contour_values[0],
                   contour_values[1],
                   contour_values[2],
                   alpha = .5)
    if oracle_coords == (1, 1): # CCA
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_title)
    if not ax_is_preset:
        fig.tight_layout()
        fig.savefig(output_filename)
    
def compute_projection_length(vec1,
                              vec2):
    '''
    Compute length of projection of vector 1 onto vector 2
    @param vec1: np array, (len)
    @param vec2: np array, (len)
    @return: float
    '''
    assert len(vec1) == len(vec2)
    assert len(vec1.shape) == 1
    assert len(vec2.shape) == 1
    return float(np.dot(vec1, vec2)/np.linalg.norm(vec2))

def compute_projection_coords_for_model(model,
                                        final_oracle_layer_vecs,
                                        all_prev_layer_proj_dirs,
                                        final_layer_proj_dirs):
    '''
    Compute projection length of (model weights - oracle weights) in each layer onto the 2 directions
    @param model: block_model
    @param final_oracle_layer_vecs: list of np arrays, flattened and concatenated weights from each layer of oracle model,
                                    assumes same architecture as block_model
    @param all_prev_layer_proj_dirs: list of np arrays, flattened and concatenated projection directions for each layer,
                                     from (weights of ERM on all previous data) - (oracle weights)
    @param final_layer_proj_dirs: list of np arrays, flattened and concatenated projection directions for each layer,
                                  from (weights of ERM on limited final data) - (oracle weights)
    @return: list of lists of floats, outer list over layers with last entry for entire model,
             inner list over 2 projection directions
    '''
    # Compute coordinates for each layer
    layer_coords         = []
    model_all_layer_vecs = []
    for layer_idx in range(len(model.layers)):
        model_vec    = model.get_param_vec(layer_idx) - final_oracle_layer_vecs[layer_idx]
        layer_coord1 = compute_projection_length(model_vec,
                                                 all_prev_layer_proj_dirs[layer_idx])
        layer_coord2 = compute_projection_length(model_vec,
                                                 final_layer_proj_dirs[layer_idx])
        layer_coords.append([layer_coord1, layer_coord2])
        model_all_layer_vecs.append(model_vec)
    
    # Compute coordinates for entire layer
    entire_model_vec         = np.concatenate(model_all_layer_vecs)
    entire_final_oracle_vec  = np.concatenate(final_oracle_layer_vecs)
    entire_all_prev_proj_dir = np.concatenate(all_prev_layer_proj_dirs)
    entire_final_proj_dir    = np.concatenate(final_layer_proj_dirs)
    entire_coord1            = compute_projection_length(entire_model_vec,
                                                         np.concatenate(all_prev_layer_proj_dirs))
    entire_coord2            = compute_projection_length(entire_model_vec,
                                                         np.concatenate(final_layer_proj_dirs))
    layer_coords.append([entire_coord1, entire_coord2])
    return layer_coords
    
def compute_projection_coords_for_joint_model(joint_model,
                                              final_oracle_layer_vecs,
                                              all_prev_layer_proj_dirs,
                                              final_layer_proj_dirs):
    '''
    Compute projection length of (model weights - oracle weights) in each layer onto the 2 directions
    @param joint_model: joint_block_model
    @param final_oracle_layer_vecs: list of np arrays, flattened and concatenated weights from each layer of oracle model,
                                    assumes same architecture as each time step in block_model
    @param all_prev_layer_proj_dirs: list of np arrays, flattened and concatenated projection directions for each layer,
                                     from (weights of ERM on all previous data) - (oracle weights)
    @param final_layer_proj_dirs: list of np arrays, flattened and concatenated projection directions for each layer,
                                  from (weights of ERM on limited final data) - (oracle weights)
    @return: list of lists of lists of floats, outer list over layers with last entry for entire model,
             middle list over time steps,
             inner list over 2 projection directions
    '''
    assert joint_model.mode != 'side_tune'
    # Compute coordinates for each layer
    layer_coords             = []
    model_all_layer_vecs     = [[] for _ in range(joint_model.num_time_steps)]
    for layer_idx in range(len(joint_model.layers)):
        this_layer_coords    = []
        for time_idx in range(joint_model.num_time_steps):
            if time_idx == 0 or joint_model.module_path[time_idx, layer_idx] != joint_model.module_path[time_idx - 1, layer_idx]:
                model_vec    = joint_model.get_param_vec(layer_idx, time_idx) - final_oracle_layer_vecs[layer_idx]
                layer_coord1 = compute_projection_length(model_vec,
                                                         all_prev_layer_proj_dirs[layer_idx])
                layer_coord2 = compute_projection_length(model_vec,
                                                         final_layer_proj_dirs[layer_idx])
            # otherwise no change since previous time step
            this_layer_coords.append([layer_coord1, layer_coord2])
            model_all_layer_vecs[time_idx].append(model_vec)
        layer_coords.append(this_layer_coords)
    
    # Compute coordinates for entire layer
    entire_final_oracle_vec  = np.concatenate(final_oracle_layer_vecs)
    entire_all_prev_proj_dir = np.concatenate(all_prev_layer_proj_dirs)
    entire_final_proj_dir    = np.concatenate(final_layer_proj_dirs)
    entire_coords            = []
    for time_idx in range(joint_model.num_time_steps):
        entire_model_vec         = np.concatenate(model_all_layer_vecs[time_idx])
        entire_coord1            = compute_projection_length(entire_model_vec,
                                                             np.concatenate(all_prev_layer_proj_dirs))
        entire_coord2            = compute_projection_length(entire_model_vec,
                                                             np.concatenate(final_layer_proj_dirs))
        entire_coords.append([entire_coord1, entire_coord2])
    layer_coords.append(entire_coords)
    return layer_coords

def compute_interpolated_accuracies(prev_model,
                                    curr_model,
                                    model_class,
                                    test_data_loader,
                                    interpolation_vals):
    '''
    Compute accuracies when linearly interpolating weights from prev_model to curr_model
    @param prev_model: instance of model_class
    @param curr_model: instance of model_class
    @param model_class: class of block_models
    @param test_data_loader: pytorch DataLoader, samples to evaluate
    @param interpolation_vals: list of floats, increasing from 0 to 1 exclusive, e.g. create model with weights at (1 - val) * prev + val * cur
    @return: list of floats, accuracies starting at prev_model, then interpolation_vals, and ending at curr_model
    '''
    assert np.all(np.array(interpolation_vals) < 1)
    assert np.all(np.array(interpolation_vals) > 0)
    assert np.all(np.array(interpolation_vals[1:]) - np.array(interpolation_vals[:-1]) > 0)
    accuracies = [float(compute_accuracy(prev_model,
                                         test_data_loader))]
                               
    for interpolation_val in interpolation_vals:
        interpolated_model = model_class.interpolate_weights(prev_model,
                                                             curr_model,
                                                             interpolation_val)
        accuracies.append(float(compute_accuracy(interpolated_model,
                                                 test_data_loader)))
    accuracies.append(float(compute_accuracy(curr_model,
                                             test_data_loader)))
    return accuracies

def compute_interpolated_accuracies_for_joint_model(joint_model,
                                                    t,
                                                    test_data_loader,
                                                    interpolation_vals):
    '''
    Compute accuracies when linearly interpolating weights in the joint model from t - 1 to t
    For side modules, this means interpolating the side module weights from 0 to the final side module
    @param joint_model: joint_block_model
    @param t: int, time step
    @param test_data_loader: pytorch DataLoader, samples to evaluate
    @param interpolation_vals: list of floats, increasing from 0 to 1 exclusive, e.g. create model with weights at (1 - val) * prev + val * cur
    @return: list of floats, accuracies starting at prev_model, then interpolation_vals, and ending at curr_model
    '''
    assert joint_model.mode in {'separate', 'side_tune'}
    assert t > 0 and t < joint_model.num_time_steps

    accuracies = [float(compute_joint_model_accuracy_for_a_time_step(joint_model,
                                                                     test_data_loader,
                                                                     t - 1)[0])]
    for interpolation_val in interpolation_vals:
        interpolated_model = joint_model.interpolate_weights(t,
                                                             interpolation_val)
        if joint_model.mode == 'separate':
            # interpolated_model is a single block_model
            accuracies.append(float(compute_accuracy(interpolated_model,
                                                 test_data_loader)))
        else:
            # interpolated_model is a joint model with different side weights
            accuracies.append(float(compute_joint_model_accuracy_for_a_time_step(interpolated_model,
                                                                                 test_data_loader,
                                                                                 t)[0]))
    accuracies.append(float(compute_joint_model_accuracy_for_a_time_step(joint_model,
                                                                         test_data_loader,
                                                                         t)[0]))
    return accuracies

def visualize_interpolation_accuracies(dataset_name,
                                       model_type,
                                       num_blocks,
                                       shift_sequence,
                                       models_to_include,
                                       readable_method_names,
                                       plot_title,
                                       seed):
    '''
    For each model type, linearly interpolate the weights from t = T - 1 to t = T and compute the loss at time T
    For side modules, linearly interpolate from 0 to the final weight since that is equivalent to no side module to new side module
    @param dataset_name: str, cifar10, cifar100, or portraits
    @param model_type: str, densenet, convnet, or resnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of corruption, rotation, label_flip, 
                           label_shift, rotation_cond, recoloring, recoloring_cond, subpop
    @param models_to_include: list of str, names of model directories
    @param readable_method_names: list of str, names of models for plot legend
    @param plot_title: str, title of plot
    @param seed: int, seed for data generation
    @return: None
    '''
    for model_name in models_to_include:
        assert not (model_name.endswith('_all') or model_name.endswith('_final'))
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert len(models_to_include) == len(readable_method_names)
    assert model_type in {'densenet', 'convnet', 'resnet'}
    if dataset_name == 'portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 20
    else:
        num_classes = 1000

    if dataset_name == 'portraits':
        num_target_steps = 7
        shift_sequence_with_sample_size = dataset_name
    else:
        shift_sequence_split = shift_sequence.split(':')
        num_target_steps     = len(shift_sequence_split)
        if num_target_steps > 3:
            # longest rotation sequence
            assert num_target_steps == 7
            source_sample_size = 4000
            target_sample_sizes = [2000, 2000, 2000, 2000, 2000, 2000, 4000]
        else:
            source_sample_size   = 6000
            target_sample_sizes  = [4000, 6000, 4000]
        shifts_with_sample_size = [shift_sequence_split[i] + str(target_sample_sizes[i])
                               for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    num_time_steps       = num_target_steps + 1

    experiment_dir    = config.output_dir + model_type + '_' + str(num_blocks) + 'blocks_' + dataset_name + '_experiment/'
    
    visualization_dir = experiment_dir + 'loss_interpolation_plots/' + shift_sequence_with_sample_size \
                      + '/seed' + str(seed) + '/'
    logging_dir       = visualization_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'visualize_parameters_over_time_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Dataset: '           + dataset_name)
    if dataset_name != 'portraits':
        logger.info('Shift sequence: '    + shift_sequence)
    logger.info('Representation: Loss interpolation')
    logger.info('Seed: '              + str(seed))
    logger.info('Model type: '        + model_type)
    logger.info('# of blocks: '       + str(num_blocks))
    logger.info('Model directories: ' + '; '.join(models_to_include))
    logger.info('Model names: '       + '; '.join(readable_method_names))
    metrics_summary_df = pd.read_csv(config.output_dir + 'metrics_summary.csv')

    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger)
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         target_sample_sizes[-1],
                                                         seed             = seed,
                                                         visualize_shifts = False,
                                                         output_dir       = experiment_dir)
    test_data_loader = data_loaders_over_time[-1]['test']
    
    interpolation_vals = [.01 * i for i in range(1, 100)]
    accuracies_to_plot_json = visualization_dir + 'interpolated_accuracies.json'
    if os.path.exists(accuracies_to_plot_json):
        with open(accuracies_to_plot_json, 'r') as f:
            accuracies_to_plot = json.load(f)
    else:
        accuracies_to_plot = dict()
    
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks)
    joint_model_class, _, _       = get_joint_model_class_and_properties(model_type,
                                                                         num_blocks)

    for model_name in models_to_include:
        if model_name in accuracies_to_plot:
            continue
        model_dir  = experiment_dir + model_name + '/' + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
        if 'sequential_fine_tune' in model_name:
            # sequential_fine_tune variants: load time t = T - 1 and t = T
            prev_model  = model_class(model_properties = model_properties,
                                      num_classes = num_classes)
            curr_model  = model_class(model_properties = model_properties,
                                      num_classes = num_classes)

            model_metrics_summary_df \
                = metrics_summary_df.loc[np.logical_and.reduce((metrics_summary_df['Method'] == model_name,
                                                                metrics_summary_df['Shift sequence'] == shift_sequence_with_sample_size,
                                                                metrics_summary_df['Seed'] == seed))]
            best_init_lr  = model_metrics_summary_df['Starting learning rate'].values[0]
            best_lr_decay = model_metrics_summary_df['Learning rate decay factor'].values[0]
            if model_name.endswith('_lr_decay'):
                prev_filename = model_dir + model_name + '_init_lr' + str(best_init_lr) + '_decay' + str(best_lr_decay) \
                              + 'time' + str(num_time_steps - 2) + '_all_best_model.pt.gz'
                curr_filename = model_dir + model_name + '_init_lr' + str(best_init_lr) + '_decay' + str(best_lr_decay) \
                              + 'time' + str(num_time_steps - 1) + '_all_best_model.pt.gz'
            else:
                prev_filename  = model_dir + model_name + '_time' + str(num_time_steps - 2) + '_all_best_model.pt.gz'
                curr_filename  = model_dir + model_name + '_time' + str(num_time_steps - 1) + '_all_best_model.pt.gz'
            prev_model  = load_state_dict_from_gz(prev_model,
                                                  prev_filename)
            curr_model  = load_state_dict_from_gz(curr_model,
                                                  curr_filename)
            if torch.cuda.is_available():
                prev_model = prev_model.cuda()
                curr_model = curr_model.cuda()
            accuracies_to_plot[model_name] = compute_interpolated_accuracies(prev_model,
                                                                             curr_model,
                                                                             model_class,
                                                                             test_data_loader,
                                                                             interpolation_vals)
            
        elif 'fine_tune' in model_name:
            # erm_fine_tune, irm_fine_tune, dro_fine_tune, erm_linear_probe_then_fine_tune
            prev_model = model_class(model_properties = model_properties,
                                     num_classes = num_classes)
            curr_model  = model_class(model_properties = model_properties,
                                      num_classes = num_classes)

            all_prev_method = model_name[:3] # erm, irm, or dro
            all_prev_model  = model_class(model_properties = model_properties,
                                          num_classes = num_classes)
            if all_prev_method == 'erm':
                all_prev_file_name_part = all_prev_method + '_all_prev_all'
            else:
                all_prev_file_name_part = all_prev_method + '_all_prev_' + all_prev_method + '_all'
            all_prev_filename  = experiment_dir + all_prev_method + '_all_prev/' + shift_sequence_with_sample_size \
                               + '/seed' + str(seed) + '/models/' + all_prev_file_name_part + '_best_model.pt.gz'
            all_prev_model  = load_state_dict_from_gz(all_prev_model,
                                                      all_prev_filename,
                                                      remove_uncompressed_file = False)

            final_model = model_class(model_properties = model_properties,
                                      num_classes = num_classes)
            adapted_regex    = model_dir + model_name + '_*_best_model.pt.gz'
            adapted_filename_options = glob.glob(adapted_regex)
            final_filename = adapted_filename_options[0]
            final_model = load_state_dict_from_gz(final_model,
                                                  final_filename)
            if torch.cuda.is_available():
                all_prev_model = all_prev_model.cuda()
                final_model    = final_model.cuda()
            accuracies_to_plot[model_name] = compute_interpolated_accuracies(all_prev_model,
                                                                             final_model,
                                                                             model_class,
                                                                             test_data_loader,
                                                                             interpolation_vals)
            
        else:
            # sequential_side_tune, erm_side_tune, joint_model_side_tune, or standard joint_model
            # load final model for each of these approaches and interpolate between the last two time steps
            assert model_name.startswith('joint_model') or model_name.startswith('sequential_side_tune') \
                or model_name.startswith('erm_side_tune')
            side_layer_sizes = [[] for _ in range(num_blocks + 2)]
            if 'side_tune' in model_name:
                mode = 'side_tune'
                if 'block' in model_name:
                    side_layer_sizes = [[]] + ['block' for _ in range(num_blocks)] + [[]]
            else:
                mode = 'separate'
            model_name_last_part = model_name.split('_')[-1]
            if 'all' in model_name_last_part or 'conv1' in model_name_last_part or 'layer' in model_name_last_part \
                or 'fc' in model_name_last_part:
                separate_layers_str = model_name_last_part
                separate_layers = [layers.split(',') for layers in model_name_last_part.split(':')]
                model_dir = experiment_dir + '_'.join(model_name.split('_')[:-1]) + '/' \
                          + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
            else:
                separate_layers_str = 'all'
                empty_model = model_class(model_properties = model_properties,
                                          num_classes = num_classes)
                separate_layers = [empty_model.layer_names]
            if model_name.startswith('erm_'):
                joint_num_time_steps = 2
            else:
                joint_num_time_steps = num_time_steps
            if len(separate_layers) == 1:
                separate_layers = [separate_layers[0] for _ in range(joint_num_time_steps - 1)]
                if separate_layers_str != 'all':
                    separate_layers_str = ':'.join([separate_layers_str for _ in range(joint_num_time_steps - 1)])
            joint_model  = joint_model_class(num_time_steps   = joint_num_time_steps,
                                             separate_layers  = separate_layers,
                                             model_properties = model_properties,
                                             num_classes      = num_classes,
                                             mode             = mode,
                                             side_layers      = side_layer_sizes)
            if model_name.startswith('joint_model'):
                model_fileheader = 'joint_model'
            else:
                model_fileheader = model_name
            if model_name.startswith('sequential_'):
                joint_filename = model_dir + model_fileheader + '_time' + str(joint_num_time_steps - 1) + '_' + separate_layers_str \
                               + '_separate_best_model.pt.gz'
            else:
                joint_filename = model_dir + model_fileheader + '_' + separate_layers_str + '_separate_best_model.pt.gz'
                if not os.path.exists(joint_filename):
                    empty_model = model_class(model_properties = model_properties,
                                          num_classes = num_classes)
                    this_separate_layers = empty_model.layer_names
                    this_separate_layers_str = ','.join(this_separate_layers)
                    joint_filename = model_dir + model_fileheader + '_' + this_separate_layers_str + '_separate_best_model.pt.gz'
            joint_model  = load_state_dict_from_gz(joint_model,
                                                   joint_filename)
            if torch.cuda.is_available():
                joint_model = joint_model.cuda()
            accuracies_to_plot[model_name] = compute_interpolated_accuracies_for_joint_model(joint_model,
                                                                                             joint_num_time_steps - 1,
                                                                                             test_data_loader,
                                                                                             interpolation_vals)
    with open(accuracies_to_plot_json, 'w') as f:
        json.dump(accuracies_to_plot, f)

    all_interpolation_vals = np.tile([0] + interpolation_vals + [1],
                                     len(models_to_include))
    all_accuracy_vals      = np.concatenate([accuracies_to_plot[model_name] for model_name in models_to_include])
    all_methods            = np.repeat(np.array(readable_method_names),
                                       len(interpolation_vals) + 2)
    all_historical         = np.where(np.char.startswith(all_methods, 'ERM'),
                                      'ERM start',
                                      np.where(np.logical_or(np.char.startswith(all_methods, 'IRM'),
                                                             np.char.startswith(all_methods, 'DRO')),
                                               'IRM/DRO start',
                                               np.where(np.logical_or.reduce((np.char.startswith(all_methods, 'SFT'),
                                                                              np.char.startswith(all_methods, 'sequential'),
                                                                              all_methods == 'EWC')),
                                                        'Sequential',
                                                        'Joint')))
        
    accuracies_to_plot_df = pd.DataFrame(data    = {'Linear Interpolation': all_interpolation_vals,
                                                    'Accuracy'            : all_accuracy_vals,
                                                    'Method'              : all_methods,
                                                    'Historical'          : all_historical},
                                         columns = ['Linear Interpolation', 'Accuracy', 'Method', 'Historical'])
    accuracies_to_plot_df.to_csv(visualization_dir + 'interpolated_accuracies.csv',
                                 index = False)

    plt.clf()
    fig, ax = plt.subplots()
    sns.lineplot(data        = accuracies_to_plot_df,
                 x           = 'Linear Interpolation',
                 y           = 'Accuracy',
                 hue         = 'Method',
                 style       = 'Historical',
                 hue_order   = readable_method_names,
                 style_order = ['ERM start', 'IRM/DRO start', 'Sequential', 'Joint'],
                 dashes      = {'ERM start'    : '',
                                'IRM/DRO start': (5, 5),
                                'Sequential'   : (3, 3),
                                'Joint'        : (1, 1)},
                 palette     = 'colorblind',
                 ax          = ax)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, min(1, accuracies_to_plot_df['Accuracy'].max() + .05)])
    ax.set_title(plot_title)
    handles, labels = ax.get_legend_handles_labels()
    style_index = labels.index('Historical')
    for handle, label in zip(handles[1:style_index], labels[1:style_index]):
        if label.startswith('ERM'):
            handle.set_linestyle((0, ()))
        elif label.startswith('IRM') or label.startswith('DRO'):
            handle.set_linestyle((0, (5, 5)))
        elif label.startswith('SFT') or label.startswith('sequential'):
            handle.set_linestyle((0, (3, 3)))
        else:
            handle.set_linestyle((0, (1, 1)))
    ax.legend(handles = handles[1:style_index],
              labels  = labels[1:style_index],
              loc='upper center',
              bbox_to_anchor = (0.5, -0.17),
              ncol           = 3)
    fig.subplots_adjust(bottom = .35, left = .17, right = .83)
    fig.savefig(visualization_dir + shift_sequence_with_sample_size + '_' + model_type + '_' + str(num_blocks) 
                + 'blocks_interpolated_accuracies.pdf')
                       
def visualize_parameters_over_time(dataset_name,
                                   model_type,
                                   num_blocks,
                                   shift_sequence,
                                   representation,
                                   models_to_include,
                                   readable_method_names,
                                   plot_title,
                                   seed):
    '''
    For CCA representation, visualize mean of CCA coefficients explaining top 50% and top 90% of variance
    at each time step in sequence.
    Each model is compared to oracle and evaluated on test data from final time step.
    For weight projection representation, visualize projection of difference from oracle parameters
    onto difference between oracle and model fit with ERM on all previous data
    and difference between oracle and model fit with ERM on limited final data.
    @param dataset_name: str, cifar10 or cifar100
    @param model_type: str, densenet, convnet, or resnet
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of corruption, rotation, label_flip, 
                           label_shift, rotation_cond, recoloring, recoloring_cond, subpop
    @param representation: str, cca to define coordinates as mean of CCA coefficients explaining 50% and 90% of variance,
                                weight_proj to define coordinates as projections onto directions 
                                reflecting limitations of historical shift and limited data from the final time step
    @param models_to_include: list of str, names of model directories
    @param readable_method_names: list of str, names of models for plot legend
    @param plot_title: str, title of plot
    @param seed: int, seed for data generation
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100'}
    assert representation in {'cca', 'weight_proj'}
    assert len(models_to_include) == len(readable_method_names)
    assert model_type in {'densenet', 'convnet', 'resnet'}
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 20
    else:
        num_classes = 1000
    shift_sequence_split = shift_sequence.split(':')
    num_target_steps     = len(shift_sequence_split)
    source_sample_size   = 6000
    target_sample_sizes  = [4000, 6000, 4000]
    oracle_sample_sizes  = [4000, 6000, 20000]
    shifts_with_sample_size = [shift_sequence_split[i] + str(target_sample_sizes[i])
                               for i in range(num_target_steps)]
    oracle_shifts_with_sample_size  = [shift_sequence_split[i] + str(oracle_sample_sizes[i])
                                       for i in range(num_target_steps)]
    shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    oracle_shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(oracle_shifts_with_sample_size)
    has_equivalent_sequence = False
    if shift_sequence in {'corruption:label_flip:rotation', 'rotation:corruption:label_flip'}:
        has_equivalent_sequence = True
        if shift_sequence == 'corruption:label_flip:rotation':
            equivalent_final_shift_sequence = 'rotation:corruption:label_flip'
        else:
            equivalent_final_shift_sequence = 'corruption:label_flip:rotation'
        equivalent_shift_sequence_split = equivalent_final_shift_sequence.split(':')
        equivalent_shifts_with_sample_size = [equivalent_shift_sequence_split[i] + str(target_sample_sizes[i])
                                              for i in range(num_target_steps)]
        oracle_equivalent_shifts_with_sample_size = [equivalent_shift_sequence_split[i] + str(oracle_sample_sizes[i])
                                                     for i in range(num_target_steps)]
        equivalent_shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(equivalent_shifts_with_sample_size)
        equivalent_oracle_shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' \
                                                          + ':'.join(oracle_equivalent_shifts_with_sample_size)
    experiment_dir    = config.output_dir + model_type + '_' + str(num_blocks) + 'blocks_' + dataset_name + '_experiment/'
    
    visualization_dir = experiment_dir + representation + '_over_time_plots/' + shift_sequence_with_sample_size \
                      + '/seed' + str(seed) + '/'
    logging_dir       = visualization_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'visualize_parameters_over_time_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Dataset: '           + dataset_name)
    logger.info('Shift sequence: '    + shift_sequence)
    logger.info('Representation: '    + representation)
    logger.info('Seed: '              + str(seed))
    logger.info('Model type: '        + model_type)
    logger.info('# of blocks: '       + str(num_blocks))
    logger.info('Model directories: ' + '; '.join(models_to_include))
    logger.info('Model names: '       + '; '.join(readable_method_names))
    metrics_summary_df = pd.read_csv(config.output_dir + 'metrics_summary.csv')
    
    if representation == 'cca':
        # load test samples
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         target_sample_sizes[-1],
                                                         seed             = seed,
                                                         visualize_shifts = False,
                                                         output_dir       = experiment_dir)
        test_X = data_loaders_over_time[-1]['test'].dataset.tensors[0]
        variance_levels_to_plot = [0.5, 0.9]
    
    model_class, model_properties = get_model_class_and_properties(model_type,
                                                                   num_blocks)
    joint_model_class, _, _       = get_joint_model_class_and_properties(model_type,
                                                                         num_blocks)
    need_to_load_oracle           = True
    erm_all_prev_coords           = None
    erm_time0_coords              = None
    erm_exp_lr_decay_time0_coords = None
    
    # for each model in models_to_include, load model, compute CCA coefficients, and save sequence to plot
    coords_json_file_name = visualization_dir + representation + '_over_time.json'
    if os.path.exists(coords_json_file_name):
        with open(coords_json_file_name, 'r') as f:
            model_coords_sequences = json.load(f)
    else:
        model_coords_sequences = [dict() for i in range(num_blocks + 3)] # list over layers + entire model
    
    for model_name in models_to_include:
        if model_name in model_coords_sequences[0]:
            continue # model coordinates loaded from file
        
        if need_to_load_oracle:
            need_to_load_oracle = False
            # load oracle model and compute singular vectors for cca or projection directions for weight_decomp
            oracle_model      = model_class(model_properties = model_properties,
                                            num_classes = num_classes)
            oracle_filename      = experiment_dir + 'erm_final/' + oracle_shift_sequence_with_sample_size + '/seed' + str(seed) \
                                 + '/models/erm_final_all_best_model.pt.gz'
            if not os.path.exists(oracle_filename) and has_equivalent_sequence:
                oracle_filename  = experiment_dir + 'erm_final/' + equivalent_oracle_shift_sequence_with_sample_size + '/seed' + str(seed) \
                                 + '/models/erm_final_all_best_model.pt.gz'
            oracle_model      = load_state_dict_from_gz(oracle_model,
                                                        oracle_filename)
            if representation == 'cca':
                oracle_activations   = [convert_output_to_numpy(out) for out in oracle_model.get_activations_from_all_layers(test_X)]
                oracle_singular_vecs = [get_singular_vecs(oracle_activations[layer_idx],
                                                          variance_levels_to_plot,
                                                          logger)
                                        for layer_idx in range(len(oracle_activations))]
            else:
                oracle_weights  = [oracle_model.get_param_vec(layer_idx)
                                   for layer_idx in range(len(oracle_model.layers))]
                erm_all_prev_model = model_class(model_properties = model_properties,
                                                 num_classes = num_classes)
                erm_all_prev_filename  = experiment_dir + 'erm_all_prev/' + shift_sequence_with_sample_size \
                                       + '/seed' + str(seed) + '/models/erm_all_prev_all_best_model.pt.gz'
                erm_all_prev_model  = load_state_dict_from_gz(erm_all_prev_model,
                                                              erm_all_prev_filename,
                                                              remove_uncompressed_file = False)
                erm_all_prev_model_weights = [erm_all_prev_model.get_param_vec(layer_idx)
                                              for layer_idx in range(len(erm_all_prev_model.layers))]
                all_prev_proj_dirs = [erm_all_prev_model_weights[layer_idx] - oracle_weights[layer_idx]
                                      for layer_idx in range(len(oracle_model.layers))]
                final_model = model_class(model_properties = model_properties,
                                          num_classes = num_classes)
                final_filename = experiment_dir + 'erm_final/' + shift_sequence_with_sample_size \
                               + '/seed' + str(seed) + '/models/erm_final_all_best_model.pt.gz'
                if not os.path.exists(final_filename) and has_equivalent_sequence:
                    final_filename = experiment_dir + 'erm_final/' + equivalent_shift_sequence_with_sample_size \
                                   + '/seed' + str(seed) + '/models/erm_final_all_best_model.pt.gz'
                final_model = load_state_dict_from_gz(final_model,
                                                      final_filename)
                final_model_weights = [final_model.get_param_vec(layer_idx)
                                       for layer_idx in range(len(final_model.layers))]
                final_proj_dirs     = [final_model_weights[layer_idx] - oracle_weights[layer_idx]
                                       for layer_idx in range(len(final_model.layers))]
        
        model_dir  = experiment_dir + model_name + '/' + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
        if has_equivalent_sequence:
            equivalent_model_dir  = experiment_dir + model_name + '/' + equivalent_shift_sequence_with_sample_size \
                                  + '/seed' + str(seed) + '/models/'
        
        if model_name.endswith('_all') or model_name.endswith('_final') or (model_name.endswith('_fine_tune') and 'sequential' not in model_name):
            adapted_model = model_class(model_properties = model_properties,
                                        num_classes = num_classes)
            if 'surgical' in model_name:
                surgical_layers_json = model_dir + model_name + '_fine_tune_best_layers_to_tune.json'
                with open(surgical_layers_json, 'r') as f:
                    layer_combo  = json.load(f)['Best layers to tune']
                adapted_filename = model_dir + model_name +'_fine_tune_' + layer_combo + '_best_model.pt.gz'
            else:
                adapted_regex    = model_dir + model_name + '_*_best_model.pt.gz'
                adapted_filename_options = glob.glob(adapted_regex)
                if len(adapted_filename_options) == 0 and model_name.endswith('_final') and has_equivalent_sequence:
                    adapted_regex    = equivalent_model_dir + model_name + '_*_best_model.pt.gz'
                    adapted_filename_options = glob.glob(adapted_regex)
                adapted_filename = adapted_filename_options[0]
            adapted_model  = load_state_dict_from_gz(adapted_model,
                                                     adapted_filename)
            if representation == 'cca':
                adapted_coords = compute_model_mean_cca_coefs(adapted_model,
                                                              test_X,
                                                              variance_levels_to_plot,
                                                              oracle_singular_vecs,
                                                              logger)
            else:
                adapted_coords = compute_projection_coords_for_model(adapted_model,
                                                                     oracle_weights,
                                                                     all_prev_proj_dirs,
                                                                     final_proj_dirs)
            
            if not (model_name.endswith('_all') or model_name.endswith('_final')):
                if model_name.startswith('erm'):
                    if erm_all_prev_coords is None:
                        if representation == 'cca':
                            # load ERM on all previous time steps and compute CCA coefficients
                            erm_all_prev_model  = model_class(model_properties = model_properties,
                                                              num_classes = num_classes)
                            erm_all_prev_filename  = experiment_dir + 'erm_all_prev/' + shift_sequence_with_sample_size \
                                                   + '/seed' + str(seed) + '/models/erm_all_prev_all_best_model.pt.gz'
                            erm_all_prev_model  = load_state_dict_from_gz(erm_all_prev_model,
                                                                          erm_all_prev_filename,
                                                                          remove_uncompressed_file = False)
                            erm_all_prev_coords = compute_model_mean_cca_coefs(erm_all_prev_model,
                                                                               test_X,
                                                                               variance_levels_to_plot,
                                                                               oracle_singular_vecs,
                                                                               logger)
                        else:
                            erm_all_prev_coords = compute_projection_coords_for_model(erm_all_prev_model,
                                                                                      oracle_weights,
                                                                                      all_prev_proj_dirs,
                                                                                      final_proj_dirs)
                    all_prev_coords = erm_all_prev_coords
                else:
                    all_prev_method = model_name[:3] # irm or dro
                    # load IRM/DRO on all previous time steps and compute CCA coefficients
                    all_prev_model  = model_class(model_properties = model_properties,
                                                  num_classes = num_classes)
                    all_prev_filename  = experiment_dir + all_prev_method + '_all_prev/' + shift_sequence_with_sample_size \
                                       + '/seed' + str(seed) + '/models/' + all_prev_method + '_all_prev_' + all_prev_method \
                                       + '_all_best_model.pt.gz'
                    all_prev_model  = load_state_dict_from_gz(all_prev_model,
                                                              all_prev_filename,
                                                              remove_uncompressed_file = False)
                    if representation == 'cca':
                        all_prev_coords = compute_model_mean_cca_coefs(all_prev_model,
                                                                       test_X,
                                                                       variance_levels_to_plot,
                                                                       oracle_singular_vecs,
                                                                       logger)
                    else:
                        all_prev_coords = compute_projection_coords_for_model(all_prev_model,
                                                                              oracle_weights,
                                                                              all_prev_proj_dirs,
                                                                              final_proj_dirs)
                    
            
            for layer_idx in range(len(adapted_coords)):
                if model_name.endswith('_all') or model_name.endswith('_final'):
                    model_coords_sequences[layer_idx][model_name] = [adapted_coords[layer_idx]]
                else:
                    model_coords_sequences[layer_idx][model_name] = [all_prev_coords[layer_idx], 
                                                                     adapted_coords[layer_idx]]
        
        elif model_name.startswith('sequential_fine_tune'):
            model_metrics_summary_df \
                = metrics_summary_df.loc[np.logical_and.reduce((metrics_summary_df['Method'] == model_name,
                                                                metrics_summary_df['Shift sequence'] == shift_sequence_with_sample_size,
                                                                metrics_summary_df['Seed'] == seed))]
            best_init_lr  = model_metrics_summary_df['Starting learning rate'].values[0]
            best_lr_decay = model_metrics_summary_df['Learning rate decay factor'].values[0]

            loading_time0 = False
            erm_time0_model  = model_class(model_properties = model_properties,
                                           num_classes = num_classes)
            if model_name == 'sequential_fine_tune_exponential_lr_decay':
                if erm_exp_lr_decay_time0_coords is None:
                    # load ERM on time 0 and compute CCA coefficients
                    erm_time0_filename  = experiment_dir + 'erm_time0_exponential_lr_decay/seed' + str(seed) \
                                        + '/models/erm_time0_exponential_lr_decay_all_best_model.pt.gz'
                    loading_time0 = True
            else:
                if erm_time0_coords is None:
                    erm_time0_filename  = experiment_dir + 'erm_time0/seed' + str(seed) \
                                        + '/models/erm_time0_all_best_model.pt.gz'
                    loading_time0 = True

            if loading_time0:
                erm_time0_model  = load_state_dict_from_gz(erm_time0_model,
                                                           erm_time0_filename,
                                                           remove_uncompressed_file = False)
                
                if representation == 'cca':
                    time0_coords = compute_model_mean_cca_coefs(erm_time0_model,
                                                                test_X,
                                                                variance_levels_to_plot,
                                                                oracle_singular_vecs,
                                                                logger)
                else:
                    time0_coords = compute_projection_coords_for_model(erm_time0_model,
                                                                       oracle_weights,
                                                                       all_prev_proj_dirs,
                                                                       final_proj_dirs)
                if model_name == 'sequential_fine_tune_exponential_lr_decay':
                    erm_exp_lr_decay_time0_coords = time0_coords
                else:
                    erm_time0_coords = time0_coords
            else:
                if model_name == 'sequential_fine_tune_exponential_lr_decay':
                    time0_coords = erm_exp_lr_decay_time0_coords
                else:
                    time0_coords = erm_time0_coords
            
            for layer_idx in range(len(time0_coords)):
                model_coords_sequences[layer_idx][model_name] = [time0_coords[layer_idx]]
            for time_idx in range(num_target_steps):
                adapted_model  = model_class(model_properties = model_properties,
                                             num_classes = num_classes)
                if model_name.endswith('_lr_decay'):
                    adapted_filename  = model_dir + model_name + '_init_lr' + str(best_init_lr) + '_decay' + str(best_lr_decay) \
                                      + 'time' + str(time_idx + 1) + '_all_best_model.pt.gz'
                else:
                    adapted_filename  = model_dir + model_name + '_time' + str(time_idx + 1) + '_all_best_model.pt.gz'
                adapted_model  = load_state_dict_from_gz(adapted_model,
                                                            adapted_filename)
                if representation == 'cca':
                    adapted_coords = compute_model_mean_cca_coefs(adapted_model,
                                                                  test_X,
                                                                  variance_levels_to_plot,
                                                                  oracle_singular_vecs,
                                                                  logger)
                else:
                    adapted_coords = compute_projection_coords_for_model(adapted_model,
                                                                         oracle_weights,
                                                                         all_prev_proj_dirs,
                                                                         final_proj_dirs)
                
                for layer_idx in range(len(adapted_coords)):
                    model_coords_sequences[layer_idx][model_name].append(adapted_coords[layer_idx])
            
        else:
            assert model_name.startswith('joint_model') or model_name.startswith('sequential_side_tune') \
                or model_name.startswith('sequential_low_rank_adapt') or model_name.startswith('erm_side_tune') \
                or model_name.startswith('erm_low_rank_adapt')
            if 'side_tune' in model_name:
                assert representation != 'weight_proj'
            adapter_mode  = None
            adapter_ranks = [0 for _ in range(num_blocks + 2)]
            side_layer_sizes = [[] for _ in range(num_blocks + 2)]
            if 'side_tune' in model_name:
                mode = 'side_tune'
                if 'block' in model_name:
                    side_layer_sizes = [[]] + ['block' for _ in range(num_blocks)] + [[]]
            elif 'low_rank_adapt' in model_name:
                mode = 'low_rank_adapt'
                if 'multiply' in model_name:
                    adapter_mode = 'multiply'
                else:
                    assert 'add' in model_name
                    adapter_mode = 'add'
                model_name_split = model_name.split('_')
                for model_name_piece in model_name_split:
                    if model_name_pieces.startswith('rank'):
                        # Note: May need to specify different ranks for first and last layer if all layers were made efficient
                        adapter_ranks = [0] + [adapter_rank for _ in range(num_blocks)] + [0]
            else:
                mode = 'separate'
            model_name_last_part = model_name.split('_')[-1]
            if 'all' in model_name_last_part or 'conv1' in model_name_last_part or 'layer' in model_name_last_part \
                or 'fc' in model_name_last_part:
                separate_layers_str = model_name_last_part
                separate_layers = [layers.split(',') for layers in model_name_last_part.split(':')]
                model_dir = experiment_dir + '_'.join(model_name.split('_')[:-1]) + '/' \
                          + shift_sequence_with_sample_size + '/seed' + str(seed) + '/models/'
            else:
                separate_layers_str = 'all'
                separate_layers = [oracle_model.layer_names]
            if model_name.startswith('erm_'):
                num_time_steps = 2
            else:
                num_time_steps = num_target_steps + 1
            if len(separate_layers) == 1:
                separate_layers = [separate_layers[0] for _ in range(num_time_steps - 1)]
                if separate_layers_str != 'all':
                    separate_layers_str = ':'.join([separate_layers_str for _ in range(num_time_steps - 1)])
            joint_model  = joint_model_class(num_time_steps   = num_time_steps,
                                             separate_layers  = separate_layers,
                                             model_properties = model_properties,
                                             num_classes      = num_classes,
                                             mode             = mode,
                                             adapter_mode     = adapter_mode,
                                             adapter_ranks    = adapter_ranks,
                                             side_layers      = side_layer_sizes)
            if model_name.startswith('joint_model'):
                model_fileheader = 'joint_model'
            else:
                model_fileheader = model_name
            if model_name.startswith('sequential_'):
                joint_filename = model_dir + model_fileheader + '_time' + str(num_time_steps - 1) + '_' + separate_layers_str \
                               + '_separate_best_model.pt.gz'
            else:
                joint_filename = model_dir + model_fileheader + '_' + separate_layers_str + '_separate_best_model.pt.gz'
            joint_model  = load_state_dict_from_gz(joint_model,
                                                   joint_filename)
            if representation == 'cca':
                joint_coords = compute_joint_model_mean_cca_coefs(joint_model,
                                                                  test_X,
                                                                  variance_levels_to_plot,
                                                                  oracle_singular_vecs,
                                                                  logger)
            else:
                joint_coords = compute_projection_coords_for_joint_model(joint_model,
                                                                         oracle_weights,
                                                                         all_prev_proj_dirs,
                                                                         final_proj_dirs)
            for layer_idx in range(len(joint_coords)):
                model_coords_sequences[layer_idx][model_name] = joint_coords[layer_idx]
            
    with open(coords_json_file_name, 'w') as f:
        json.dump(model_coords_sequences, f)
        
    method_names_dict = {models_to_include[idx]: readable_method_names[idx]
                         for idx in range(len(models_to_include))}
    layer_plot_titles = get_plot_layer_names(model_type,
                                             num_blocks)
    layer_plot_titles.append('entire model')
    empty_model = model_class(model_properties = model_properties,
                              num_classes = num_classes)
    layer_file_names  = empty_model.layer_names
    layer_file_names.append('entire_model')
    if representation == 'cca':
        xlabel = 'Mean CCA coefficients explaining 50% of variance'
        ylabel = 'Mean CCA coefficients explaining 90% of variance'
        oracle_coords = (1, 1)
    else:
        xlabel = 'Projection onto direction of historical shift'
        ylabel = 'Projection onto direction of limited final sample size'
        oracle_coords = (0, 0)
    for layer_idx in range(len(model_coords_sequences)):
        layer_plot_title = plot_title + ' ' + layer_plot_titles[layer_idx]
        layer_file_name  = visualization_dir + representation + '_over_time_visualization_' \
                         + layer_file_names[layer_idx] + '.pdf'
        plot_coord_time_sequences(model_coords_sequences[layer_idx],
                                  method_names_dict,
                                  layer_plot_title,
                                  xlabel,
                                  ylabel,
                                  layer_file_name,
                                  oracle_coords = oracle_coords)
    if num_blocks == 4:
        plt.clf()
        plt.rc('font', 
               family = 'serif', 
               size   = 13)
        plt.rc('xtick', 
               labelsize = 12)
        plt.rc('ytick', 
               labelsize = 12)
        fig, ax = plt.subplots(nrows = 2,
                               ncols = 2,
                               figsize = (6.4 * 2, 4.8 * 2))
        layer_idxs = [0, 2, 5, 6]
        if representation == 'cca':
            xlabel = 'Mean CCA coefs for 50% variance'
            ylabel = 'Mean CCA coefs for 90% variance'
            oracle_coords = (1, 1)
        else:
            xlabel = 'Projection: Historical shift'
            ylabel = 'Projection: Limited final # samples'
            oracle_coords = (0, 0)
        for idx in range(len(layer_idxs)):
            row_idx = idx // 2
            col_idx = idx % 2
            layer_plot_title = plot_title + ' ' + layer_plot_titles[layer_idxs[idx]]
            plot_coord_time_sequences(model_coords_sequences[layer_idxs[idx]],
                                      method_names_dict,
                                      layer_plot_title,
                                      xlabel,
                                      ylabel,
                                      output_filename = None,
                                      oracle_coords = oracle_coords,
                                      ax = ax[row_idx, col_idx])
            if row_idx == 1 and col_idx == 1:
                ax[row_idx, col_idx].legend(loc            = 'upper center',
                                            bbox_to_anchor = (-0.15, -0.17),
                                            ncol           = 4)
            else:
                ax[row_idx, col_idx].get_legend().remove()
        if len(models_to_include) > 15:
            bottom_adjust = .23
        else:
            bottom_adjust = .2
        fig.subplots_adjust(bottom = bottom_adjust,
                            wspace = .25,
                            hspace = .28)
        fig.savefig(visualization_dir + representation + '_over_time_visualization_merged_layers.pdf')

def merge_interpolation_plots(dataset_name,
                              num_blocks,
                              shift_sequence,
                              readable_method_names,
                              plot_title_header,
                              seed):
    '''
    Create a 3-plot figure for the 3 model architectures and the same shift sequence
    Share a single legend on the right
    @param dataset_name: str, cifar10 or cifar100
    @param num_blocks: int, number of blocks
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of corruption, rotation, label_flip, 
                           label_shift, rotation_cond, recoloring, recoloring_cond, subpop
    @param readable_method_names: list of str, names of models for plot legend
    @param plot_title_header: str, start of title for each plot, will append model architecture name
    @param seed: int, seed for data generation
    @return: None
    '''
    if plot_title_header[-1] != ' ':
        plot_title_header += ' '
    if dataset_name == 'portraits':
        num_target_steps = 7
        shift_sequence_with_sample_size = dataset_name
    else:
        shift_sequence_split = shift_sequence.split(':')
        num_target_steps     = len(shift_sequence_split)
        source_sample_size   = 6000
        target_sample_sizes  = [4000, 6000, 4000]
        shifts_with_sample_size = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                   for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)

    num_time_steps = num_target_steps + 1
    model_types    = ['convnet', 'densenet', 'resnet']
    model_titles   = ['Conv-' + str(num_blocks), 'Dense-' + str(num_blocks), 'Res-' + str(num_blocks)]
    
    plt.clf()
    plt.rc('font', 
           family = 'serif', 
           size   = 14)
    plt.rc('xtick', 
           labelsize = 12)
    plt.rc('ytick', 
           labelsize = 12)
    fig, ax = plt.subplots(nrows = 3,
                           ncols = 1,
                           figsize = (10, 13))

    for model_type_idx in range(3):
        experiment_dir    = config.output_dir + model_types[model_type_idx] + '_' + str(num_blocks) + 'blocks_' + dataset_name + '_experiment/'
        visualization_dir = experiment_dir + 'loss_interpolation_plots/' + shift_sequence_with_sample_size \
                          + '/seed' + str(seed) + '/'
        accuracies_to_plot_df = pd.read_csv(visualization_dir + 'interpolated_accuracies.csv')
        accuracies_to_plot_df['Historical'] = np.where(accuracies_to_plot_df['Method'] == 'EWC',
                                                       'Sequential',
                                                       accuracies_to_plot_df['Historical'])

        if model_types[model_type_idx] == 'convnet':
            method_idxs_to_include = []
            for method_idx, method in enumerate(readable_method_names):
                if 'side_tune' in method and 'side_tune_block' not in method:
                    continue
                method_idxs_to_include.append(method_idx)
                               
        sns.lineplot(data        = accuracies_to_plot_df,
                     x           = 'Linear Interpolation',
                     y           = 'Accuracy',
                     hue         = 'Method',
                     style       = 'Historical',
                     hue_order   = readable_method_names,
                     style_order = ['ERM start', 'IRM/DRO start', 'Sequential', 'Joint'],
                     dashes      = {'ERM start'    : '',
                                    'IRM/DRO start': (5, 5),
                                    'Sequential'   : (3, 3),
                                    'Joint'        : (1, 1)},
                     palette     = 'colorblind',
                     ax          = ax[model_type_idx])
        ax[model_type_idx].set_xlim([0, 1])
        ax[model_type_idx].set_ylim([0, min(1, accuracies_to_plot_df['Accuracy'].max() + .05)])
        ax[model_type_idx].set_title(plot_title_header + model_titles[model_type_idx])
        if model_type_idx == 1:
            handles, labels = ax[model_type_idx].get_legend_handles_labels()
            if model_types[model_type_idx] == 'convnet':
                handles_to_include = [handles[idx + 1] for idx in method_idxs_to_include]
                labels_to_include  = [labels[idx + 1]  for idx in method_idxs_to_include]
            else:
                style_index = labels.index('Historical')
                handles_to_include = handles[1:style_index]
                labels_to_include  = labels[1:style_index]
            for handle, label in zip(handles_to_include, labels_to_include):
                if label.startswith('ERM'):
                    handle.set_linestyle((0, ()))
                elif label.startswith('IRM') or label.startswith('DRO'):
                    handle.set_linestyle((0, (5, 5)))
                elif label.startswith('SFT') or label.startswith('sequential') or label == 'EWC':
                    handle.set_linestyle((0, (3, 3)))
                else:
                    handle.set_linestyle((0, (1, 1)))
            ax[model_type_idx].legend(handles        = handles_to_include,
                                      labels         = labels_to_include,
                                      loc            = 'center left',
                                      bbox_to_anchor = (1.05, .5),
                                      ncol           = 1)
        else:
            ax[model_type_idx].get_legend().remove()

    output_dir  = config.output_dir + 'merged_loss_interpolation_plots/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + shift_sequence_with_sample_size + '_all_architectures_interpolated_accuracies.pdf'
    fig.subplots_adjust(right  = .55,
                        hspace = .35)
    fig.savefig(output_file)
    
def create_parser():
    '''
    Create an argument parser
    @return: argument parser
    '''
    parser = argparse.ArgumentParser(description = 'Visualize parameters of different types of models over time steps.')
    parser.add_argument('--dataset',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify cifar10, cifar100, or portraits.')
    parser.add_argument('--model_type',
                        action  = 'store',
                        type    = str,
                        help    = ('Specify whether to use resnet, convnet, or densenet architecture. '
                                   'May also be merge to show all 3 in 1 figure for loss_interpolate mode.'))
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
    parser.add_argument('--representation',
                        action  = 'store',
                        type    = str,
                        help    = ('Specify cca to define coordinates as mean of CCA coefficients '
                                   'explaining 50% and 90% of variance, '
                                   'weight_proj to define coordinates as projections onto directions '
                                   'reflecting limitations of historical shift and limited data from the final time step, '
                                   'or loss_interpolate to plot loss as weights are linearly interpolated from T - 1 to T.')) 
    parser.add_argument('--gpu_num',
                        action  = 'store',
                        type    = int,
                        default = 1,
                        help    = 'Specify which GPUs to use.')
    parser.add_argument('--seed',
                        action  = 'store',
                        type    = int,
                        default = 1007,
                        help    = 'Specify a random seed for data generation.')
    parser.add_argument('--model_dirs',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify colon-separated list of directory names of models to plot in addition to oracle.')
    parser.add_argument('--model_names',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify colon-separated list of names for plot legend matching models argument.')
    parser.add_argument('--plot_title',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify plot title.')
    parser.add_argument('--merge_plots',
                        action  = 'store_true',
                        help    = 'Specify to create a shared legend for all 3 architectures in a single set-up.')
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    assert args.representation in {'cca', 'weight_proj', 'loss_interpolate'}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    torch.cuda.device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    assert(torch.cuda.is_available())
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    
    method_names = args.model_dirs.split(':')
    readable_method_names = args.model_names.split(':')

    if args.dataset == 'portraits':
        # other methods require oracle, which real-world datasets do not have
        assert args.representation == 'loss_interpolate'
    else:
        assert len(args.shift_sequence) > 0

    if args.model_type == 'merge':
        assert args.representation == 'loss_interpolate'
        merge_interpolation_plots(args.dataset,
                                  args.num_blocks,
                                  args.shift_sequence,
                                  readable_method_names,
                                  args.plot_title,
                                  args.seed)
    elif args.representation == 'loss_interpolate':
        visualize_interpolation_accuracies(args.dataset,
                                           args.model_type,
                                           args.num_blocks,
                                           args.shift_sequence,
                                           method_names,
                                           readable_method_names,
                                           args.plot_title,
                                           args.seed)
    else:
        visualize_parameters_over_time(args.dataset,
                                       args.model_type,
                                       args.num_blocks,
                                       args.shift_sequence,
                                       args.representation,
                                       method_names,
                                       readable_method_names,
                                       args.plot_title,
                                       args.seed)