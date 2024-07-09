import os
import sys
import time
import math
import json
import shutil
import argparse

from itertools import product
from functools import partial
from os.path import dirname, abspath, join
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from munkres import Munkres

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from model_property_helper_functions import (
    define_layer_combos,
    get_layer_names,
    get_plot_layer_names,
    get_joint_model_class_and_properties,
    compute_max_fc_adapter_rank
)

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from pytorch_model_utils import save_state_dict_to_gz, load_state_dict_from_gz

sys.path.append(join(dirname(dirname(abspath(__file__))), 'data_loaders'))
from load_image_data_with_shifts import load_datasets_over_time
from load_yearbook_data import load_yearbook_data

sys.path.append(dirname(dirname(abspath(__file__))))
import config

'''
Supported variants of joint model:
1. Separate layers at different steps with L2 regularization on difference between weights at adjacent steps (with support for different L2 regularization weighting schemes)
2. Side modules: Add a side module at each step. Side module size can range from 1 convolution layer + 1 batch normalization layer to original-sized block.
3. Low-rank adapters: Multiply weights by low-dim parameters at each time step (beta mode)
Can also specify a different weight for the loss at each time step in the total training objective
'''

def compute_joint_model_loss_for_a_time_step(joint_model,
                                             loader,
                                             time_step):
    '''
    Compute cross entropy loss for a time step
    @param joint_model: joint_block_model
    @param loader: torch DataLoader
    @param time_step: int, time step to evaluate
    @return: float, loss
    '''
    assert time_step < joint_model.num_time_steps
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss  = 0.
    num_samples = 0
    joint_model.eval()
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if torch.cuda.is_available():
            batch_x  = batch_x.cuda()
            batch_y  = batch_y.cuda()
        with torch.no_grad():
            outputs  = joint_model(batch_x, time_step)
            loss     = loss_fn(outputs, batch_y)
        if torch.cuda.is_available():
            loss     = float(loss.detach().cpu().numpy())
        else:
            loss     = float(loss.detach().numpy())
        batch_size   = len(batch_y)
        total_loss  += loss * batch_size
        num_samples += batch_size
    joint_model.train()
    return total_loss/num_samples

def compute_joint_model_accuracy_for_a_time_step(joint_model,
                                                 loader,
                                                 time_step):
    '''
    Compute 0-1 accuracy for a time step
    @param joint_model: joint_block_model
    @param loader: torch DataLoader
    @param time_step: int, time step to evaluate
    @return: 1. float, accuracy
             2. int, number of samples
    '''
    assert time_step < joint_model.num_time_steps
    num_correct = 0
    num_samples = 0
    joint_model.eval()
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        if torch.cuda.is_available():
            batch_x  = batch_x.cuda()
        with torch.no_grad():
            outputs  = torch.argmax(joint_model(batch_x, time_step), dim = 1)
        if torch.cuda.is_available():
            outputs  = outputs.detach().cpu().numpy()
        else:
            outputs  = outputs.detach().numpy()
        batch_y      = batch_y.detach().numpy()
        num_correct += np.sum(np.where(outputs.astype(int) == batch_y.astype(int), 1, 0))
        batch_size   = len(batch_y)
        num_samples += batch_size
    joint_model.train()
    return num_correct/float(num_samples), num_samples

def plot_joint_model_losses_over_epochs(train_total_losses,
                                        valid_total_losses,
                                        train_final_losses,
                                        valid_final_losses,
                                        plot_filename,
                                        logger,
                                        plot_title = None,
                                        accuracy   = False):
    '''
    Plot training and validation losses over epochs, total and final time step
    @param train_total_losses: list of floats, training losses over epochs, weighted average across time steps
    @param valid_total_losses: list of floats, validation losses over epochs, weighted average across time steps
    @param train_final_losses: list of floats, training losses over epochs, final time step only
    @param valid_final_losses: list of floats, validation losses over epochs, final time step only
    @param plot_filename: str, path to save plot
    @param logger: logger, for INFO messages
    @param plot_title: str, plot title if provided
    @param accuracy: bool, if True, then assumes function is plotting accuracies instead, total is unweighted
    @return: None
    '''
    start_time = time.time()
    plt.clf()
    plt.rc('font', 
           size = 14)
    fig, ax = plt.subplots(figsize = (6.4, 4.8))
    ax.plot(train_total_losses,
            label = 'Train total')
    ax.plot(valid_total_losses,
            label = 'Valid total')
    ax.plot(train_final_losses,
            label = 'Train final time')
    ax.plot(valid_final_losses,
            label = 'Valid final time')
    ax.legend()
    ax.set_xlabel('Epoch')
    if accuracy:
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0,1])
    else:
        ax.set_ylabel('Loss')
        ax.set_ylim(bottom = 0)
    if plot_title is not None:
        ax.set_title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)
    if accuracy:
        logger.info('Plotted accuracies over epochs to ' + plot_filename + ' in ' + str(time.time() - start_time) + ' seconds')
    else:
        logger.info('Plotted losses over epochs to ' + plot_filename + ' in ' + str(time.time() - start_time) + ' seconds')
        
def compute_joint_model_losses_for_all_time_steps(joint_model,
                                                  data_loaders_over_time,
                                                  losses,
                                                  time_step_weights,
                                                  logger):
    '''
    Compute losses and accuracies for all time steps and weighted total across time steps
    @param joint_model: joint_block_model, model to compute losses for
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step
    @param losses: dict mapping str to list of floats, loss quantity to list of losses over epochs, 
                   will be modified and returned
    @param time_step_weights: np array of floats, weights on losses at each time step in total loss
    @param logger: logger, for INFO messages
    @return: losses dict updated
    '''
    assert len(data_loaders_over_time) == len(time_step_weights)
    epoch = len(losses['train_total_losses'])
    num_time_steps = joint_model.num_time_steps
    total_train_loss = 0.
    total_valid_loss = 0.
    total_train_num_correct = 0.
    total_valid_num_correct = 0.
    total_train_num_samples = 0.
    total_valid_num_samples = 0.
    for t in range(num_time_steps):
        if t == num_time_steps - 1:
            assert data_loaders_over_time[t] is not None
        elif data_loaders_over_time[t] is None:
            continue
        t_train_loss = compute_joint_model_loss_for_a_time_step(joint_model,
                                                                data_loaders_over_time[t]['train'],
                                                                t)
        total_train_loss += time_step_weights[t] * t_train_loss / num_time_steps
        losses['train_step' + str(t) + '_losses'].append(t_train_loss)

        t_valid_loss = compute_joint_model_loss_for_a_time_step(joint_model,
                                                                data_loaders_over_time[t]['valid'],
                                                                t)
        total_valid_loss += time_step_weights[t] * t_valid_loss / num_time_steps
        losses['valid_step' + str(t) + '_losses'].append(t_valid_loss)

        t_train_acc, t_train_num_samples = compute_joint_model_accuracy_for_a_time_step(joint_model,
                                                                                        data_loaders_over_time[t]['train'],
                                                                                        t)
        total_train_num_correct += t_train_acc * t_train_num_samples
        total_train_num_samples += t_train_num_samples
        losses['train_step' + str(t) + '_accuracies'].append(t_train_acc)

        t_valid_acc, t_valid_num_samples = compute_joint_model_accuracy_for_a_time_step(joint_model,
                                                                                        data_loaders_over_time[t]['valid'],
                                                                                        t)
        total_valid_num_correct += t_valid_acc * t_valid_num_samples
        total_valid_num_samples += t_valid_num_samples
        losses['valid_step' + str(t) + '_accuracies'].append(t_valid_acc)

        logger.info('Epoch ' + str(epoch) + ' achieved training loss ' + str(t_train_loss) 
                    + ', training accuracy ' + str(t_train_acc)
                    + ', validation loss ' + str(t_valid_loss) 
                    + ', validation accuracy ' + str(t_valid_acc) 
                    + ' at time step ' + str(t))

    total_train_acc = total_train_num_correct/float(total_train_num_samples)
    total_valid_acc = total_valid_num_correct/float(total_valid_num_samples)
    losses['train_total_losses'].append(total_train_loss)
    losses['train_total_accuracies'].append(total_train_acc)
    losses['valid_total_losses'].append(total_valid_loss)
    losses['valid_total_accuracies'].append(total_valid_acc)
    logger.info('Epoch ' + str(epoch) + ' achieved total training loss ' + str(total_train_loss)
                + ', total training accuracy ' + str(total_train_acc)
                + ', total validation loss ' + str(total_valid_loss)
                + ', total validation accuracy ' + str(total_valid_acc))
    return losses
        
def fit_joint_model(model_type,
                    num_blocks,
                    separate_layers,
                    data_loaders_over_time,
                    num_classes,
                    learning_rate,
                    regularization_params,
                    time_step_loss_weights,
                    n_epochs,
                    fileheader,
                    logger,
                    model_name                        = None,
                    save_model                        = True,
                    resume_from_n_epochs              = 0,
                    early_stopping_n_epochs           = float('inf'),
                    start_with_imagenet               = False,
                    mode                              = 'separate',
                    adapter_ranks                     = [0, 0, 0, 0, 0, 0],
                    adapter_mode                      = None,
                    side_layers                       = ['separate', [],[],[],[], 'separate'],
                    partial_model_file_name           = None,
                    use_partial_to_init_final         = False,
                    remove_uncompressed_partial_model = True,
                    ablate                            = False):
    '''
    Fit a joint model on data from multiple time points
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param separate_layers: list of list of str, specify which layers will be new modules at each time step,
                            outer list over time steps (first entry is 2nd time step 
                            since all modules are new at 1st time step), second list), 
                            outer list has length num_time_steps - 1
                            inner list over layers: subset of conv1, layer1, layer2, layer3, layer4, fc,
                            example: [[conv1, layer1], [fc]] means 3 time step model has 
                            - one set of conv1, layer1 at time 0 and another set shared at times 1 and 2
                            - one fc module at times 0 and 1, another fc module at time 2
                            - all other modules shared across all 3 time steps
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step, None if no data at a time step for ablate
    @param num_classes: int, number of output classes
    @param learning_rate: float, learning rate
    @param regularization_params: dict mapping str to bool, str, or float, contains
                                  - weight_decay_type: str, l1 or l2 for regularization parameter norm
                                  - weight_decay: float, regularization constant for parameter norm
                                  - adjacent_reg_type: str, l1, l2, or l2_fisher for difference between weights
                                                       at consecutive steps (l2_fisher weights L2 regularization 
                                                       between adjacent parameters by Fisher info of parameters 
                                                       at previous time step in previous epoch)
                                  - adjacent_reg: float, constant for regularizing adjacent differences
                                  - weight_reg_by_time: bool, if True weight reg on param norm by 1/(t+1) 
                                                        and adjacent diff reg by 1/(T-t)
                                  - dropout: float, probability each output entry after each layer/block is zero'd 
                                             during training
                                  - parameter_efficient_weight_decay: float, regularization constant for side modules 
                                                                      and low-rank adapters, 
                                                                      same as weight_decay if not specified
    @param time_step_loss_weights: np array of floats, weight for losses at each time step
    @param n_epochs: int, number of epochs to fit model (total including epochs prior to resuming)
    @param fileheader: str, start of path to save model
    @param logger: logger, for INFO messages
    @param model_name: str, name of model for plotting loss and accuracy if provided
    @param save_model: bool, whether to save model to file
    @param resume_from_n_epochs: int, number of epochs to resume fitting from, 0 to start from pre-trained weights
    @param early_stopping_n_epochs: int or inf, number of epochs before stopping if validation accuracy no longer improves
    @param start_with_imagenet: bool, if True initialize all modules with ImageNet pretrained weights
    @param mode: str, separate for separate modules at each time step,
                 side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                 low_rank_adapt to use low-rank adapters for residual blocks at each time step
    @param adapter_ranks: list of int, rank of adapters for multiplying or adding to fully connected and convolutional weights
                          in each layer
    @param adapter_mode: str, multiply or add
    @param side_layers: list of str or lists of int, number of output channels 
                        for each of the intermediate convolution layers in side modules,
                        str option is "block" for side module to match original block
                        or "separate" for non-block layers
    @param partial_model_file_name: str, path to load modules up to T - 1, 
                                    these modules will be frozen and only final time step is fit
    @param use_partial_to_init_final: bool, whether to use module at time T - 1 to initialize module at T
    @param remove_uncompressed_partial_model: bool, whether to remove uncompressed .pt file when loading model, 
                                              usually can set to True 
                                              unless expecting multiple threads to load the same model at the same time
    @param ablate: bool, set side modules to 0 or low-rank adapters to identity function
    @return: 1. joint_block_model, fitted model from best epoch
             2. float, validation accuracy of fitted model at final time step at best epoch
             3. joint_block_model, fitted model from last epoch
             4. torch Adam optimizer, from last epoch
             5. int, epoch when fitting was stopped, may be less than n_epochs if early stopping applied
    '''
    assert model_type in {'resnet', 'densenet', 'convnet'}
    assert num_blocks >= 1
    assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
    if mode != 'separate':
        assert regularization_params['adjacent_reg_type'] != 'l2_fisher'
        assert not regularization_params['weight_reg_by_time']
    if ablate:
        assert mode != 'separate'
        assert partial_model_file_name is None
    assert {'weight_decay_type', 'weight_decay', 'adjacent_reg_type', 
            'adjacent_reg', 'weight_reg_by_time', 'dropout'}.issubset(set(regularization_params.keys()))
    assert regularization_params['weight_decay_type'] in {'l1', 'l2'}
    assert regularization_params['weight_decay'] >= 0
    assert regularization_params['adjacent_reg_type'] in {'l1', 'l2', 'l2_fisher'}
    assert regularization_params['adjacent_reg'] >= 0
    assert regularization_params['dropout'] >= 0 and regularization_params['dropout'] < 1
    if 'parameter_efficient_weight_decay' in regularization_params:
        regularization_params['parameter_efficient_decay_ratio'] \
            = regularization_params['parameter_efficient_weight_decay']/regularization_params['weight_decay']
    else:
        regularization_params['parameter_efficient_decay_ratio'] = 1
    
    assert len(data_loaders_over_time) == len(time_step_loss_weights)
    assert fileheader[-1] == '_'
    best_model_filename     = fileheader + 'best_model.pt.gz'
    best_model_exists       = os.path.exists(best_model_filename)
    last_model_filename     = fileheader + str(n_epochs) + 'epochs_last_model.pt.gz'
    last_model_exists       = os.path.exists(last_model_filename)
    last_optimizer_filename = fileheader + str(n_epochs) + 'epochs_last_adam_optimizer.pt.gz'
    last_optimizer_exists   = os.path.exists(last_optimizer_filename)
    losses_filename         = fileheader + str(n_epochs) + 'epochs_losses.json'
    losses_exist            = os.path.exists(losses_filename)
    loaded_from_files       = False
    
    num_time_steps = len(data_loaders_over_time)
    joint_model_class, model_class, model_properties = get_joint_model_class_and_properties(model_type,
                                                                                            num_blocks,
                                                                                            logger)
    joint_model = joint_model_class(num_time_steps,
                                    separate_layers,
                                    model_properties = model_properties,
                                    num_classes      = num_classes,
                                    mode             = mode,
                                    adapter_ranks    = adapter_ranks,
                                    adapter_mode     = adapter_mode,
                                    side_layers      = side_layers)
    best_joint_model = joint_model_class(num_time_steps,
                                         separate_layers,
                                         model_properties = model_properties,
                                         num_classes      = num_classes,
                                         mode             = mode,
                                         adapter_ranks    = adapter_ranks,
                                         adapter_mode     = adapter_mode,
                                         side_layers      = side_layers)
    adam_optimizer = torch.optim.Adam(joint_model.parameters(),
                                      lr           = learning_rate,
                                      weight_decay = 0.)
    if torch.cuda.is_available():
        joint_model      = joint_model.cuda()
        best_joint_model = best_joint_model.cuda()
        
    if best_model_exists and last_model_exists and last_optimizer_exists and losses_exist:
        loaded_from_files = True
        best_joint_model = load_state_dict_from_gz(best_joint_model,
                                                   best_model_filename)
        logger.info('Loaded best model from ' + best_model_filename)
        
        joint_model = load_state_dict_from_gz(joint_model,
                                              last_model_filename)
        logger.info('Loaded last model from ' + last_model_filename)
        
        adam_optimizer = load_state_dict_from_gz(adam_optimizer,
                                                 last_optimizer_filename)
        logger.info('Loaded Adam optimizer from ' + last_optimizer_filename)
        
        with open(losses_filename, 'r') as f:
            losses = json.load(f)
        logger.info('Loaded losses and accuracies from ' + losses_filename)
    else:
        overall_start_time = time.time()
        
        assert resume_from_n_epochs <= n_epochs
        if resume_from_n_epochs > 0:
            resume_last_model_filename     = fileheader + str(resume_from_n_epochs) + 'epochs_last_model.pt.gz'
            resume_last_optimizer_filename = fileheader + str(resume_from_n_epochs) + 'epochs_last_adam_optimizer.pt.gz'
            resume_losses_filename         = fileheader + str(resume_from_n_epochs) + 'epochs_losses.json'
            assert os.path.exists(resume_last_model_filename)
            assert os.path.exists(resume_last_optimizer_filename)
            assert os.path.exists(resume_losses_filename)
            
            joint_model = load_state_dict_from_gz(joint_model,
                                                  resume_last_model_filename)
            logger.info('Loaded model to resume from ' + resume_last_model_filename)
            
            adam_optimizer = load_state_dict_from_gz(adam_optimizer,
                                                     resume_last_optimizer_filename)
            logger.info('Loaded Adam optimizer to resume from ' + resume_last_optimizer_filename)
            
            with open(resume_losses_filename, 'r') as f:
                losses     = json.load(f)
            for loss_quantity in losses:
                assert len(losses[loss_quantity]) == resume_from_n_epochs + 1
            best_valid_acc = max(losses['valid_step' + str(num_time_steps - 1) + '_accuracies'])
            found_better   = False
        else:
            losses         = defaultdict(list)
            if partial_model_file_name is not None:
                assert num_time_steps > 1
                if num_time_steps == 2:
                    previous_model = model_class(model_properties = model_properties,
                                                 num_classes      = num_classes)
                else:
                    previous_model = joint_model_class(num_time_steps - 1,
                                                       separate_layers[:-1],
                                                       model_properties = model_properties,
                                                       num_classes      = num_classes,
                                                       mode             = mode,
                                                       adapter_ranks    = adapter_ranks,
                                                       adapter_mode     = adapter_mode,
                                                       side_layers      = side_layers)
                previous_model = load_state_dict_from_gz(previous_model,
                                                         partial_model_file_name,
                                                         remove_uncompressed_partial_model)
                if torch.cuda.is_available():
                    previous_model = previous_model.cuda()
                joint_model.load_partial_state_dict(pretrained_model          = previous_model,
                                                    use_partial_to_init_final = use_partial_to_init_final)
                joint_model.freeze_params_up_to_time(num_time_steps - 2)
            elif start_with_imagenet:
                joint_model.load_partial_state_dict(all_modules = True)
            if ablate:
                if mode == 'side_tune':
                    joint_model.ablate_side_modules()
                else:
                    assert mode == 'low_rank_adapt'
                    joint_model.ablate_adapters()
            losses         = compute_joint_model_losses_for_all_time_steps(joint_model,
                                                                           data_loaders_over_time,
                                                                           losses,
                                                                           time_step_loss_weights,
                                                                           logger)
            best_valid_acc = losses['valid_step' + str(num_time_steps - 1) + '_accuracies'][0]
            
        best_epoch  = 0
        best_params = deepcopy(joint_model.state_dict())
        loss_fn     = torch.nn.CrossEntropyLoss()

        data_generators         = [None if data_loaders_over_time[t] is None
                                   else iter(data_loaders_over_time[t]['train'])
                                   for t in range(num_time_steps)]
        n_epochs_stop           = n_epochs
        n_epochs_no_improvement = 0
        prev_epoch_fisher_infos = {joint_model.layer_names[layer_idx]: 
                                   [defaultdict(lambda: 1)
                                    for t in range(len(joint_model.layers[layer_idx]) - 1)]
                                   for layer_idx in range(len(joint_model.layer_names))}
        max_batches_time_step   = np.argmax(np.array([0 if data_loaders_over_time[t] is None
                                                      else len(data_loaders_over_time[t]['train']) 
                                                      for t in range(num_time_steps)]))
        for epoch in range(resume_from_n_epochs, n_epochs):
            start_time = time.time()
            if regularization_params['adjacent_reg_type'] == 'l2_fisher':
                this_epoch_fisher_infos = {joint_model.layer_names[layer_idx]: 
                                           [defaultdict(lambda: 0)
                                            for t in range(len(joint_model.layers[layer_idx]) - 1)]
                                           for layer_idx in range(len(joint_model.layer_names))}
                total_num_samples = 0
            for max_time_batch_idx, (max_time_batch_x, max_time_batch_y) \
                in enumerate(data_loaders_over_time[max_batches_time_step]['train']):
                # compute total loss across time steps
                loss_train      = 0
                for time_idx in range(joint_model.num_time_steps):
                    if time_idx == max_batches_time_step:
                        time_batch_x = max_time_batch_x
                        time_batch_y = max_time_batch_y
                    else:
                        if data_generators[time_idx] is None:
                            continue
                        try:
                            time_batch_x, time_batch_y = next(data_generators[time_idx])
                        except StopIteration:
                            # restart data loader for this time step if reach end
                            data_generators[time_idx]  = iter(data_loaders_over_time[time_idx]['train'])
                            time_batch_x, time_batch_y = next(data_generators[time_idx])

                    if torch.cuda.is_available():
                        time_batch_x = time_batch_x.cuda()
                        time_batch_y = time_batch_y.cuda()

                    t_outputs   = joint_model(time_batch_x, time_idx, regularization_params['dropout'])
                    t_loss      = loss_fn(t_outputs, time_batch_y) * time_step_loss_weights[time_idx] / num_time_steps
                    loss_train += t_loss
                
                loss_train \
                    += (regularization_params['weight_decay'] 
                        * joint_model.compute_param_norm(regularization_params['weight_decay_type'],
                                                         regularization_params['weight_reg_by_time'],
                                                         regularization_params['parameter_efficient_decay_ratio']))
                loss_train \
                    += (regularization_params['adjacent_reg']
                        * joint_model.compute_adjacent_param_norm(regularization_params['adjacent_reg_type'][:2],
                                                                  regularization_params['weight_reg_by_time'],
                                                                  prev_epoch_fisher_infos))

                if torch.cuda.is_available():
                    loss_train = loss_train.cuda()
                loss_train.backward(retain_graph = True)
                
                if regularization_params['adjacent_reg_type'] == 'l2_fisher':
                    this_batch_fisher_infos = joint_model.compute_fisher_info()
                    for layer_name in this_batch_fisher_infos:
                        for time_step in range(len(this_batch_fisher_infos[layer_name])):
                            for param in this_batch_fisher_infos[layer_name][time_step]:
                                this_epoch_fisher_infos[layer_name][time_step][param] \
                                    += this_batch_fisher_infos[layer_name][time_step][param]
                    # Note for future: Consider normalizing by # of samples that go through each module instead of total
                    total_num_samples += sum([len(batch_y) for batch_y in batch_ys])

                adam_optimizer.step()
                adam_optimizer.zero_grad()
    
            losses = compute_joint_model_losses_for_all_time_steps(joint_model,
                                                                   data_loaders_over_time,
                                                                   losses,
                                                                   time_step_loss_weights,
                                                                   logger)
                
            t_valid_acc = losses['valid_step' + str(num_time_steps - 1) + '_accuracies'][-1]
            if t_valid_acc > best_valid_acc:
                best_valid_acc   = t_valid_acc
                best_params      = deepcopy(joint_model.state_dict())
                best_epoch       = epoch + 1
                if resume_from_n_epochs > 0:
                    found_better = True
            
            logger.info('Epoch ' + str(epoch + 1) + ' finished in ' + str(time.time() - start_time) + ' seconds')
                    
            if losses['valid_step' + str(num_time_steps - 1) + '_accuracies'][-1] \
                <= losses['valid_step' + str(num_time_steps - 1) + '_accuracies'][-2]:
                n_epochs_no_improvement += 1
            else:
                n_epochs_no_improvement = 0
            
            if n_epochs_no_improvement > early_stopping_n_epochs:
                logger.info('Early stopping at epoch ' + str(epoch + 1))
                last_model_filename     = fileheader + str(epoch + 1) + 'epochs_last_model.pt.gz'
                last_optimizer_filename = fileheader + str(epoch + 1) + 'epochs_last_adam_optimizer.pt.gz'
                losses_filename         = fileheader + str(epoch + 1) + 'epochs_losses.json'
                n_epochs_stop           = epoch + 1
                break
            
            if regularization_params['adjacent_reg_type'] == 'l2_fisher':
                for layer_name in this_epoch_fisher_infos:
                    for time_step in range(len(this_epoch_fisher_infos[layer_name])):
                        for param in this_epoch_fisher_infos[layer_name][time_step]:
                            this_epoch_fisher_infos[layer_name][time_step][param] /= float(total_num_samples)
                for name in this_epoch_fisher_infos:
                    logger.info('Average Fisher info for ' + name + ' in epoch ' + str(epoch) + ': ' 
                                + str(torch.mean(this_epoch_fisher_infos[name])))
                prev_epoch_fisher_infos = this_epoch_fisher_infos
            
        if resume_from_n_epochs > 0:
            if found_better:
                logger.info('Best final time step val acc at epoch ' + str(best_epoch))
            else:
                logger.info('Resuming fitting did not improve final time step val acc')
        else:
            logger.info('Best final time step val acc at epoch ' + str(best_epoch))
        
        best_joint_model.load_state_dict(best_params)
        
        logger.info('Fitted model in ' + str(time.time() - overall_start_time) + ' seconds')
        
        if save_model:
            save_state_dict_to_gz(best_joint_model,
                                  best_model_filename)
            logger.info('Saved best model to ' + best_model_filename)
            
            save_state_dict_to_gz(joint_model,
                                  last_model_filename)
            logger.info('Saved last model to ' + last_model_filename)
            
            save_state_dict_to_gz(adam_optimizer,
                                  last_optimizer_filename)
            logger.info('Saved last Adam optimizer to ' + last_optimizer_filename)
        
        with open(losses_filename, 'w') as f:
            json.dump(losses, f)
        logger.info('Saved losses and accuracies to ' + losses_filename)
    
    losses_plot_filename = fileheader + 'losses.pdf'
    acc_plot_filename    = fileheader + 'accuracies.pdf'
    
    if not os.path.exists(losses_plot_filename):
        plot_joint_model_losses_over_epochs(losses['train_total_losses'],
                                            losses['valid_total_losses'],
                                            losses['train_step' + str(num_time_steps - 1) + '_losses'],
                                            losses['valid_step' + str(num_time_steps - 1) + '_losses'],
                                            losses_plot_filename,
                                            logger,
                                            plot_title = model_name)
    
    if not os.path.exists(acc_plot_filename):
        plot_joint_model_losses_over_epochs(losses['train_total_accuracies'],
                                            losses['valid_total_accuracies'],
                                            losses['train_step' + str(num_time_steps - 1) + '_accuracies'],
                                            losses['valid_step' + str(num_time_steps - 1) + '_accuracies'],
                                            acc_plot_filename,
                                            logger,
                                            plot_title = model_name,
                                            accuracy   = True)
    
    best_valid_acc = max(losses['valid_step' + str(num_time_steps - 1) + '_accuracies'])
    return best_joint_model, best_valid_acc, joint_model, adam_optimizer, n_epochs_stop
    
def tune_joint_model_hyperparameters_for_a_single_combo(model_type,
                                                        num_blocks,
                                                        separate_layers,
                                                        data_loaders_over_time,
                                                        num_classes,
                                                        learning_rates,
                                                        regularization_params,
                                                        last_time_step_loss_wts,
                                                        n_epochs,
                                                        fileheader,
                                                        logger,
                                                        allow_load_best_from_disk         = True,
                                                        loss_weight_increasing            = False,
                                                        start_with_imagenet               = False,
                                                        mode                              = 'separate',
                                                        adapter_ranks                     = [0, 0, 0, 0, 0, 0],
                                                        adapter_mode                      = None,
                                                        side_layers                       = ['separate',[],[],[],[],'separate'],
                                                        partial_model_file_name           = None,
                                                        use_partial_to_init_final         = False,
                                                        remove_uncompressed_partial_model = True,
                                                        ablate                            = False,
                                                        sweep_efficient_reg               = False):
    '''
    Fit joint model with hyperparameters that result in best validation accuracy at final time step
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param separate_layers: list of list of str, specify which layers will be new modules at each time step,
                            outer list over time steps (first entry is 2nd time step 
                            since all modules are new at 1st time step), second list), 
                            outer list has length num_time_steps - 1
                            inner list over layers: subset of conv1, layer1, layer2, layer3, layer4, fc,
                            example: [[conv1, layer1], [fc]] means 3 time step model has 
                            - one set of conv1, layer1 at time 0 and another set shared at times 1 and 2
                            - one fc module at times 0 and 1, another fc module at time 2
                            - all other modules shared across all 3 time steps
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step
    @param num_classes: int, number of output classes
    @param learning_rates: list of floats, learning rates to try
    @param regularization_params: dict mapping str to bool, str, or list of float, contains
                                  - weight_decay_type: str, l1 or l2 for regularization parameter norm
                                  - weight_decays: list of float, regularization constant for parameter norm
                                  - adjacent_reg_type: str, l1, l2, or l2_fisher for difference between weights
                                                       at consecutive steps (l2_fisher weights L2 regularization 
                                                       between adjacent parameters by Fisher info of parameters 
                                                       at previous time step in previous epoch)
                                  - adjacent_regs: list of float, constant for regularizing adjacent differences
                                  - weight_reg_by_time: bool, if True weight reg on param norm by 1/(t+1) 
                                                        and adjacent diff reg by 1/(T-t)
                                  - dropouts: list of float, probability each output entry after each layer/block is zero'd 
                                              during training
    @param last_time_step_loss_wts: list of float, try giving these weights to final time step 
                                    when averaging losses over time steps, 
                                    expected to be [1] if not using loss_weight_increasing
    @param n_epochs: int, number of epochs to fit model
    @param fileheader: str, start of path to save model, ends in _
    @param logger: logger, for INFO messages
    @param allow_load_best_from_disk: bool, whether best model can be loaded from disk, 
                                      otherwise will overwrite best model on disk,
                                      set to False if expanding set of hyperparameters
    @param loss_weight_increasing: bool, specify to weight loss at t by 1/(T - t + 1) instead of alpha at T and 1 elsewhere
    @param start_with_imagenet: bool, specify to initialize all time steps with ImageNet
    @param mode: str, separate for separate modules at each time step,
                 side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                 low_rank_adapt to use low-rank adapters for residual blocks at each time step
    @param adapter_ranks: list of int, rank of adapters for multiplying or adding to fully connected and convolutional weights
                          for each layer
    @param adapter_mode: str, multiply or add
    @param side_layers: list of str or lists of int, number of output channels 
                        for each of the intermediate convolution layers in side modules,
                        str option is "block" for side module to match original block or "separate" for non-blocks
    @param partial_model_file_name: str, path to load modules up to T - 1, 
                                    these modules will be frozen and only final time step is fit
    @param use_partial_to_init_final: bool, whether to use module at time T - 1 to initialize module at T
    @param remove_uncompressed_partial_model: bool, whether to remove uncompressed .pt file when loading model, 
                                              usually can set to True 
                                              unless expecting multiple threads to load the same model at the same time
    @param ablate: bool, set side modules to 0 or low-rank adapters to identity function
    @param sweep_efficient_reg: bool, whether to plot validation performance against different regularization constants 
                                for the side modules or low-rank adapters. In constrast, default is same regularization 
                                constant on parameter-efficient modules as t = 0 modules.
    @return: 1. joint_block_model, fitted with best hyperparameters
             2. float, validation accuracy at final time step
             3. dict mapping str to float, hyperparameter name to best value
    '''
    assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
    if ablate:
        assert mode != 'separate'
    assert {'weight_decay_type', 'weight_decays', 'adjacent_reg_type', 
            'adjacent_regs', 'weight_reg_by_time', 'dropouts'}.issubset(set(regularization_params.keys()))
    assert regularization_params['weight_decay_type'] in {'l1', 'l2'}
    assert np.all(np.array(regularization_params['weight_decays']) >= 0)
    assert regularization_params['adjacent_reg_type'] in {'l1', 'l2', 'l2_fisher'}
    assert np.all(np.array(regularization_params['adjacent_regs']) >= 0)
    assert np.all(np.array(regularization_params['dropouts']) >= 0) and np.all(np.array(regularization_params['dropouts']) < 1)
    
    if loss_weight_increasing:
        assert len(last_time_step_loss_wts) == 1
        assert last_time_step_loss_wts[0] == 1
    assert fileheader[-1] == '_'
    num_time_steps = len(data_loaders_over_time)
    assert len(separate_layers) == num_time_steps - 1
    num_layers_separate = sum([len(time_separate_layers) for time_separate_layers in separate_layers])
    if num_layers_separate == 0:
        combo_name = 'all_shared'
    elif num_layers_separate == (num_blocks + 2) * (num_time_steps - 1):
        combo_name = 'all_separate'
    else:
        combo_name = ':'.join([','.join(time_separate_layers) for time_separate_layers in separate_layers]) + '_separate'
    
    # load best model if allowed to load from disk and files exist
    joint_model_class, model_class, model_properties = get_joint_model_class_and_properties(model_type,
                                                                                            num_blocks,
                                                                                            logger)
    best_hyperparams_filename = fileheader + combo_name + '_best_hyperparams.json'
    best_model_filename       = fileheader + combo_name + '_best_model.pt.gz'
    if allow_load_best_from_disk and os.path.exists(best_hyperparams_filename) and os.path.exists(best_model_filename):
        best_joint_model = joint_model_class(num_time_steps,
                                             separate_layers,
                                             model_properties = model_properties,
                                             num_classes      = num_classes,
                                             mode             = mode,
                                             adapter_ranks    = adapter_ranks,
                                             adapter_mode     = adapter_mode,
                                             side_layers      = side_layers)
        best_joint_model = load_state_dict_from_gz(best_joint_model,
                                                   best_model_filename)
        if torch.cuda.is_available():
            best_joint_model = best_joint_model.cuda()
        
        with open(best_hyperparams_filename, 'r') as f:
            best_hyperparams = json.load(f)
        if best_hyperparams['learning_rate'] in learning_rates \
        and best_hyperparams['weight_decay_type'] == regularization_params['weight_decay_type'] \
        and best_hyperparams['weight_decay'] in regularization_params['weight_decays'] \
        and best_hyperparams['adjacent_reg_type'] == regularization_params['adjacent_reg_type'] \
        and best_hyperparams['adjacent_reg'] in regularization_params['adjacent_regs'] \
        and best_hyperparams['weight_reg_by_time'] == regularization_params['weight_reg_by_time'] \
        and best_hyperparams['dropout'] in regularization_params['dropouts'] \
        and best_hyperparams['last_time_step_loss_wt'] in last_time_step_loss_wts \
        and best_hyperparams['early_stopping_epochs'] <= n_epochs:
            # best hyperparams are among options given
            losses_filename = fileheader + combo_name \
                            + '_lr' + str(best_hyperparams['learning_rate']) \
                            + '_' + best_hyperparams['weight_decay_type'] + '_reg' + str(best_hyperparams['weight_decay']) \
                            + '_' + best_hyperparams['adjacent_reg_type'] \
                            + '_adjreg' + str(best_hyperparams['adjacent_reg']) \
                            + '_dropout' + str(best_hyperparams['dropout']) \
                            + '_finalwt' + str(best_hyperparams['last_time_step_loss_wt']) \
                            + '_' + str(best_hyperparams['early_stopping_epochs']) + 'epochs_losses.json'
            if os.path.exists(losses_filename):
                with open(losses_filename, 'r') as f:
                    losses = json.load(f)
                best_valid_acc = max(losses['valid_step' + str(num_time_steps - 1) + '_accuracies'])

                logger.info('Loaded best model from ' + best_model_filename)
                return best_joint_model, best_valid_acc, best_hyperparams
    
    combo_best_valid_acc = -1
    if sweep_efficient_reg:
        efficient_regs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1] #, 10, 100]
        #efficient_regs = [0.01, 0.1, 1, 10, 100]
        efficient_reg_best_val_accs = []
        assert len(regularization_params['adjacent_regs']) == 1
    else:
        efficient_regs = [0]
    for efficient_reg in efficient_regs:
        if sweep_efficient_reg:
            best_val_acc_for_efficient_reg = -1
        for learning_rate, weight_decay, adjacent_reg, dropout, last_time_step_loss_wt \
            in product(learning_rates, 
                       regularization_params['weight_decays'],
                       regularization_params['adjacent_regs'],
                       regularization_params['dropouts'],
                       last_time_step_loss_wts):
            if loss_weight_increasing:
                time_step_loss_weights = np.array([1./(num_time_steps - t) for t in range(num_time_steps)])
            else:
                time_step_loss_weights = np.ones(num_time_steps)
                time_step_loss_weights[-1] = last_time_step_loss_wt
            hyperparams_fileheader = fileheader + combo_name + '_lr' + str(learning_rate) \
                                   + '_' + str(regularization_params['weight_decay_type']) + '_reg' + str(weight_decay) \
                                   + '_' + str(regularization_params['adjacent_reg_type']) + '_adjreg' + str(adjacent_reg) \
                                   + '_dropout' + str(dropout) \
                                   + '_finalwt' + str(last_time_step_loss_wt) + '_'
            these_regularization_params = {'weight_decay_type' : regularization_params['weight_decay_type'],
                                           'weight_decay'      : weight_decay,
                                           'adjacent_reg_type' : regularization_params['adjacent_reg_type'],
                                           'adjacent_reg'      : adjacent_reg,
                                           'weight_reg_by_time': regularization_params['weight_reg_by_time'],
                                           'dropout'           : dropout}
            if sweep_efficient_reg:
                these_regularization_params['parameter_efficient_weight_decay'] = efficient_reg
            else:
                these_regularization_params['parameter_efficient_weight_decay'] = weight_decay
            joint_model, val_acc, last_joint_model, last_optimizer, n_epochs_stop \
                = fit_joint_model(model_type,
                                  num_blocks,
                                  separate_layers,
                                  data_loaders_over_time,
                                  num_classes,
                                  learning_rate,
                                  these_regularization_params,
                                  time_step_loss_weights,
                                  n_epochs,
                                  hyperparams_fileheader,
                                  logger,
                                  save_model                        = False,
                                  early_stopping_n_epochs           = 5,
                                  start_with_imagenet               = start_with_imagenet,
                                  mode                              = mode,
                                  adapter_ranks                     = adapter_ranks,
                                  adapter_mode                      = adapter_mode,
                                  side_layers                       = side_layers,
                                  partial_model_file_name           = partial_model_file_name,
                                  use_partial_to_init_final         = use_partial_to_init_final,
                                  remove_uncompressed_partial_model = remove_uncompressed_partial_model,
                                  ablate                            = ablate)
            logger.info('Joint ' + model_type + ' model with ' + str(num_blocks) + ' blocks and ' 
                        + combo_name.replace(',', ', ').replace('_', ' ') + ' layers, '
                        + 'learning rate ' + str(learning_rate) + ', '
                        + regularization_params['weight_decay_type'] + ' regularization constant ' + str(weight_decay) + ', '
                        + regularization_params['adjacent_reg_type'] + ' regularization between adjacent time steps ' 
                        + str(adjacent_reg) + ', '
                        + 'regularization constant ' + str(efficient_reg) + ' on parameter-efficient modules, '
                        + 'dropout ' + str(dropout) + ', '
                        + 'and weight on last time step loss ' + str(last_time_step_loss_wt) + ' '
                        + 'achieves val acc: ' + str(val_acc))
            if val_acc > combo_best_valid_acc:
                combo_best_valid_acc        = val_acc
                combo_best_joint_model      = joint_model
                combo_best_last_joint_model = last_joint_model
                combo_best_last_optimizer   = last_optimizer
                combo_best_hyperparams = {'learning_rate'         : learning_rate,
                                          'weight_decay_type'     : regularization_params['weight_decay_type'],
                                          'weight_decay'          : weight_decay,
                                          'adjacent_reg_type'     : regularization_params['adjacent_reg_type'],
                                          'adjacent_reg'          : adjacent_reg,
                                          'weight_reg_by_time'    : regularization_params['weight_reg_by_time'],
                                          'dropout'               : dropout,
                                          'last_time_step_loss_wt': last_time_step_loss_wt,
                                          'early_stopping_epochs' : n_epochs_stop,
                                          'parameter_efficient_weight_decay': efficient_reg}
            if sweep_efficient_reg:
                if val_acc > best_val_acc_for_efficient_reg:
                    best_val_acc_for_efficient_reg = val_acc
        if sweep_efficient_reg:
            efficient_reg_best_val_accs.append(best_val_acc_for_efficient_reg)

    if sweep_efficient_reg:
        with open(fileheader + combo_name + '_sweep_efficient_reg_val_accs.json', 'w') as f:
            json.dump({'Parameter-efficient module regularization': efficient_regs,
                       'Validation accuracy': efficient_reg_best_val_accs},
                      f)
    
    with open(best_hyperparams_filename, 'w') as f:
        json.dump(combo_best_hyperparams,
                  f)
    best_hyperparams_str = 'Best hyperparameters: '
    for hyperparam in combo_best_hyperparams:
        best_hyperparams_str += hyperparam + ' ' + str(combo_best_hyperparams[hyperparam]) + ', '
    logger.info(best_hyperparams_str[:-2])
    logger.info('Best val acc: ' + str(combo_best_valid_acc))

    save_state_dict_to_gz(combo_best_joint_model,
                          best_model_filename)
    logger.info('Saved best model to ' + best_model_filename)
    
    last_model_filename = fileheader + combo_name + '_' + str(combo_best_hyperparams['early_stopping_epochs']) \
                        + 'epochs_last_model.pt.gz'
    save_state_dict_to_gz(combo_best_last_joint_model,
                          last_model_filename)
    logger.info('Saved last model to ' + last_model_filename)
    
    last_optimizer_filename = fileheader + combo_name + '_' + str(combo_best_hyperparams['early_stopping_epochs']) \
                            + 'epochs_last_adam_optimizer.pt.gz'
    save_state_dict_to_gz(combo_best_last_optimizer,
                          last_optimizer_filename)
    logger.info('Saved last Adam optimizer to ' + last_optimizer_filename)
    
    return combo_best_joint_model, combo_best_valid_acc, combo_best_hyperparams

def resume_best_hyperparams_for_a_single_joint_model_combo(model_type,
                                                           num_blocks,
                                                           separate_layers,
                                                           data_loaders_over_time,
                                                           num_classes,
                                                           n_epochs,
                                                           resume_n_epochs,
                                                           fileheader,
                                                           logger,
                                                           regularization_params  = {'weight_decay_type' : 'l2',
                                                                                     'adjacent_reg_type' : 'l2',
                                                                                     'weight_reg_by_time': False},
                                                           loss_weight_increasing = False,
                                                           mode                   = 'separate',
                                                           adapter_ranks          = [0,0,0,0,0,0],
                                                           adapter_mode           = None,
                                                           side_layers            = ['separate',[],[],[],[],'separate'],
                                                           ablate                 = False):
    '''
    Resume fitting for best hyperparameters
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param separate_layers: list of list of str, specify which layers will be new modules at each time step,
                            outer list over time steps (first entry is 2nd time step 
                            since all modules are new at 1st time step), second list), 
                            outer list has length num_time_steps - 1
                            inner list over layers: subset of conv1, layer1, layer2, layer3, layer4, fc,
                            example: [[conv1, layer1], [fc]] means 3 time step model has 
                            - one set of conv1, layer1 at time 0 and another set shared at times 1 and 2
                            - one fc module at times 0 and 1, another fc module at time 2
                            - all other modules shared across all 3 time steps
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step
    @param num_classes: int, number of output classes
    @param n_epochs: int, number of epochs to fit model
    @param resume_n_epochs: int, number of epochs to resume model fitting from
    @param fileheader: str, start of path to save model, ends in _
    @param logger: logger, for INFO messages
    @param regularization_params: dict mapping str to bool or str, contains
                                  - weight_decay_type: str, l1 or l2 for regularization parameter norm
                                  - adjacent_reg_type: str, l1, l2, or l2_fisher for difference between weights
                                                       at consecutive steps (l2_fisher weights L2 regularization 
                                                       between adjacent parameters by Fisher info of parameters 
                                                       at previous time step in previous epoch)
                                  - weight_reg_by_time: bool, if True weight reg on param norm by 1/(t+1) 
                                                        and adjacent diff reg by 1/(T-t)
    @param loss_weight_increasing: bool, specify to weight loss at t by 1/(T - t + 1) instead of alpha at T and 1 elsewhere
    @param mode: str, separate for separate modules at each time step,
                 side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                 low_rank_adapt to use low-rank adapters for residual blocks at each time step
    @param adapter_ranks: list of int, rank of adapters for multiplying or adding to fully connected and convolutional weights
                          for each layer
    @param adapter_mode: str, multiply or add
    @param side_layers: list of str or lists of int, number of output channels 
                        for each of the intermediate convolution layers in side modules,
                        str option is "block" for side module to match original block or "separate" for non-blocks
    @param ablate: bool, set side modules to 0 or low-rank adapters to identity function
                   and adjacent regularization to 10 to check if matches
    @return: 1. joint_block_model, fitted with best hyperparameters
             2. float, validation accuracy at final time step
             3. dict mapping str to float, hyperparameter name to best value
    '''
    assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
    assert adjacent_reg_type in {'l2', 'l2_fisher', 'l1'}
    assert {'weight_decay_type', 'adjacent_reg_type', 'weight_reg_by_time'}.issubset(set(regularization_params.keys()))
    assert regularization_params['weight_decay_type'] in {'l1', 'l2'}
    assert regularization_params['adjacent_reg_type'] in {'l1', 'l2', 'l2_fisher'}
    if ablate:
        assert mode != 'separate'
    
    assert fileheader[-1] == '_'
    num_time_steps = len(data_loaders_over_time)
    assert len(separate_layers) == num_time_steps - 1
    num_layers_separate = sum([len(time_separate_layers) for time_separate_layers in separate_layers])
    if num_layers_separate == 0:
        combo_name = 'all_shared'
    elif num_layers_separate == (num_blocks + 2) * (num_time_steps - 1):
        combo_name = 'all_separate'
    else:
        combo_name = ':'.join([','.join(time_separate_layers) for time_separate_layers in separate_layers]) + '_separate'
    with open(fileheader + combo_name + '_best_hyperparams.json', 'r') as f:
        combo_best_hyperparams = json.load(f)
    assert combo_best_hyperparams['weight_decay_type']  == regularization_params['weight_decay_type']
    assert combo_best_hyperparams['adjacent_reg_type']  == regularization_params['adjacent_reg_type']
    assert combo_best_hyperparams['weight_reg_by_time'] == regularization_params['weight_reg_by_time']
    best_hyperparams_str = 'Resuming from best hyperparameters: '
    for hyperparam in combo_best_hyperparams:
        best_hyperparams_str += hyperparam + ' ' + str(combo_best_hyperparams[hyperparam]) + ', '
    logger.info(best_hyperparams_str[:-2])
        
    # copy last model to path that will be loaded when resuming training
    last_model_filename     = fileheader + combo_name + '_' + str(resume_n_epochs) + 'epochs_last_model.pt.gz'
    last_optimizer_filename = fileheader + combo_name + '_' + str(resume_n_epochs) + 'epochs_last_adam_optimizer.pt.gz'
    best_hyperparams_fileheader = fileheader + combo_name + '_lr' + str(combo_best_hyperparams['learning_rate']) \
                                + '_' + regularization_params['weight_decay_type'] \
                                + '_reg' + str(combo_best_hyperparams['weight_decay']) \
                                + '_' + str(regularization_params['adjacent_reg_type']) \
                                + '_adjreg' + str(combo_best_hyperparams['adjacent_reg']) \
                                + '_dropout' + str(combo_best_hyperparams['dropout']) \
                                + '_finalwt' + str(last_time_step_loss_wt) \
                                + '_' + str(resume_n_epochs) + 'epochs_'
    best_hyperparams_last_model_filename     = best_hyperparams_fileheader + 'last_model.pt.gz'
    best_hyperparams_last_optimizer_filename = best_hyperparams_fileheader + 'last_adam_optimizer.pt.gz'
        
    shutil.copyfile(last_model_filename,
                    best_hyperparams_last_model_filename)
    shutil.copyfile(last_optimizer_filename,
                    best_hyperparams_last_optimizer_filename)
    
    # move best model so it does not get overwritten
    best_model_filename      = fileheader + combo_name + '_best_model.pt.gz'
    best_model_dest_filename = fileheader + combo_name + '_' + str(resume_n_epochs) + 'epochs_best_model.pt.gz'
    shutil.move(best_model_filename, best_model_dest_filename)
    logger.info('Best model from ' + str(resume_n_epochs) + ' epochs moved to ' + best_model_dest_filename)
    
    if loss_weight_increasing:
        time_step_loss_weights = np.array([1./(num_time_steps - t) for t in range(num_time_steps)])
    else:
        time_step_loss_weights = np.ones(num_time_steps)
        time_step_loss_weights[-1] = combo_best_hyperparams['weight on last time step loss']
    
    combo_best_regularization_params = {'weight_decay_type' : regularization_params['weight_decay_type'],
                                        'weight_decay'      : combo_best_hyperparams['weight_decay'],
                                        'adjacent_reg_type' : regularization_params['adjacent_reg_type'],
                                        'adjacent_reg'      : combo_best_hyperparams['adjacent_reg'],
                                        'weight_reg_by_time': regularization_params['weight_reg_by_time'],
                                        'dropout'           : combo_best_hyperparams['dropout']}
    joint_model, val_acc, last_joint_model, last_optimizer, n_epochs_stop \
        = fit_joint_model(model_type,
                          num_blocks,
                          separate_layers,
                          data_loaders_over_time,
                          num_classes,
                          combo_best_hyperparams['learning_rate'],
                          combo_best_hyperparams['weight decay'],
                          combo_best_regularization_params,
                          time_step_loss_weights,
                          n_epochs,
                          fileheader,
                          logger,
                          save_model              = False,
                          resume_from_n_epochs    = resume_n_epochs,
                          early_stopping_n_epochs = 10,
                          mode                    = mode,
                          adapter_ranks           = adapter_ranks,
                          adapter_mode            = adapter_mode,
                          side_layers             = side_layers,
                          ablate                  = ablate)
    
    # remove copies to paths for loading
    os.remove(best_hyperparams_last_model_filename)
    os.remove(best_hyperparams_last_optimizer_filename)
    
    logger.info('Best val acc: ' + str(val_acc))
    
    save_state_dict_to_gz(joint_model,
                          best_model_filename)
    logger.info('Saved best model to ' + best_model_filename)
    
    last_model_filename     = fileheader + combo_name + '_' + str(n_epochs_stop) + 'epochs_last_model.pt.gz'
    save_state_dict_to_gz(last_joint_model,
                          last_model_filename)
    logger.info('Saved last model to ' + last_model_filename)
    
    last_optimizer_filename = fileheader + combo_name + '_' + str(n_epochs_stop) + 'epochs_last_adam_optimizer.pt.gz'
    save_state_dict_to_gz(last_optimizer,
                          last_optimizer_filename)
    logger.info('Saved last Adam optimizer to ' + last_optimizer_filename)
    
    return joint_model, val_acc, combo_best_hyperparams
    
def run_best_layer_combo_search_for_joint_model(model_type,
                                                num_blocks,
                                                data_loaders_over_time,
                                                num_classes,
                                                learning_rates,
                                                regularization_params,
                                                last_time_step_loss_wts,
                                                n_epochs,
                                                fileheader,
                                                logger,
                                                loss_weight_increasing = False,
                                                start_with_imagenet    = False,
                                                mode                   = 'separate',
                                                adapter_ranks          = [0, 0, 0, 0, 0, 0],
                                                adapter_mode           = None,
                                                side_layers            = ['separate',[],[],[],[],'separate'],
                                                ablate                 = False):
    '''
    Find best combinations of layers to have separate/shared for all time steps
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step
    @param num_classes: int, number of output classes
    @param learning_rates: list of floats, learning rates to try
    @param regularization_params: dict mapping str to bool, str, or list of float, contains
                                  - weight_decay_type: str, l1 or l2 for regularization parameter norm
                                  - weight_decays: list of float, regularization constant for parameter norm
                                  - adjacent_reg_type: str, l1, l2, or l2_fisher for difference between weights
                                                       at consecutive steps (l2_fisher weights L2 regularization 
                                                       between adjacent parameters by Fisher info of parameters 
                                                       at previous time step in previous epoch)
                                  - adjacent_regs: list of float, constant for regularizing adjacent differences
                                  - weight_reg_by_time: bool, if True weight reg on param norm by 1/(t+1) 
                                                        and adjacent diff reg by 1/(T-t)
                                  - dropouts: list of float, probability each output entry after each layer/block is zero'd 
                                              during training
    @param last_time_step_loss_wts: list of float, try giving these weights to final time step 
                                    when averaging losses over time steps
    @param n_epochs: int, number of epochs to fit model
    @param fileheader: str, start of path to save model, ends in _
    @param logger: logger, for INFO messages
    @param loss_weight_increasing: bool, specify to weight loss at t by 1/(T - t + 1) instead of alpha at T and 1 elsewhere
    @param start_with_imagenet: bool, specify to initialize all time steps with ImageNet
    @param mode: str, separate for separate modules at each time step,
                 side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                 low_rank_adapt to use low-rank adapters for residual blocks at each time step
    @param adapter_ranks: list of int, rank of adapters for multiplying or adding to fully connected and convolutional weights
                          in each layer
    @param adapter_mode: str, multiply or add
    @param side_layers: list of str or lists of int, number of output channels 
                        for each of the intermediate convolution layers in side modules,
                        str option is "block" for side module to match original block or "separate" for non-block layers
    @param ablate: bool, set side modules to 0 or low-rank adapters to identity function
                   and adjacent regularization to 10 to check if matches
    @return: 1. joint_block_model, best fitted model
             2. str, combination of layers that are separate in the best fitted model, 
                comma-separated layers at each time step, colon-separated time steps
    '''
    assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
    if mode == 'low_rank_adapt':
        assert adapter_mode in {'add', 'multiply'}
    if ablate:
        assert mode != 'separate'
    assert {'weight_decay_type', 'weight_decays', 'adjacent_reg_type', 
            'adjacent_regs', 'weight_reg_by_time', 'dropouts'}.issubset(set(regularization_params.keys()))
    assert regularization_params['weight_decay_type'] in {'l1', 'l2'}
    assert np.all(np.array(regularization_params['weight_decays']) >= 0)
    assert regularization_params['adjacent_reg_type'] in {'l1', 'l2', 'l2_fisher'}
    assert np.all(np.array(regularization_params['adjacent_regs']) >= 0)
    assert np.all(np.array(regularization_params['dropouts']) >= 0) and np.all(np.array(regularization_params['dropouts']) < 1)
    
    assert fileheader[-1] == '_'
    if os.path.exists(fileheader + 'best_combo.txt'):
        with open(fileheader + 'best_combo.txt', 'r') as f:
            combo = f.read()
        model_filename = fileheader + combo + '_model.pt.gz'
        joint_model_class, model_class, model_properties = get_joint_model_class_and_properties(model_type,
                                                                                                num_blocks,
                                                                                                logger)
        joint_model = joint_model_class(num_time_steps,
                                        separate_layers,
                                        model_properties = model_properties,
                                        num_classes      = num_classes,
                                        mode             = mode,
                                        adapter_ranks    = adapter_ranks,
                                        adapter_mode     = adapter_mode,
                                        side_layers      = side_layers)
        joint_model = load_state_dict_from_gz(joint_model,
                                              model_filename)
        if torch.cuda.is_available():
            joint_model = joint_model.cuda()
        logger.info('Best combo: ' + combo)
        logger.info('Loaded model from ' + model_filename)
        return joint_model, combo
    
    combos = define_layer_combos(model_type,
                                 num_blocks)
    
    best_valid_acc = -1
    for combo in combos:
        combo_adjacent_regs = deepcopy(adjacent_regs)
        if len(combo) == 0:
            combo_name = 'all_shared'
        elif combo == 'all':
            combo_name = 'all_separate'
            if 0 in combo_adjacent_regs:
                combo_adjacent_regs.remove(0)
        else:
            combo_name = ':'.join([combo for _ in range(num_time_steps)]) + '_separate'
        separate_layers = [combo.split(',') for _ in range(num_time_steps)]
        
        combo_best_joint_model, combo_best_valid_acc, combo_best_hyperparams \
            = tune_joint_model_hyperparameters_for_a_single_combo(model_type,
                                                                  num_blocks,
                                                                  separate_layers,
                                                                  data_loaders_over_time,
                                                                  num_classes,
                                                                  learning_rates,
                                                                  regularization_params,
                                                                  last_time_step_loss_wts,
                                                                  n_epochs,
                                                                  fileheader,
                                                                  logger,
                                                                  loss_weight_increasing = loss_weight_increasing,
                                                                  start_with_imagenet    = start_with_imagenet,
                                                                  mode                   = mode,
                                                                  adapter_ranks          = adapter_ranks,
                                                                  adapter_mode           = adapter_mode,
                                                                  side_layers            = side_layers,
                                                                  ablate                 = ablate)
                
        if combo_best_valid_acc > best_valid_acc:
            best_valid_acc      = combo_best_valid_acc
            best_joint_model    = combo_best_joint_model
            best_combo_name     = combo_name
            best_hyperparams    = combo_best_hyperparams
    
    logger.info('Best combo: ' + ', '.join(best_combo))
    with open(fileheader + 'best_combo.txt', 'w') as f:
        f.write(best_combo_name)
    for hyperparam in best_hyperparams:
        logger.info('Best ' + hyperparam + ': ' + str(best_hyperparams[hyperparam]))
    logger.info('Best val acc: ' + str(best_valid_acc))
    
    return best_joint_model, best_combo_name

def visualize_weight_differences(joint_model,
                                 filename):
    '''
    Plot L2 norm of difference between weights at consecutive time steps for each layer
    @param joint_model: joint_block_model
    @param filename: str, path to save plot
    @return: None
    '''
    assert joint_model.mode == 'separate'
    layers = get_plot_layer_names(joint_model.model_type,
                                  joint_model.num_blocks)
    weight_differences = dict()
    for layer_idx in range(len(layers)):
        with torch.no_grad():
            layer_weight_diff = joint_model.layers[layer_idx].compute_adjacent_param_norm_at_each_time_step().detach()
            if torch.cuda.is_available():
                layer_weight_diff = layer_weight_diff.cpu()
        weight_differences[layers[layer_idx]] = np.sqrt(layer_weight_diff.numpy())
        
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
        ax[int(layer_idx / 3), int(layer_idx % 3)].plot(np.arange(1, joint_model.num_time_steps),
                                                        weight_differences[layers[layer_idx]],
                                                        linewidth = 3)
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_title(layers[layer_idx])
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_ylim(bottom = 0)
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_xlim([1, joint_model.num_time_steps - 1])
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_xlim([1, joint_model.num_time_steps - 1])
        ax[int(layer_idx / 3), int(layer_idx % 3)].set_xticks(range(1, joint_model.num_time_steps))
        if int(layer_idx / 3) == 1:
            ax[int(layer_idx / 3), int(layer_idx % 3)].set_xlabel('Time step')
        if int(layer_idx % 3) == 0:
            ax[int(layer_idx / 3), int(layer_idx % 3)].set_ylabel('L2 norm of weight change')
    
    fig.tight_layout()
    fig.savefig(filename)
    
def find_nearest_neighbors(data_loaders_over_time,
                           fileheader,
                           logger,
                           optimal_matching = False):
    '''
    For each sample in the training set at time T, find its nearest neighbor at time T - 1 with the same label
    Then, for those neighbors, find the nearest neighbor at time T - 2, etc. until time 0
    Create train data loaders for times 0 to T - 1 containing these nearest neighbors instead
    Plot histograms showing how many times each sample is used as a nearest neighbor at each time step
    @param data_loaders_over_time: list of dict mapping str to torch DataLoaders, list over time steps, 
                                   loaders for data splits at each time step, will be modified and returned
    @param fileheader: str, start of file path to save histograms showing how many times each sample 
                       is used as a nearest neighbor at each time step
    @param logger: logger, for INFO messages
    @param optimal_matching: bool, match samples to minimize total distance, 
                             spreads weight to more samples at earlier time steps
    @return: data_loaders_over_time with train loaders at times 0 to T - 1 replaced with nearest neighbors
    '''
    num_time_steps = len(data_loaders_over_time)
    num_cols       = 2
    num_rows       = int(math.ceil(num_time_steps/num_cols))
    fig, ax = plt.subplots(nrows   = num_rows,
                           ncols   = num_cols,
                           figsize = (num_cols * 6.4, num_rows * 4.8),
                           sharex  = True,
                           sharey  = True)
    row_idx = 0
    col_idx = 0
    x_name  = '# time sample is neighbor'
    sample_freq_df = pd.DataFrame(data    = {x_name: np.ones(len(data_loaders_over_time[-1]['train']))},
                                  columns = [x_name])
    sns.histplot(data     = sample_freq_df,
                 x        = x_name,
                 discrete = True,
                 stat     = 'probability',
                 ax       = ax[row_idx, col_idx])
    ax[row_idx, col_idx].set_title('Time ' + str(num_time_steps - 1))
    col_idx += 1
    batch_size = data_loaders_over_time[0]['train'].batch_size
    for t in range(len(data_loaders_over_time) - 2, -1, -1):
        t_plus_one_images, t_plus_one_labels = data_loaders_over_time[t+1]['train'].dataset.tensors
        t_images, t_labels = data_loaders_over_time[t]['train'].dataset.tensors
        t_neighbor_images  = []
        t_neighbor_idxs    = []
        if optimal_matching:
            label_classes     = torch.unique(t_plus_one_labels)
            t_neighbor_labels = []
            for label_class in label_classes:
                start_time    = time.time()
                t_plus_one_label_idxs = torch.nonzero(torch.where(t_plus_one_labels == label_class, 1, 0), 
                                                      as_tuple = True)[0]
                t_label_idxs          = torch.nonzero(torch.where(t_labels == label_class, 1, 0), 
                                                      as_tuple = True)[0]
                # (# samples at t + 1) x (# samples at t)
                distance_matrix       = np.array([[float(torch.norm(t_plus_one_images[i] - t_images[j]))
                                                   for j in t_label_idxs]
                                                  for i in t_plus_one_label_idxs])
                sample_size_ratio     = math.ceil(len(t_plus_one_label_idxs)*6./(len(t_label_idxs)*5.))
                distance_matrix       = np.tile(distance_matrix, (1, sample_size_ratio))
                munkres_alg           = Munkres()
                match_indices         = munkres_alg.compute(distance_matrix)
                selected_t_label_idxs = [t_idx for _, t_idx in match_indices[:len(t_plus_one_label_idxs)]]
                t_neighbor_idxs.extend(selected_t_label_idxs)
                t_neighbor_images.extend([t_images[i] for i in selected_t_label_idxs])
                t_neighbor_labels.extend([t_labels[i] for i in selected_t_label_idxs])
                logger.info('Time to run Kuhn-Munkres algorithm for label class ' + str(int(label_class))
                            + ' at time ' + str(t) + ': ' + str(time.time() - start_time) + ' seconds')
            t_neighbor_labels = torch.LongTensor(t_neighbor_labels)
        else:
            for t_plus_one_idx in range(len(t_plus_one_labels)):
                t_same_label_idxs = torch.nonzero(torch.where(t_labels == t_plus_one_labels[t_plus_one_idx], 1, 0), 
                                                  as_tuple = True)[0]
                t_dists = np.array([float(torch.norm(t_images[i] - t_plus_one_images[t_plus_one_idx])) 
                                    for i in t_same_label_idxs])
                t_neighbor_idx = t_same_label_idxs[np.argmin(t_dists)]
                t_neighbor_images.append(t_images[t_neighbor_idx])
                t_neighbor_idxs.append(int(t_neighbor_idx))
            t_neighbor_labels = deepcopy(t_plus_one_labels)
        t_neighbor_idxs = np.array(t_neighbor_idxs)
        t_neighbor_images = torch.stack(t_neighbor_images)
        shuffled_idxs = np.arange(len(t_neighbor_idxs))
        np.random.shuffle(shuffled_idxs)
        t_neighbor_images = t_neighbor_images[shuffled_idxs]
        t_neighbor_labels = t_neighbor_labels[shuffled_idxs]
        t_neighbor_dataset = torch.utils.data.TensorDataset(t_neighbor_images, t_neighbor_labels)
        data_loaders_over_time[t]['train'] = torch.utils.data.DataLoader(t_neighbor_dataset,
                                                                         batch_size = batch_size,
                                                                         shuffle    = True)
        
        _, sample_freqs = np.unique(t_neighbor_idxs, return_counts = True)
        sample_freqs    = np.concatenate((sample_freqs, np.zeros(len(t_labels) - len(sample_freqs))))
        sample_freq_df  = pd.DataFrame(data    = {x_name: sample_freqs},
                                       columns = [x_name])
        if col_idx == num_cols:
            col_idx = 0
            row_idx += 1
        sns.histplot(data     = sample_freq_df,
                     x        = x_name,
                     discrete = True,
                     stat     = 'probability',
                     ax       = ax[row_idx, col_idx])
        ax[row_idx, col_idx].set_title('Time ' + str(t))
        ax[row_idx, col_idx].set_yscale('log')
        col_idx += 1
    
    while col_idx < num_cols:
        ax[num_rows - 1, col_idx].axis('off')
        ax[num_rows - 2, col_idx].set_xlabel(x_name)
        col_idx += 1
    fig.tight_layout()
    fig.savefig(fileheader + 'nearest_neighbor_freq.pdf')
        
    return data_loaders_over_time

def plot_efficient_reg_val_accs(fileheader,
                                plot_title):
    '''
    Read validation accuracies and parameter-efficient regularizations from json and plot
    @param fileheader: str, start of path with _sweep_efficient_reg_val_accs.json appended to read values from,
                       will save plot with _sweep_efficient_reg_val_accs.pdf appended
    @param plot_title: str, title of plot
    @return: None
    '''
    json_filename = fileheader + '_sweep_efficient_reg_val_accs.json'
    with open(json_filename, 'r') as f:
        val_accs = json.load(f)
    reg_name = 'Parameter-efficient module regularization'
    acc_name = 'Validation accuracy'
    df = pd.DataFrame(data    = {reg_name: val_accs[reg_name],
                                 acc_name: val_accs[acc_name]},
                      columns = [reg_name, acc_name])
    fig, ax = plt.subplots()
    sns.lineplot(data = df,
                 x    = reg_name,
                 y    = acc_name,
                 ax   = ax)
    ax.set_xscale('log')
    ax.set_title(plot_title)
    fig.savefig(fileheader + '_sweep_efficient_reg_val_accs.pdf')
    
def run_joint_model_experiment(dataset_name,
                               model_type,
                               num_blocks,
                               shift_sequence,
                               source_sample_size,
                               target_sample_sizes,
                               final_target_test_size,
                               seed                   = 1007,
                               visualize_shifts       = False,
                               separate_layers        = 'all',
                               resume_n_epochs        = (0, 100),
                               weight_reg_by_time     = False,
                               loss_samples           = 'all',
                               loss_weight_increasing = False,
                               adjacent_reg_type      = 'l2',
                               start_with_imagenet    = False,
                               mode                   = 'separate',
                               adapter_rank           = 0,
                               adapter_mode           = None,
                               side_layer_sizes       = [],
                               ablate                 = False,
                               sweep_efficient_reg    = False,
                               all_layers_efficient   = False):
    '''
    Learn best joint model for shift sequence. Compute test accuracy at final time step.
    @param dataset_name: str, cifar10, cifar100, or portraits
                         for portraits, shift_sequence, source_sample_size, target_sample_sizes, 
                         and final_target_test_size arguments are disregarded
    @param model_type: str, resnet, densenet, or convnet
    @param num_blocks: int, number of blocks in model
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of corruption, rotation, label_flip, 
                           label_shift, rotation_cond, recoloring, recoloring_cond, subpop
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param seed: int, seed for data generation
    @param visualize_shifts: bool, whether to visualize images from shift sequence
    @param separate_layers: str, all to have separate layers at each time step,
                            search to find best combination for layers to have separate at all time steps,
                            colon-separated time steps with comma-separated layers for which modules are new at each time step,
                            layers: conv1, layer1, layer2, layer3, layer4, fc
    @param resume_n_epochs: tuple of ints, 
                            # epochs to resume from (0 to start from pre-trained weights),
                            total # epochs to train
    @param weight_reg_by_time: bool, if True weight parameter norm reg by 1/(t+1) and adjacent diff reg by 1/(T-t)
    @param loss_samples: str, 'all' to use all previous samples,
                         'nearest neighbors' to use nearest neighbor with same label in previous time steps,
                         'optimal matching' to use sample with same label in previous time steps minimizing total distance
    @param loss_weight_increasing: bool, specify to weight loss at t by 1/(T - t + 1) instead of alpha at T and 1 elsewhere
    @param adjacent_reg_type: str, none to not penalize difference between adjacent parameters (ablate to erm_final),
                              l2 to penalize L2 norm of difference between adjacent parameters,
                              l2_fisher to weight L2 norms by Fisher info of parameters at previous time step in previous epoch
                              l2_high to set adjacent regularization to 10 (ablate to erm_all),
                              l1 to use L1 norm instead,
                              not applied if there are no separate layers
    @param start_with_imagenet: bool, specify to initialize all time steps with ImageNet
    @param mode: str, separate for separate modules at each time step,
                 side_tune to add 1 convolution layer side modules instead of residual blocks at each time step,
                 low_rank_adapt to use low-rank adapters for residual blocks at each time step
    @param adapter_rank: int, rank of adapters for multiplying or adding to fully connected and convolutional weights
    @param adapter_mode: str, multiply or add
    @param side_layer_sizes: str or list of int, number of output channels for each of the intermediate convolution layers 
                             in side modules, same for all residual block layers,
                             str option is "block" for side module to be a block
                             or "separate" for non-block layer to be a new module that is not added to previous modules
    @param ablate: bool, set side modules to 0 or low-rank adapters to identity function
                   and adjacent regularization to 10 to check if matches erm_all
    @param sweep_efficient_reg: bool, whether to plot validation performance against different regularization constants 
                                for the side modules or low-rank adapters. In constrast, default is same regularization 
                                constant on parameter-efficient modules as t = 0 modules.
    @param all_layers_efficient: bool, whether to make non-block modules parameter-efficient if mode is not separate,
                                 side modules for non-block modules are single layer modules,
                                 low-rank adapters for non-block modules are fixed at rank 10,
                                 other than the input conv layer in resnets that can only be rank 1 to be more efficient,
                                 when False, non-block modules are just separate
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100', 'portraits'}
    assert model_type in {'resnet', 'convnet', 'densenet'}
    assert num_blocks >= 1
    real_world_datasets = {'portraits'}
    if dataset_name not in real_world_datasets:
        assert source_sample_size >= 1
        assert np.all(np.array(target_sample_sizes) >= 1)
        assert final_target_test_size >= 1
    assert mode in {'separate', 'side_tune', 'low_rank_adapt'}
    if mode == 'low_rank_adapt':
        assert adapter_rank > 0
        assert adapter_mode in {'add', 'multiply'}
    if mode != 'separate':
        assert adjacent_reg_type != 'l2_fisher'
        assert not weight_reg_by_time
    if ablate or sweep_efficient_reg:
        assert mode != 'separate'
    has_separate_layers = ((mode == 'separate') or (mode == 'side_tune' and 'separate' in side_layer_sizes)
                           or (mode == 'low_rank_adapt' and 0 in adapter_ranks))
    
    # set up logger
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
        num_target_steps = 7
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    joint_model_variant = 'joint_model'
    if mode != 'separate':
        joint_model_variant += '_' + mode
        if mode == 'low_rank_adapt':
            joint_model_variant += '_' + str(adapter_mode) + '_rank' + str(adapter_rank)
        if mode == 'side_tune' and len(side_layer_sizes) > 0:
            if isinstance(side_layer_sizes, str):
                assert side_layer_sizes == 'block'
                joint_model_variant += '_block'
            else:
                joint_model_variant += '_' + str(len(side_layer_sizes) + 1) + 'layers_size' \
                                     + ', '.join([str(size) for size in side_layer_sizes])
        if ablate:
            joint_model_variant += '_ablate'
    if weight_reg_by_time:
        joint_model_variant += '_weight_reg_by_time'
    if loss_samples != 'all':
        joint_model_variant += '_' + loss_samples.replace(' ', '_') + '_loss'
    if loss_weight_increasing:
        joint_model_variant += '_increasing'
    joint_model_variant += '_adjacent_reg_' + adjacent_reg_type
    if start_with_imagenet:
        joint_model_variant += '_start_with_imagenet'
    model_architecture = model_type + '_' + str(num_blocks) + 'blocks'
    method_dir  = config.output_dir + model_architecture + '_' + dataset_name + '_experiment/' + joint_model_variant + '/' \
                + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    skip_model_learning = False
    if os.path.exists(method_dir + 'metrics.csv'):
        if not sweep_efficient_reg:
            return # this experiment has been run before
        skip_model_learning = True # skip learning models and just create efficient regularization plot at the end
    logging_dir = method_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'run_joint_model_learning_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Learning a joint model')
    logger.info('Dataset: '                + dataset_name)
    logger.info('Model architecture: '     + model_type)
    logger.info('# blocks: '               + str(num_blocks))
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: ' + ', '.join([str(target_sample_size) 
                                                         for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    logger.info('Separate layers: '        + separate_layers)
    logger.info('Weight reg by time: '     + str(weight_reg_by_time))
    logger.info('Loss samples: '           + loss_samples)
    logger.info('Loss weight increasing: ' + str(loss_weight_increasing))
    logger.info('Adjacent reg type: '      + str(adjacent_reg_type))
    logger.info('Start with ImageNet: '    + str(start_with_imagenet))
    logger.info('Mode: '                   + mode)
    if mode == 'low_rank_adapt':
        logger.info('Adapter rank: '       + str(adapter_rank))
        logger.info('Adapter mode: '       + str(adapter_mode))
    if mode == 'side_tune':
        if model_type == 'convnet':
            assert isinstance(side_layer_sizes, str)
        if isinstance(side_layer_sizes, str):
            assert side_layer_sizes == 'block'
            logger.info('Side modules are blocks')
        else:
            logger.info('Side modules: '       + ('No intermediate layers' if len(side_layer_sizes) == 0
                                                  else 'intermediate output sizes are ' 
                                                  + ', '.join([str(size) for size in side_layer_sizes])))
    logger.info('Ablate: '                 + str(ablate))
    logger.info('Plotting performance vs regularization on parameter-efficient modules: ' + str(sweep_efficient_reg))
    logger.info('Making non-block layers parameter-efficient: ' + str(all_layers_efficient))
    
    # set model names and file headers
    model_dir = method_dir + 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fileheader = model_dir + 'joint_model_'
    
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
                                                         seed,
                                                         visualize_shifts,
                                                         model_dir)
    
    if loss_samples != 'all':
        optimal_matching = (loss_samples == 'optimal matching')
        data_loaders_over_time = find_nearest_neighbors(data_loaders_over_time,
                                                        fileheader,
                                                        logger,
                                                        optimal_matching = optimal_matching)
    
    if dataset_name =='portraits':
        num_classes = 2
    elif dataset_name == 'cifar10':
        num_classes = 10
    else:
        # cifar-100
        num_classes = 20
    
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
    
    if not skip_model_learning:
        if separate_layers == 'search':
            assert not sweep_efficient_reg
            if adjacent_reg_type == 'l2_very_high':
                adjacent_regs       = [1e4]
                adjacent_reg_type   = 'l2'
            elif adjacent_reg_type == 'l2_high':
                adjacent_regs       = [10]
                adjacent_reg_type   = 'l2'
            elif adjacent_reg_type == 'none':
                adjacent_regs       = [0]
                adjacent_reg_type   = 'l2'
            else:
                adjacent_regs       = [0.1]
            learning_rates          = [1e-3]
            weight_decays           = [0.0001]
            dropouts                = [0]
            last_time_step_loss_wts = [1]
            n_epochs                = int((resume_n_epochs[1] - resume_n_epochs[0])/2 + resume_n_epochs[0])
            regularization_params   = {'weight_decay_type' : 'l2',
                                       'weight_decays'     : weight_decays,
                                       'adjacent_reg_type' : adjacent_reg_type,
                                       'adjacent_regs'     : adjacent_regs,
                                       'weight_reg_by_time': weight_reg_by_time,
                                       'dropouts'          : dropouts}
            best_joint_model, _ \
                = run_best_layer_combo_search_for_joint_model(model_type,
                                                              num_blocks,
                                                              data_loaders_over_time,
                                                              learning_rates,
                                                              regularization_params,
                                                              last_time_step_loss_wts,
                                                              n_epochs,
                                                              fileheader,
                                                              logger,
                                                              loss_weight_increasing = loss_weight_increasing,
                                                              start_with_imagenet    = start_with_imagenet, 
                                                              mode                   = mode,
                                                              adapter_ranks          = adapter_ranks,
                                                              adapter_mode           = adapter_mode,
                                                              side_layers            = side_layers,
                                                              ablate                 = ablate)
        else:
            if adjacent_reg_type == 'none':
                adjacent_regs     = [0]
                adjacent_reg_type = 'l2'
            elif sweep_efficient_reg:
                adjacent_regs = [0.1]
                adjacent_reg_type = 'l2'
            elif adjacent_reg_type == 'l2_very_high':
                adjacent_regs     = [1e4, 1e5, 1e6]
                adjacent_reg_type = 'l2'
            elif adjacent_reg_type == 'l2_high':
                adjacent_regs     = [10, 100, 1000]
                adjacent_reg_type = 'l2'
            else:
                adjacent_regs       = [0.01, 0.1, 1]
            learning_rates          = [1e-3] #[1e-3, 1e-4]
            weight_decays           = [0.0001] #[0.0001, 0.01]
            dropouts                = [0.5] #[0, 0.5]
            if loss_weight_increasing:
                last_time_step_loss_wts = [1]
            else:
                last_time_step_loss_wts = [3] #[1, 3]
            n_epochs                = resume_n_epochs[1]
            regularization_params   = {'weight_decay_type' : 'l2',
                                       'weight_decays'     : weight_decays,
                                       'adjacent_reg_type' : adjacent_reg_type,
                                       'adjacent_regs'     : adjacent_regs,
                                       'weight_reg_by_time': weight_reg_by_time,
                                       'dropouts'          : dropouts}
            if separate_layers == 'all':
                separate_layers_list = [['conv1'] + ['layer' + str(i + 1) for i in range(num_blocks)] + ['fc']
                                        for _ in range(num_target_steps)]
            else:
                separate_layers_list = [time_layers.split(',') for time_layers in separate_layers.split(':')]
                if len(separate_layers_list) == 1:
                    separate_layers_list = [separate_layers_list[0] for _ in range(num_target_steps)]
            if resume_n_epochs[0] > 0:
                assert not sweep_efficient_reg
                # learning_rates, weight_decays, adjacent_regs, dropouts set above are ignored
                best_joint_model, _, _ \
                    = resume_best_hyperparams_for_a_single_joint_model_combo(model_type,
                                                                             num_blocks,
                                                                             separate_layers_list,
                                                                             data_loaders_over_time,
                                                                             num_classes,
                                                                             n_epochs,
                                                                             resume_n_epochs[0],
                                                                             fileheader,
                                                                             logger,
                                                                             regularization_params  = regularization_params,
                                                                             loss_weight_increasing = loss_weight_increasing,
                                                                             mode                   = mode,
                                                                             adapter_ranks          = adapter_ranks,
                                                                             adapter_mode           = adapter_mode,
                                                                             side_layers            = side_layers,
                                                                             ablate                 = ablate)
            else:
                best_joint_model, _, _ \
                    = tune_joint_model_hyperparameters_for_a_single_combo(model_type,
                                                                          num_blocks,
                                                                          separate_layers_list,
                                                                          data_loaders_over_time,
                                                                          num_classes,
                                                                          learning_rates,
                                                                          regularization_params,
                                                                          last_time_step_loss_wts,
                                                                          n_epochs,
                                                                          fileheader,
                                                                          logger,
                                                                          loss_weight_increasing = loss_weight_increasing,
                                                                          start_with_imagenet    = start_with_imagenet,
                                                                          mode                   = mode,
                                                                          adapter_ranks          = adapter_ranks,
                                                                          adapter_mode           = adapter_mode,
                                                                          side_layers            = side_layers,
                                                                          ablate                 = ablate,
                                                                          sweep_efficient_reg    = sweep_efficient_reg)

        final_time_step = len(data_loaders_over_time) - 1
        final_test_accuracy, _ = compute_joint_model_accuracy_for_a_time_step(best_joint_model,
                                                                              data_loaders_over_time[final_time_step]['test'],
                                                                              final_time_step)
        logger.info('Test accuracy at final time step: ' + str(final_test_accuracy))

        if mode == 'separate' and separate_layers == 'all':
            visualize_weight_differences(best_joint_model,
                                         fileheader + separate_layers + '_weight_differences.pdf')

        model_name    = model_type + ' ' + str(num_blocks) + ' blocks'
        df = pd.DataFrame(data    = {'Model'         : [model_name],
                                     'Method'        : [joint_model_variant],
                                     'Shift sequence': [shift_sequence_with_sample_size],
                                     'Seed'          : [seed],
                                     'Test accuracy' : [final_test_accuracy]},
                          columns = ['Model', 'Method', 'Shift sequence', 'Seed', 'Test accuracy'])
        df.to_csv(method_dir + 'metrics.csv',
                  index = False)
    
    if sweep_efficient_reg:
        assert separate_layers != 'search'
        if len(separate_layers) == 0:
            combo_name = 'all_shared'
        elif separate_layers == 'all' or len(separate_layers.split(',')) == num_blocks + 2:
            layers = ['conv1'] + ['layer' + str(layer_idx) for layer_idx in range(1, num_blocks + 1)] + ['fc']
            combo_name = ':'.join([','.join(layers) for _ in range(num_target_steps)]) + '_separate'
        else:
            combo_name = ':'.join([separate_layers for _ in range(num_target_steps)]) + '_separate'
        if mode == 'separate':
            readable_mode = 'separate modules'
        elif mode == 'side_tune':
            readable_mode = str(len(side_layers) + 1) + '-layer side modules'
        else:
            if adapter_mode == 'add':
                readable_adapter_mode = 'additive'
            else:
                readable_adapter_mode = 'multiplicative'
            readable_mode = 'rank-' + str(adapter_ranks[1]) + ' ' + readable_adapter_mode + ' adapters'
        if model_type == 'resnet':
            model_type_readable = 'ResNet'
        elif model_type == 'convnet':
            model_type_readable = 'ConvNet'
        else:
            model_type_readable = 'DenseNet'
        plot_efficient_reg_val_accs(fileheader + combo_name,
                                    model_type_readable + ' with ' + str(num_blocks) + ' blocks and ' + readable_mode)
    
def create_parser():
    '''
    Create argument parser
    @return: ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = 'Fit a joint model.')
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
                        help    = ('Specify colon-separated sequence of shifts at each time step. '
                                   'Each step is a comma-separated combination of corruption, rotation, '
                                   'label_flip, label_shift, recolor, recolor_cond, rotation_cond, subpop.'))
    parser.add_argument('--source_sample_size',
                        action  = 'store',
                        type    = int,
                        default = 10000,
                        help    = 'Specify total number of training/validation samples source domain.')
    parser.add_argument('--target_sample_size',
                        action  = 'store',
                        type    = int,
                        default = 1000,
                        help    = 'Specify total number of training/validation samples for each target domain.')
    parser.add_argument('--target_test_size',
                        action  = 'store',
                        type    = int,
                        default = 1000,
                        help    = 'Specify number of test samples for final target domain.')
    parser.add_argument('--target_sample_size_seq',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = ('Specify colon-separated sequence of sample sizes for target domains. '
                                   'Overrides target_sample_size argument for training/validation.'))
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
    parser.add_argument('--visualize_images',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether to visualize examples from each time step.')
    parser.add_argument('--separate_layers',
                        action  = 'store',
                        type    = str,
                        default = 'all',
                        help    = ('Specify which layers are separate at each time step. '
                                   'Default is all layers are separate. '
                                   'search will find best combination of layers to be separate at all time steps.'
                                   'Otherwise specify comma-separated list of new modules introduced at each time step '
                                   ' with time steps separated by colons.'))
    parser.add_argument('--resume',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify to resume fitting for best hyperparameters instead of searching all settings. '
                                   'Only applies when not searching layers.'))
    parser.add_argument('--weight_reg_by_time',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify to weight parameter norm regularization by 1/(t+1) '
                                   'and adjacent diff regularization by 1/(T-t).'))
    parser.add_argument('--loss_samples',
                        action  = 'store',
                        type    = str,
                        default = 'all',
                        help    = ('Specify all to use all previous samples in historical loss, '
                                   'nearest neighbors to use nearest neighbor path for previous samples, '
                                   'or optimal matching to use optimal pairing that minimizes total distance.'))
    parser.add_argument('--loss_weight_increasing',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify to weight loss at each t by 1/(T - t + 1). '
                                   'Default: higher alpha weight at final time point, equal at all previous.'))
    parser.add_argument('--adjacent_reg_type',
                        action  = 'store',
                        type    = str,
                        default = 'l2',
                        help    = ('Specify none to not penalize difference between adjacent parameters (ablate to erm_final). '
                                   'l2 to penalize L2 norm of difference between adjacent parameters. '
                                   'l2_fisher to weight L2 norms by Fisher info. '
                                   'l2_high to use higher L2 norm (ablate towards erm_all). '
                                   'l2_very_high to use even higher L2 norm. '
                                   'l1 to use L1 norm instead.'))
    parser.add_argument('--start_with_imagenet',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to initialize weights at all time steps with ImageNet.')
    parser.add_argument('--mode',
                        action  = 'store',
                        type    = str,
                        default = 'separate',
                        help    = ('Specify separate for separate layers at each time step. '
                                   'side_tune to add side modules for each residual block. '
                                   'low_rank_adapt to add low-rank adapters for layers in residual blocks.'))
    parser.add_argument('--adapter_rank',
                        action  = 'store',
                        type    = int,
                        default = 0,
                        help    = ('Specify rank of low-rank adapters that are multiplied to weights at each time step '
                                   'instead of using separate layers.'))
    parser.add_argument('--adapter_mode',
                        action  = 'store',
                        type    = str,
                        default = None,
                        help    = 'Specify whether to multiply or add low-rank adapters.')
    parser.add_argument('--side_layer_sizes',
                        action  = 'store',
                        type    = str,
                        default = '',
                        help    = ('Specify comma-separated list of number of output channels in intermediate convolution '
                                   'layers of side modules. Default is 1 layer that maps input size to output size. '
                                   'Another option is block for each side module to match original block.'))
    parser.add_argument('--ablate',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to ablate side modules or low-rank adapters.')
    parser.add_argument('--sweep_efficient_reg',
                        action  = 'store_true',
                        default = False,
                        help    = ('Specify to plot validation accuracy vs regularization constant on side modules '
                                   'or low-rank adapters. Otherwise, default is same regularization constant '
                                   'as t = 0 modules.'))
    parser.add_argument('--all_layers_efficient',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify to make non-block layers also parameter-efficient.')
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    assert args.loss_samples in {'all', 'nearest neighbors', 'optimal matching'}
    assert args.mode in {'separate', 'side_tune', 'low_rank_adapt'}
    assert args.adjacent_reg_type in {'none', 'l2', 'l2_fisher', 'l2_high', 'l2_very_high', 'l1'}
    if args.mode == 'low_rank_adapt':
        assert args.adapter_mode in {'multiply', 'add'}
        assert args.adapter_rank > 0
    if args.ablate:
        assert args.mode != 'separate'
    assert args.model_type in {'resnet', 'densenet', 'convnet'}
    assert args.num_blocks >= 1
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    torch.cuda.device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    assert(torch.cuda.is_available())
    
    real_world_datasets = {'portraits'}
    if args.dataset not in real_world_datasets:
        num_target_steps = len(args.shift_sequence.split(':'))
        if len(args.target_sample_size_seq) > 0:
            target_sample_sizes = [int(i) for i in args.target_sample_size_seq.split(':')]
            assert len(target_sample_sizes) == num_target_steps
        else:
            target_sample_sizes = [args.target_sample_size for i in range(num_target_steps)]
    else:
        target_sample_sizes = []
    if args.resume:
        resume_n_epochs = (100, 200)
    else:
        resume_n_epochs = (0, 100)
        
    if args.mode == 'side_tune' and len(args.side_layer_sizes) > 0:
        if args.model_type == 'convnet':
            assert args.side_layer_sizes == 'block'
        if args.side_layer_sizes == 'block':
            side_layer_sizes = args.side_layer_sizes
        else:
            side_layer_sizes = [int(size) for size in args.side_layer_sizes.split(',')]
    else:
        side_layer_sizes = []
    run_joint_model_experiment(args.dataset,
                               args.model_type,
                               args.num_blocks,
                               args.shift_sequence,
                               args.source_sample_size,
                               target_sample_sizes,
                               args.target_test_size,
                               seed                   = args.seed,
                               visualize_shifts       = args.visualize_images,
                               separate_layers        = args.separate_layers,
                               resume_n_epochs        = resume_n_epochs,
                               weight_reg_by_time     = args.weight_reg_by_time,
                               loss_samples           = args.loss_samples,
                               loss_weight_increasing = args.loss_weight_increasing,
                               adjacent_reg_type      = args.adjacent_reg_type,
                               start_with_imagenet    = args.start_with_imagenet,
                               mode                   = args.mode,
                               adapter_rank           = args.adapter_rank,
                               adapter_mode           = args.adapter_mode,
                               side_layer_sizes       = side_layer_sizes,
                               ablate                 = args.ablate,
                               sweep_efficient_reg    = args.sweep_efficient_reg,
                               all_layers_efficient   = args.all_layers_efficient)