import os
import sys
import json
import glob
import math
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from os.path import dirname, abspath, join

sys.path.append(join(dirname(dirname(abspath(__file__))), 'model_classes'))
from joint_resnet_with_dropout import joint_resnet_with_dropout

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from pytorch_model_utils import load_state_dict_from_gz

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def plot_valid_acc_vs_adjacent_reg(dataset_name,
                                   shift_sequence_str,
                                   seed,
                                   include_very_high = False):
    '''
    Plot best validation accuracy from any combination of the other hyperparameters
    again the 5 adjacent regularization parameters that are tried for the joint model with all separate modules
    @param dataset_name: str, cifar10 or cifar100
    @param shift_sequence_str: str, name of shift
    @param seed: int, seed that was used for experiment
    @param include_very_high: bool, whether very high adjacent reg setting was run
    @return: None
    '''
    experiment_dir   = config.output_dir \
                     + 'resnet_4blocks_{dataset_name}_experiment/joint_model_adjacent_reg_{reg_type}/{shift_sequence_str}/seed{seed}/models/'
    
    reg_types        = ['none', 'l2', 'l2_high']
    if include_very_high:
        reg_types.append('l2_very_high')
    reg_dirs         = {reg_type: experiment_dir.format(dataset_name       = dataset_name,
                                                        reg_type           = reg_type,
                                                        shift_sequence_str = shift_sequence_str,
                                                        seed               = seed)
                        for reg_type in reg_types}
    for reg_type in reg_dirs:
        assert os.path.exists(reg_dirs[reg_type])
    
    final_step_idx          = len(shift_sequence_str.split(':')) - 1
    final_step_val_acc_name = 'valid_step{step_idx}_accuracies'.format(step_idx = final_step_idx)
    
    no_reg_best_hyperparams_json = reg_dirs['none'] + 'joint_model_all_separate_best_hyperparams.json'
    with open(no_reg_best_hyperparams_json, 'r') as f:
        no_reg_best_hyperparams = json.load(f)
    val_json_name = ('joint_model_all_separate_lr{lr}_l2_reg{weight_decay}_l2_adjreg{adjacent_reg}_dropout{dropout}'
                     '_finalwt{final_wt}_{n_epochs}epochs_losses.json')
    no_reg_best_hyperparams_val_json \
        = reg_dirs['none'] + val_json_name.format(lr           = no_reg_best_hyperparams['learning_rate'],
                                                  weight_decay = no_reg_best_hyperparams['weight_decay'],
                                                  adjacent_reg = 0,
                                                  dropout      = no_reg_best_hyperparams['dropout'],
                                                  final_wt     = no_reg_best_hyperparams['last_time_step_loss_wt'],
                                                  n_epochs     = no_reg_best_hyperparams['early_stopping_epochs'])
    
    with open(no_reg_best_hyperparams_val_json, 'r') as f:
        no_reg_best_hyperparams_val_acc = max(json.load(f)[final_step_val_acc_name])
    
    val_accs_to_plot = [no_reg_best_hyperparams_val_acc]
    
    reg_levels     = {'l2'          : [0.01, 0.1, 1], 
                      'l2_high'     : [10, 100, 1000],
                      'l2_very_high': [1e4, 1e5, 1e6]}
    learning_rates = [0.0001, 0.001]
    weight_decays  = [0.01, 0.0001]
    dropouts       = [0, 0.5]
    final_wts      = [1, 3]
    
    for reg_type in reg_types[1:]:
        for reg_level in reg_levels[reg_type]:
            best_val_acc = -1
            for learning_rate, weight_decay, dropout, final_wt in product(learning_rates, weight_decays, dropouts, final_wts):
                reg_val_json_search \
                    = reg_dirs[reg_type] + val_json_name.format(lr           = learning_rate,
                                                                weight_decay = weight_decay,
                                                                adjacent_reg = reg_level,
                                                                dropout      = dropout,
                                                                final_wt     = final_wt,
                                                                n_epochs     = '*')
                reg_val_json = glob.glob(reg_val_json_search)[0]
                with open(reg_val_json, 'r') as f:
                    reg_val_acc = max(json.load(f)[final_step_val_acc_name])
                if reg_val_acc > best_val_acc:
                    best_val_acc = reg_val_acc
            val_accs_to_plot.append(best_val_acc)
    
    erm_experiment_dir = config.output_dir + 'resnet_4blocks_{dataset_name}_experiment/{erm_alg}/{shift_sequence_str}/seed{seed}/models/'
    erm_all_dir        = erm_experiment_dir.format(dataset_name       = dataset_name,
                                                   erm_alg            = 'erm_all',
                                                   shift_sequence_str = shift_sequence_str,
                                                   seed               = seed)
    erm_final_dir      = erm_experiment_dir.format(dataset_name       = dataset_name,
                                                   erm_alg            = 'erm_final',
                                                   shift_sequence_str = shift_sequence_str,
                                                   seed               = seed)
    
    erm_all_best_hyperparams_json = erm_all_dir + 'erm_all_all_best_hyperparams.json'
    with open(erm_all_best_hyperparams_json, 'r') as f:
        erm_all_best_hyperparams = json.load(f)
    
    erm_val_json_name = '{erm_alg}_all_lr{lr}_standard_reg{weight_decay}_dropout{dropout}_{n_epochs}epochs_losses.json'
    
    erm_all_best_hyperparams_val_json \
        = erm_all_dir + erm_val_json_name.format(erm_alg      = 'erm_all',
                                                 lr           = erm_all_best_hyperparams['learning rate'],
                                                 weight_decay = erm_all_best_hyperparams['weight decay'],
                                                 dropout      = erm_all_best_hyperparams['dropout'],
                                                 n_epochs     = erm_all_best_hyperparams['early stopping epochs'])
    
    with open(erm_all_best_hyperparams_val_json, 'r') as f:
        erm_all_val_acc   = max(json.load(f)['valid_accuracies'])
        
    erm_final_best_hyperparams_json = erm_final_dir + 'erm_final_all_best_hyperparams.json'
    with open(erm_final_best_hyperparams_json, 'r') as f:
        erm_final_best_hyperparams = json.load(f)
    
    erm_final_best_hyperparams_val_json \
        = erm_final_dir + erm_val_json_name.format(erm_alg      = 'erm_final',
                                                   lr           = erm_final_best_hyperparams['learning rate'],
                                                   weight_decay = erm_final_best_hyperparams['weight decay'],
                                                   dropout      = erm_final_best_hyperparams['dropout'],
                                                   n_epochs     = erm_final_best_hyperparams['early stopping epochs'])
    
    with open(erm_final_best_hyperparams_val_json, 'r') as f:
        erm_final_val_acc = max(json.load(f)['valid_accuracies'])
    
    if include_very_high:
        max_reg_power = 7
    else:
        max_reg_power = 4
    adjacent_regs = [math.pow(10, i) for i in range(-3, max_reg_power)] # 0.001 is placeholder for 0 to plot on log scale
    df = pd.DataFrame(data    = {'Adjacent regularization' : adjacent_regs,
                                 'Best validation accuracy': val_accs_to_plot,
                                 'Model'                   : ['Joint model' for _ in range(len(adjacent_regs))] },
                      columns = ['Adjacent regularization', 'Best validation accuracy', 'Model'])
    fig, ax = plt.subplots()
    sns.lineplot(data      = df,
                 x         = 'Adjacent regularization',
                 y         = 'Best validation accuracy',
                 hue       = 'Model',
                 hue_order = ['Joint model', 'ERM final', 'ERM all'],
                 ax        = ax,
                 legend    = None)
    
    erm_df = pd.DataFrame(data    = {'Adjacent regularization' : [adjacent_regs[0], adjacent_regs[-1]],
                                     'Best validation accuracy': [erm_final_val_acc, erm_all_val_acc],
                                     'Model'                   : ['ERM final', 'ERM all']},
                          columns = ['Adjacent regularization', 'Best validation accuracy', 'Model'])
    sns.scatterplot(data      = erm_df,
                    x         = 'Adjacent regularization',
                    y         = 'Best validation accuracy',
                    hue       = 'Model',
                    hue_order = ['Joint model', 'ERM final', 'ERM all'],
                    ax        = ax)
    
    ax.set_xscale('log')
    ax.set_title('Joint model with separate modules')
    plot_filename = reg_dirs['l2'] + 'joint_model_adjacent_reg_v_val_acc_' + shift_sequence_str + '.pdf'
    fig.savefig(plot_filename)
    
def plot_param_change_vs_adjacent_reg(dataset_name,
                                      shift_sequence_str,
                                      seed,
                                      include_very_high = False):
    '''
    Plot norm of difference between parameters at adjacent time steps for best hyperparameter setting
    in 4 adjacent regularization set-ups
    @param dataset_name: str, cifar10, cifar100, imagenet
    @param shift_sequence_str: str, name of shift
    @param seed: int, seed that was used for experiment
    @param include_very_high: bool, whether very high adjacent reg setting was run
    @return: None
    '''
    experiment_dir   = config.output_dir \
                     + 'resnet_4blocks_{dataset_name}_experiment/joint_model_adjacent_reg_{reg_type}/{shift_sequence_str}/seed{seed}/models/'
    num_time_steps   = len(shift_sequence_str.split(':'))
    
    reg_types        = ['none', 'l2', 'l2_high']
    if include_very_high:
        reg_types.append('l2_very_high')
    reg_dirs         = {reg_type: experiment_dir.format(dataset_name       = dataset_name,
                                                        reg_type           = reg_type,
                                                        shift_sequence_str = shift_sequence_str,
                                                        seed               = seed)
                        for reg_type in reg_types}
    reg_levels       = []
    layers           = ['Input convolution', 'Residual block 0', 'Residual block 1', 
                        'Residual block 2', 'Residual block 3', 'Output fully connected']
    separate_layers  = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    separate_layers  = [separate_layers for _ in range(num_time_steps - 1)]
    adjacent_norms   = []
    for reg_type in reg_dirs:
        assert os.path.exists(reg_dirs[reg_type])
        
        with open(reg_dirs[reg_type] + 'joint_model_all_separate_best_hyperparams.json', 'r') as f:
            reg_level = json.load(f)['adjacent_reg']
            if reg_level == 0:
                reg_levels.append(0.0001)
            else:
                reg_levels.append(reg_level)
        
        joint_model = joint_resnet_with_dropout(num_time_steps  = num_time_steps,
                                                separate_layers = separate_layers)
        
        joint_model = load_state_dict_from_gz(joint_model,
                                              reg_dirs[reg_type] + 'joint_model_all_separate_best_model.pt.gz')
        
        for layer_idx in range(len(layers)):
            with torch.no_grad():
                layer_weight_diff = joint_model.layers[layer_idx].compute_adjacent_param_norm_at_each_time_step().detach()
                if torch.cuda.is_available():
                    layer_weight_diff = layer_weight_diff.cpu()
            adjacent_norms.append(np.sqrt(np.sum(layer_weight_diff.numpy())))
    
    adjacent_norm_df = pd.DataFrame(data    = {'Adjacent regularization'     : np.repeat(reg_levels, len(layers)),
                                               'Layer'                       : np.tile(layers, len(reg_levels)),
                                               'L2 norm of parameter changes': adjacent_norms},
                                    columns = ['Adjacent regularization', 'Layer', 'L2 norm of parameter changes'])
    
    fig, ax = plt.subplots()
    sns.lineplot(data      = adjacent_norm_df,
                 x         = 'Adjacent regularization',
                 y         = 'L2 norm of parameter changes',
                 hue       = 'Layer',
                 hue_order = layers,
                 ax        = ax)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Joint model with separate modules')
    plot_filename = reg_dirs['l2'] + 'joint_model_adjacent_reg_v_norm_diff_' + shift_sequence_str + '.pdf'
    fig.savefig(plot_filename)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = ('Plot best validation accuracy or norm of parameter changes '
                                                    'vs adjacent regularization.'))
    parser.add_argument('--dataset',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify cifar10 or cifar100.')
    parser.add_argument('--shift_sequence_str',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify name of directory for shift sequence.')
    parser.add_argument('--seed',
                        action  = 'store',
                        type    = int,
                        help    = 'Specify seed used to run original experiments.')
    parser.add_argument('--plot',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify val_acc or param_norm for what to plot.')
    parser.add_argument('--gpu_num',
                        action  = 'store',
                        type    = int,
                        default = 1,
                        help    = 'Specify which GPUs to use.')
    parser.add_argument('--include_very_high',
                        action  = 'store_true',
                        default = False,
                        help    = 'Specify whether very high adjacent regularization setting should be included in plot.')
    
    args   = parser.parse_args()
    assert args.plot in {'val_acc', 'param_norm'}
    if args.plot == 'val_acc':
        plot_valid_acc_vs_adjacent_reg(args.dataset,
                                       args.shift_sequence_str,
                                       args.seed,
                                       args.include_very_high)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
        torch.cuda.device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
        assert(torch.cuda.is_available())
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
        plot_param_change_vs_adjacent_reg(args.dataset,
                                          args.shift_sequence_str,
                                          args.seed,
                                          args.include_very_high)