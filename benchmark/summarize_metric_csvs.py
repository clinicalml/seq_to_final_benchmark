import os
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from os.path import dirname, abspath, join

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'model_classes'))
from model_property_helper_functions import get_model_class_and_properties

def aggregate_csvs():
    '''
    Output a csv containing metrics from all experiments
    '''
    architectures = ['convnet', 'densenet', 'resnet']
    datasets      = ['cifar10', 'cifar100', 'portraits']
    dfs_to_concat = []
    real_world_datasets = {'portraits'}
    for architecture, num_blocks, dataset in product(architectures, range(1, 5), datasets):
        exp_dir = config.output_dir + architecture + '_' + str(num_blocks) + 'blocks_' + dataset + '_experiment/'
        if os.path.exists(exp_dir):
            exp_method_dirs = [exp_dir + method_dir + '/' for method_dir in os.listdir(exp_dir)]
        else:
            exp_method_dirs = []
        for exp_method_dir in exp_method_dirs:
            if 'time0' in exp_method_dir or ('all_prev' in exp_method_dir and 'reg_all_prev' not in exp_method_dir):
                continue
            setup_dirs = os.listdir(exp_method_dir)
            for setup_dir in setup_dirs:
                if dataset in real_world_datasets:
                    if setup_dir != dataset:
                        continue
                else:
                    if not (setup_dir.endswith('4000') or (exp_method_dir.endswith('erm_final/')
                                                           and setup_dir.endswith('20000'))):
                        continue
                exp_method_setup_dir = exp_method_dir + setup_dir + '/'
                seed_dirs = os.listdir(exp_method_setup_dir)
                for seed_dir in seed_dirs:
                    if seed_dir != 'seed1007':
                        continue
                    exp_method_setup_seed_dir = exp_method_setup_dir + seed_dir + '/'
                    metrics_csv = exp_method_setup_seed_dir + 'metrics.csv'
                    if not os.path.exists(metrics_csv):
                        continue
                    metric_df = pd.read_csv(metrics_csv)
                    dfs_to_concat.append(metric_df)
    final_df = pd.concat(dfs_to_concat)
    final_df.to_csv(config.output_dir + 'metrics_summary.csv',
                    index = False)
    
def create_latex_table(tolerance = .02):
    '''
    Output latex metric table for each set-up
    Model & 4k & ERM & IRM & DRO & FT & LP-FT & I-FT & D-FT & ST-1 & ST-B
    Model & SFT & EWC & SST-1 & SST-B & JM & JST-1 & JST-B & & & 20k
    Mean (std) if more than 1 seed. Otherwise, just mean.
    Bold or italicize if within tolerance of best or oracle mean, respectively.
    Add rows for mean and std dev across methods.
    Bold or italicize mean if mean + 1 std dev of difference from best or oracle is above 0, respectively.
    @param tolerance: float, bold/italicize all mean test accuracies that are within this tolerance of best/oracle
    @return: None
    '''
    df = pd.read_csv(config.output_dir + 'metrics_summary.csv')
    setups = list(df['Shift sequence'].unique())
    architectures = sorted(list(df['Model'].unique()))
    abbrevs       = ['Conv-1', 'Conv-2', 'Conv-3', 'Conv-4', 'Dense-1', 'Dense-2', 'Dense-3', 'Dense-4',
                     'Res-1', 'Res-2', 'Res-3', 'Res-4']
    rotation_setups = ['source16000:rotation4000',
                       'source10000:rotation6000:rotation4000',
                       'source6000:rotation4000:rotation6000:rotation4000',
                       'source4000:rotation4000:rotation4000:rotation4000:rotation4000',
                       'source4000:rotation3000:rotation3000:rotation3000:rotation3000:rotation4000',
                       'source4000:rotation2000:rotation3000:rotation2000:rotation3000:rotation2000:rotation4000',
                       'source4000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000']
    rotation_oracle_setups \
        = ['source16000:rotation20000',
           'source10000:rotation6000:rotation20000',
           'source6000:rotation4000:rotation6000:rotation20000',
           'source4000:rotation4000:rotation4000:rotation4000:rotation20000',
           'source4000:rotation3000:rotation3000:rotation3000:rotation3000:rotation20000',
           'source4000:rotation2000:rotation3000:rotation2000:rotation3000:rotation2000:rotation20000',
           'source4000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation20000']
    rotation_sequence_setups_to_omit = set(rotation_setups).union(set(rotation_oracle_setups))
    real_world_datasets = {'portraits'}
    for setup in setups:
        if not (setup.endswith('4000') or setup in real_world_datasets):
            continue
        if setup in rotation_sequence_setups_to_omit:
            continue
        if setup in real_world_datasets:
            setup_df = df.loc[df['Shift sequence'] == setup]
        else:
            setup_20k = setup[:-4] + '20000'
            setup_df = df.loc[df['Shift sequence'].isin({setup, setup_20k})]
        alt_setups = {'source6000:corruption4000:label_flip6000:rotation4000', 
                      'source6000:rotation4000:corruption6000:label_flip4000'}
        if setup == 'source6000:corruption4000:label_flip6000:rotation4000':
            alt_setup_df = df.loc[df['Shift sequence'].isin({'source6000:rotation4000:corruption6000:label_flip4000',
                                                             'source6000:rotation4000:corruption6000:label_flip20000'})]
        elif setup == 'source6000:rotation4000:corruption6000:label_flip4000':
            alt_setup_df = df.loc[df['Shift sequence'].isin({'source6000:corruption4000:label_flip6000:rotation4000',
                                                             'source6000:corruption4000:label_flip6000:rotation20000'})]
        setup_output = 'Model & '
        if setup in real_world_datasets:
            setup_output += 'Final & '
        else:
            setup_output += '4k & '
        setup_output += 'ERM & IRM & DRO & FT & LP-FT & I-FT & D-FT & ST-1 & ST-B \\\\\n\\midrule\n'
        setup_output2 = '\\midrule\nModel & SFT & EWC & SST-1 & SST-B & JM & JST-1 & JST-B & & & '
        if setup not in real_world_datasets:
            setup_output2 += '20k '
        setup_output2 += '\\\\\n\\midrule\n'
        setup_output_with_std  = setup_output
        setup_output2_with_std = setup_output2
        if setup in real_world_datasets:
            methods = ['erm_final']
        else:
            methods = ['erm_final_4000_samples']
        methods += ['erm_all', 'irm_all', 'dro_all', 'erm_fine_tune', 'erm_linear_probe_then_fine_tune', 
                    'irm_fine_tune', 'dro_fine_tune', 'erm_side_tune', 
                    'erm_side_tune_block', 'sequential_fine_tune', 
                    'sequential_fine_tune_reg_all_previous_fisher', 'sequential_side_tune', 
                    'sequential_side_tune_block', 'joint_model_adjacent_reg_l2', 
                    'joint_model_side_tune_adjacent_reg_none', 'joint_model_side_tune_block_adjacent_reg_none']
        if setup not in real_world_datasets:
            methods.append('erm_final_20000_samples')
        other_decay_methods = {'sequential_fine_tune_exponential_lr_decay', 'sequential_fine_tune_linear_lr_decay'}
        if setup in real_world_datasets:
            # real data experiments don't have oracle
            all_methods_except_oracle = set(methods)
        else:
            all_methods_except_oracle = set(methods[:-1])
            oracle_name = methods[-1]
        all_methods_except_oracle.update(other_decay_methods)
        method_mean_accs  = defaultdict(dict)
        for architecture_idx, architecture in enumerate(architectures):
            architecture_df = setup_df.loc[setup_df['Model'] == architecture]
            if setup in alt_setups:
                alt_setup_architecture_df = alt_setup_df.loc[np.logical_and(alt_setup_df['Model'] == architecture,
                                                                            alt_setup_df['Method'].str.startswith('erm_final'))]
                combined_df   = pd.concat((architecture_df, alt_setup_architecture_df))
                not_oracle_df = combined_df.loc[combined_df['Method'].isin(all_methods_except_oracle)]
                not_oracle_df['Method'] = np.where(not_oracle_df['Method'].isin(other_decay_methods),
                                                   'sequential_fine_tune',
                                                   not_oracle_df['Method'])
                not_oracle_df = not_oracle_df.groupby(by = ['Method', 'Seed']).max().reset_index()
                if len(not_oracle_df) >= 1:
                    best_mean = np.max(not_oracle_df[['Method','Test accuracy']].groupby(by = 'Method').mean()['Test accuracy'].values)
                else:
                    best_mean = 1.1 # placeholder since no results
                oracle_df     = combined_df.loc[combined_df['Method'] == oracle_name]
                oracle_df.drop_duplicates(subset = 'Seed', inplace = True)
                if len(oracle_df) >= 1:
                    oracle_mean = oracle_df['Test accuracy'].mean()
                else:
                    oracle_mean = 1.1 # because oracle hasn't been run yet, cannot say if other methods are as good
            else:
                not_oracle_df = architecture_df.loc[architecture_df['Method'].isin(all_methods_except_oracle)]
                if len(not_oracle_df) >= 1:
                    best_mean = np.max(not_oracle_df[['Method', 'Test accuracy']].groupby(by = 'Method').mean()['Test accuracy'].values)
                else:
                    best_mean = 1.1 # placeholder since no results
                if setup in real_world_datasets:
                    oracle_mean = 1.1 # no oracle for real data experiments
                else:
                    oracle_df     = architecture_df.loc[architecture_df['Method'] == oracle_name]
                    if len(oracle_df) >= 1:
                        oracle_mean = oracle_df['Test accuracy'].mean()
                    else:
                        oracle_mean = 1.1 # because oracle hasn't been run yet, cannot say if other methods are as good
            architecture_header = abbrevs[architecture_idx] + ' & '
            setup_output       += architecture_header
            setup_output2      += architecture_header
            setup_output_with_std  += architecture_header
            setup_output2_with_std += architecture_header
            for method_idx, method in enumerate(methods):
                if architecture.startswith('convnet'):
                    if method == 'erm_side_tune':
                        method = 'erm_side_tune_block'
                    elif method == 'erm_side_tune_block':
                        setup_output += '--- & '
                        setup_output_with_std += '--- & '
                        continue
                    elif method == 'sequential_side_tune':
                        method = 'sequential_side_tune_block'
                    elif method == 'sequential_side_tune_block':
                        setup_output2 += '--- & '
                        setup_output2_with_std += '--- & '
                        continue
                    elif method == 'joint_model_side_tune_adjacent_reg_none':
                        method = 'joint_model_side_tune_block_adjacent_reg_none'
                    elif method == 'joint_model_side_tune_block_adjacent_reg_none':
                        setup_output2 += '--- & '
                        setup_output2_with_std += '--- & '
                        continue
                if method == 'sequential_fine_tune':
                    sequential_ft_methods = {method}
                    sequential_ft_methods.update(other_decay_methods)
                    method_df = architecture_df.loc[architecture_df['Method'].isin(sequential_ft_methods)]
                    method_df = method_df.groupby(by = 'Seed').max()
                elif method.startswith('erm_final') and setup in alt_setups:
                    method_df = pd.concat((architecture_df.loc[architecture_df['Method'] == method],
                                           alt_setup_architecture_df.loc[alt_setup_architecture_df['Method'] == method]))
                    method_df.drop_duplicates(subset = 'Seed',
                                              inplace = True)
                else:
                    method_df = architecture_df.loc[architecture_df['Method'] == method]
                if len(method_df) == 0:
                    method_output = '& '
                    method_output_with_std = '& '
                else:
                    if method == 'erm_final_20000_samples':
                        method_output = '& & '
                    else:
                        method_output = ''
                    mean_acc = method_df['Test accuracy'].mean()
                    method_mean_accs[method][architecture] = mean_acc
                    if method == oracle_name:
                        method_output += '{:.2f}'.format(mean_acc).lstrip('0')
                    elif mean_acc >= oracle_mean - tolerance:
                        method_output += '\\textit{\\textbf{' + '{:.2f}'.format(mean_acc).lstrip('0') + '}}'
                    elif best_mean - mean_acc <= tolerance:
                        method_output += '\\textbf{' + '{:.2f}'.format(mean_acc).lstrip('0') + '}'
                    else:
                        method_output += '{:.2f}'.format(mean_acc).lstrip('0')
                    method_output_with_std = method_output
                    if len(method_df) > 1:
                        std_acc  = method_df['Test accuracy'].std()
                        method_output_with_std += ' (' + '{:.2f}'.format(std_acc).lstrip('0') + ')'
                    method_output += ' & '
                    method_output_with_std += ' & '
                if method_idx < 10:
                    setup_output  += method_output
                    setup_output_with_std  += method_output_with_std
                else:
                    setup_output2 += method_output
                    setup_output2_with_std += method_output_with_std
            setup_output  = setup_output[:-2]  + '\\\\\n'
            setup_output2 = setup_output2[:-2] + '\\\\\n'
            setup_output_with_std  = setup_output_with_std[:-2]  + '\\\\\n'
            setup_output2_with_std = setup_output2_with_std[:-2] + '\\\\\n'
            if architecture_idx in {3, 7}:
                setup_output  += '\\midrule\n'
                setup_output2 += '\\midrule\n'
                setup_output_with_std  += '\\midrule\n'
                setup_output2_with_std += '\\midrule\n'
        # Add row for mean accuracies with bold/italic based on mean diff from best/oracle being less than 1 std dev below 0
        best_method_mean_acc = -1
        for method in method_mean_accs:
            if setup not in real_world_datasets and method == oracle_name:
                continue
            method_mean = np.mean(np.array([method_mean_accs[method][architecture]
                                            for architecture in method_mean_accs[method]]))
            if method_mean > best_method_mean_acc:
                best_method_mean_acc = method_mean
                best_method   = method
        mean_output  = '\\midrule\nMean '
        std_output   = 'Std dev '
        mean_output2 = '\\midrule\nMean '
        std_output2  = 'Std dev '
        for method_idx, method in enumerate(methods):
            if method == oracle_name:
                if method_idx < 10:
                    mean_output += '& & & '
                    std_output  += '& & & '
                else:
                    mean_output2 += '& & & '
                    std_output2  += '& & & '
            else:
                if method_idx < 10:
                    mean_output += '& '
                    std_output  += '& '
                else:
                    mean_output2 += '& '
                    std_output2  += '& '
            if method not in method_mean_accs:
                continue
            method_mean = np.mean(np.array([method_mean_accs[method][architecture]
                                            for architecture in method_mean_accs[method]]))
            method_equiv_to_best = False
            if method == best_method:
                method_equiv_to_best = True
            else:
                method_best_shared_architectures = set(method_mean_accs[best_method].keys()).intersection(set(method_mean_accs[method].keys()))
                if len(method_best_shared_architectures) > 0:
                    method_diffs_from_best = np.array([method_mean_accs[best_method][architecture] - method_mean_accs[method][architecture]
                                                       for architecture in method_best_shared_architectures])
                    method_mean_diff_from_best = np.mean(method_diffs_from_best)
                    method_std_diff_from_best  = np.std(method_diffs_from_best)
                    if method_mean_diff_from_best - method_std_diff_from_best < 0:
                        method_equiv_to_best = True
            method_equiv_to_oracle = False
            if setup not in real_world_datasets:
                method_oracle_shared_architectures = set(method_mean_accs[oracle_name].keys()).intersection(set(method_mean_accs[method].keys()))
                if len(method_oracle_shared_architectures) > 0:
                    method_diffs_from_oracle = np.array([method_mean_accs[oracle_name][architecture] - method_mean_accs[method][architecture]
                                                         for architecture in method_oracle_shared_architectures])
                    method_mean_diff_from_oracle = np.mean(method_diffs_from_oracle)
                    method_std_diff_from_oracle  = np.std(method_diffs_from_oracle)
                    if method_mean_diff_from_oracle - method_std_diff_from_oracle < 0:
                        method_equiv_to_oracle = True
            if setup not in real_world_datasets and method == oracle_name:
                method_mean_output = '{:.2f}'.format(method_mean).lstrip('0') + ' '
            elif method_equiv_to_oracle:
                method_mean_output = '\\textit{\\textbf{' + '{:.2f}'.format(method_mean).lstrip('0') + '}} '
            elif method_equiv_to_best:
                method_mean_output = '\\textbf{' + '{:.2f}'.format(method_mean).lstrip('0') + '} '
            else:
                method_mean_output = '{:.2f}'.format(method_mean).lstrip('0') + ' '
            method_std     = np.std(np.array([method_mean_accs[method][architecture]
                                              for architecture in method_mean_accs[method]]))
            method_std_str = '{:.2f}'.format(method_std).lstrip('0') + ' '
            if method_idx < 10:
                mean_output += method_mean_output
                std_output  += method_std_str
            else:
                mean_output2 += method_mean_output
                std_output2  += method_std_str
        mean_output  += '\\\\\n'
        mean_output2 += '\\\\\n'
        std_output   += '\\\\\n'
        std_output2  += '\\\\\n'
        with open(config.output_dir + setup + '_metrics_latex_table.txt', 'w') as f:
            f.write(setup_output + mean_output + std_output + setup_output2 + mean_output2 + std_output2)
        with open(config.output_dir + setup + '_metrics_with_std_latex_table.txt', 'w') as f:
            f.write(setup_output_with_std + setup_output2_with_std)
            
def create_rotation_sequence_latex_table(tolerance = .02):
    '''
    Output latex metric table for each set-up
    # steps & 4k & ERM & IRM & DRO & FT & LP-FT & I-FT & D-FT & ST-1 & ST-B
    # steps & SFT & EWC & SST-1 & SST-B & JM & JST-1 & JST-B & & & 20k
    Mean (std) if more than 1 seed. Otherwise, just mean
    @param tolerance: float, bold/italicize all mean test accuracies that are within this tolerance of best/oracle
    @return: None
    '''
    rotation_setups = ['source16000:rotation4000',
                       'source10000:rotation6000:rotation4000',
                       'source6000:rotation4000:rotation6000:rotation4000',
                       'source4000:rotation4000:rotation4000:rotation4000:rotation4000',
                       'source4000:rotation3000:rotation3000:rotation3000:rotation3000:rotation4000',
                       'source4000:rotation2000:rotation3000:rotation2000:rotation3000:rotation2000:rotation4000',
                       'source4000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation4000']
    rotation_oracle_setups \
        = ['source16000:rotation20000',
           'source10000:rotation6000:rotation20000',
           'source6000:rotation4000:rotation6000:rotation20000',
           'source4000:rotation4000:rotation4000:rotation4000:rotation20000',
           'source4000:rotation3000:rotation3000:rotation3000:rotation3000:rotation20000',
           'source4000:rotation2000:rotation3000:rotation2000:rotation3000:rotation2000:rotation20000',
           'source4000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation2000:rotation20000']
    rotation_setup_set = set(rotation_setups).union(set(rotation_oracle_setups))
    
    methods = ['erm_final_4000_samples', 'erm_all', 'irm_all', 'dro_all', 'erm_fine_tune', 'erm_linear_probe_then_fine_tune',
               'irm_fine_tune', 'dro_fine_tune', 'erm_side_tune', 
               'erm_side_tune_block', 'sequential_fine_tune', 'sequential_fine_tune_reg_all_previous_fisher', 
               'sequential_side_tune', 'sequential_side_tune_block', 'joint_model_adjacent_reg_l2', 
               'joint_model_side_tune_adjacent_reg_none', 'joint_model_side_tune_block_adjacent_reg_none',
               'erm_final_20000_samples']
    other_decay_methods = {'sequential_fine_tune_exponential_lr_decay', 'sequential_fine_tune_linear_lr_decay'}
    all_methods_except_oracle = set(methods[:-1])
    all_methods_except_oracle.update(other_decay_methods)
    oracle_name = methods[-1]
        
    df = pd.read_csv(config.output_dir + 'metrics_summary.csv')
    rotation_seq_df = df.loc[df['Shift sequence'].isin(rotation_setup_set)]
    architectures = sorted(list(rotation_seq_df['Model'].unique()))
    for architecture in architectures:
        architecture_df = rotation_seq_df.loc[rotation_seq_df['Model'] == architecture]
        
        setup_output  = '\\# steps & 4k & ERM & IRM & DRO & FT & LP-FT & I-FT & D-FT & ST-1 & ST-B \\\\\n\\midrule\n'
        setup_output2 = '\\midrule\n\\# steps & SFT & EWC & SST-1 & SST-B & JM & JST-1 & JST-B & & & 20k \\\\\n\\midrule\n'
        setup_output_with_std  = setup_output
        setup_output2_with_std = setup_output2
        method_mean_accs = defaultdict(dict)
        
        for seq_idx in range(len(rotation_setups)):
            not_oracle_df = architecture_df.loc[architecture_df['Shift sequence'] == rotation_setups[seq_idx]]
            best_mean     = np.max(not_oracle_df[['Method', 'Test accuracy']].groupby(by = 'Method').mean()['Test accuracy'].values)
            oracle_df     = architecture_df.loc[architecture_df['Shift sequence'] == rotation_oracle_setups[seq_idx]]
            if len(oracle_df) >= 1:
                oracle_mean = oracle_df['Test accuracy'].mean()
            else:
                oracle_mean = 1.1 # because oracle hasn't been run yet, cannot say if other methods are as good
            
            seq_header              = str(seq_idx + 2) + ' & '
            setup_output           += seq_header
            setup_output2          += seq_header
            setup_output_with_std  += seq_header
            setup_output2_with_std += seq_header
            
            for method_idx, method in enumerate(methods):
                if architecture.startswith('convnet'):
                    if method == 'erm_side_tune':
                        method = 'erm_side_tune_block'
                    elif method == 'erm_side_tune_block':
                        setup_output += '--- & '
                        setup_output_with_std += '--- & '
                        continue
                    elif method == 'sequential_side_tune':
                        method = 'sequential_side_tune_block'
                    elif method == 'sequential_side_tune_block':
                        setup_output2 += '--- & '
                        setup_output2_with_std += '--- & '
                        continue
                    elif method == 'joint_model_side_tune_adjacent_reg_none':
                        method = 'joint_model_side_tune_block_adjacent_reg_none'
                    elif method == 'joint_model_side_tune_block_adjacent_reg_none':
                        setup_output2 += '--- & '
                        setup_output2_with_std += '--- & '
                        continue
                if method == 'sequential_fine_tune':
                    sequential_ft_methods = {method}
                    sequential_ft_methods.update(other_decay_methods)
                    method_df = not_oracle_df.loc[not_oracle_df['Method'].isin(sequential_ft_methods)]
                    method_df = method_df.groupby(by = 'Seed').max()
                elif method == oracle_name:
                    method_df = oracle_df.loc[oracle_df['Method'] == method]
                else:
                    method_df = not_oracle_df.loc[not_oracle_df['Method'] == method]
                if len(method_df) == 0:
                    method_output = '& '
                    method_output_with_std = '& '
                else:
                    if method == oracle_name:
                        method_output = '& & '
                    else:
                        method_output = ''
                    mean_acc = method_df['Test accuracy'].mean()
                    method_mean_accs[method][seq_idx + 2] = mean_acc
                    if method == oracle_name:
                        method_output += '{:.2f}'.format(mean_acc).lstrip('0')
                    elif mean_acc >= oracle_mean - tolerance:
                        method_output += '\\textit{\\textbf{' + '{:.2f}'.format(mean_acc).lstrip('0') + '}}'
                    elif best_mean - mean_acc <= tolerance:
                        method_output += '\\textbf{' + '{:.2f}'.format(mean_acc).lstrip('0') + '}'
                    else:
                        method_output += '{:.2f}'.format(mean_acc).lstrip('0')
                    method_output_with_std = method_output
                    if len(method_df) > 1:
                        std_acc  = method_df['Test accuracy'].std()
                        method_output_with_std += ' (' + '{:.2f}'.format(std_acc).lstrip('0') + ')'
                    method_output += ' & '
                    method_output_with_std += ' & '
                if method_idx < 10:
                    setup_output  += method_output
                    setup_output_with_std  += method_output_with_std
                else:
                    setup_output2 += method_output
                    setup_output2_with_std += method_output_with_std
            setup_output  = setup_output[:-2]  + '\\\\\n'
            setup_output2 = setup_output2[:-2] + '\\\\\n'
            setup_output_with_std  = setup_output_with_std[:-2]  + '\\\\\n'
            setup_output2_with_std = setup_output2_with_std[:-2] + '\\\\\n'
        # Add row for mean accuracies with bold/italic based on mean diff from best/oracle being less than 1 std dev below 0
        best_method_mean_acc = -1
        for method in method_mean_accs:
            if method == oracle_name:
                continue
            method_mean = np.mean(np.array([method_mean_accs[method][seq_len]
                                            for seq_len in method_mean_accs[method]]))
            if method_mean > best_method_mean_acc:
                best_method_mean_acc = method_mean
                best_method = method
        mean_output  = '\\midrule\nMean '
        std_output   = 'Std dev '
        mean_output2 = '\\midrule\nMean '
        std_output2  = 'Std dev '
        for method_idx, method in enumerate(methods):
            if method == oracle_name:
                mean_output2 += '& & & '
                std_output2  += '& & & '
            else:
                if method_idx < 10:
                    mean_output += '& '
                    std_output  += '& '
                else:
                    mean_output2 += '& '
                    std_output2  += '& '
            if method not in method_mean_accs:
                continue
            method_mean = np.mean(np.array([method_mean_accs[method][seq_len]
                                            for seq_len in method_mean_accs[method]]))
            method_equiv_to_best = False
            if method == best_method:
                method_equiv_to_best = True
            else:
                method_best_shared_seq_len = set(method_mean_accs[best_method].keys()).intersection(set(method_mean_accs[method].keys()))
                if len(method_best_shared_seq_len) > 0:
                    method_diffs_from_best = np.array([method_mean_accs[best_method][seq_len] - method_mean_accs[method][seq_len]
                                                       for seq_len in method_best_shared_seq_len])
                    method_mean_diff_from_best = np.mean(method_diffs_from_best)
                    method_std_diff_from_best  = np.std(method_diffs_from_best)
                    if method_mean_diff_from_best - method_std_diff_from_best < 0:
                        method_equiv_to_best = True
            method_equiv_to_oracle = False
            method_oracle_shared_seq_len = set(method_mean_accs[oracle_name].keys()).intersection(set(method_mean_accs[method].keys()))
            if len(method_oracle_shared_seq_len) > 0:
                method_diffs_from_oracle = np.array([method_mean_accs[oracle_name][seq_len] - method_mean_accs[method][seq_len]
                                                     for seq_len in method_oracle_shared_seq_len])
                method_mean_diff_from_oracle = np.mean(method_diffs_from_oracle)
                method_std_diff_from_oracle  = np.std(method_diffs_from_oracle)
                if method_mean_diff_from_oracle - method_std_diff_from_oracle < 0:
                    method_equiv_to_oracle = True
            if method == oracle_name:
                method_mean_output = '{:.2f}'.format(method_mean).lstrip('0') + ' '
            elif method_equiv_to_oracle:
                method_mean_output = '\\textit{\\textbf{' + '{:.2f}'.format(method_mean).lstrip('0') + '}} '
            elif method_equiv_to_best:
                method_mean_output = '\\textbf{' + '{:.2f}'.format(method_mean).lstrip('0') + '} '
            else:
                method_mean_output = '{:.2f}'.format(method_mean).lstrip('0') + ' '
            method_std     = np.std(np.array([method_mean_accs[method][seq_len]
                                              for seq_len in method_mean_accs[method]]))
            method_std_str = '{:.2f}'.format(method_std).lstrip('0') + ' '
            if method_idx < 10:
                mean_output += method_mean_output
                std_output  += method_std_str
            else:
                mean_output2 += method_mean_output
                std_output2  += method_std_str
        mean_output  += '\\\\\n'
        mean_output2 += '\\\\\n'
        std_output   += '\\\\\n'
        std_output2  += '\\\\\n'
        file_header = config.output_dir + 'rotation_seq_' + architecture.replace(' ', '_')
        with open(file_header + '_metrics_latex_table.txt', 'w') as f:
            f.write(setup_output + mean_output + std_output + setup_output2 + mean_output2 + std_output2)
        with open(file_header + '_metrics_with_std_latex_table.txt', 'w') as f:
            f.write(setup_output_with_std + setup_output2_with_std)
    
if __name__ == '__main__':
    
    aggregate_csvs()
    create_latex_table()
    create_rotation_sequence_latex_table()