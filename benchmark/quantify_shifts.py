import os
import sys
import json
import time
import argparse
from datetime import datetime
from os.path import dirname, abspath, join

import ot
import torch
import numpy as np
from sklearn.decomposition import PCA

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'data_loaders'))
from load_image_data_with_shifts import load_datasets_over_time
from load_yearbook_data import load_yearbook_data

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

def run_pca_on_all_train_images(data_loaders_over_time,
                                logger,
                                variance_to_explain,
                                max_pca_components):
    '''
    Run PCA on train images from all time steps
    Also apply the same PCA decomposition to test images from the final time step
    @param data_loaders_over_time: list over time of dicts mapping str data split to torch data loader
    @param logger: logger, for INFO messages
    @param variance_to_explain: float, keep top components explaining this proportion of the variance
    @param max_pca_components: int, maximum number of PCA components to keep
    @return: 1. list of numpy arrays, top PCA components of each training sample at each time point
             2. numpy array, top PCA components of final time step test samples
    '''
    X = torch.vstack([data_loaders['train'].dataset.tensors[0] for data_loaders in data_loaders_over_time])
    X = X.reshape((X.shape[0], -1))
    num_samples = [data_loaders['train'].dataset.tensors[0].shape[0] for data_loaders in data_loaders_over_time]
    if torch.cuda.is_available():
        X = X.cpu()
    X = X.numpy()
    pca = PCA(n_components = max_pca_components)
    X_comps = pca.fit_transform(X)
    explained_variances = pca.explained_variance_ratio_
    explained_variances_cum = np.cumsum(explained_variances)
    num_components_needed = np.searchsorted(explained_variances_cum, variance_to_explain) + 1
    if num_components_needed >= max_pca_components:
        num_components_needed = max_pca_components
        logger.info('Including all ' + str(max_pca_components) + ' PCA components to explain ' 
                    + str(100*explained_variances_cum[-1]) + '% of the variance')
    else:
        logger.info('Including ' + str(num_components_needed) + ' PCA components to explain '
                    + str(100*explained_variances_cum[num_components_needed - 1]) + '% of the variance')
        X_comps = X_comps[:,:num_components_needed]
    X_comps_per_time = []
    time_start_idx   = 0
    for time_idx in range(len(data_loaders_over_time)):
        X_comps_per_time.append(X_comps[time_start_idx:time_start_idx + num_samples[time_idx]])
        time_start_idx += num_samples[time_idx]
    
    final_test_X = data_loaders_over_time[-1]['test'].dataset.tensors[0]
    final_test_X = final_test_X.reshape((final_test_X.shape[0], -1))
    if torch.cuda.is_available():
        final_test_X = final_test_X.cpu()
    final_test_X = final_test_X.numpy()
    final_test_X_comps = pca.transform(final_test_X)
    return X_comps_per_time, final_test_X_comps[:,:num_components_needed]

def quantify_shift_between_distributions(X1_comps,
                                         Y1,
                                         X2_comps,
                                         Y2,
                                         logger):
    '''
    Compute Wasserstein-2 distance between X1_comps and X2_comps for covariate shift
    Compute Wasserstein-2 distance between X1_comps and X2_comps for each label class and take weighted average by P(Y2)
    for conditional shift
    @param X1_comps: np array, images from time 1
    @param Y1: np array, labels from time 1
    @param X2_comps: np array, images from time 2
    @param Y2: np array, labels from time 2
    @param logger: logger, for INFO messages
    @return: 1. float, Wasserstein-2 distance for covariate shift
             2. float, Wasserstein-2 distance for conditional shift
    '''
    def wasserstein_dist(X1,
                         X2):
        '''
        Compute Wasserstein-2 distance between two sets of samples
        @param X1: np array, # samples x # dims
        @param X2: np array, # samples x # dims
        @return: float, Wasserstein-2 distance
        '''
        start_time = time.time()
        assert X1.shape[1] == X2.shape[1]
        weights1 = np.ones(X1.shape[0])/float(X1.shape[0])
        weights2 = np.ones(X2.shape[0])/float(X2.shape[0])
        wasserstein_dist = np.sqrt(ot.emd2(weights1,
                                           weights2,
                                           ot.dist(X1, X2)))
        logger.info('Time to compute Wasserstein distance: ' + str(time.time() - start_time) + ' seconds')
        return wasserstein_dist
    
    covariate_shift_W2 = wasserstein_dist(X1_comps, X2_comps)
    logger.info('Wasserstein-2 distance of covariate shift: ' + str(covariate_shift_W2))
    
    conditional_shift_W2 = 0.
    Y                    = np.concatenate((Y1, Y2))
    label_classes        = np.sort(np.unique(Y))
    for y in label_classes:
        Y1_y_idxs        = np.nonzero(Y1 == y)[0]
        X1_comps_for_y   = X1_comps[Y1_y_idxs]
        Y2_y_idxs        = np.nonzero(Y2 == y)[0]
        X2_comps_for_y   = X2_comps[Y2_y_idxs]
        Y2_prob          = len(Y2_y_idxs)/float(X2_comps.shape[0])
        logger.info('Probability of label class ' + str(y) + ': ' + str(Y2_prob))
        y_W2_dist        = wasserstein_dist(X1_comps_for_y, X2_comps_for_y)
        logger.info('Wasserstein-2 distance of covariate shift: ' + str(y_W2_dist))
        conditional_shift_W2 += Y2_prob * y_W2_dist
    logger.info('Wasserstein-2 distance of conditional shift: ' + str(conditional_shift_W2))
    
    return covariate_shift_W2, conditional_shift_W2
    
def quantity_shifts_in_sequence(dataset_name,
                                shift_sequence,
                                source_sample_size,
                                target_sample_sizes,
                                final_target_test_size,
                                seed):
    '''
    Quantify covariate and conditional shifts
    by computing Wasserstein-2 distances on top PCA components of images
    between every pair of consecutive time steps
    @param dataset_name: str, cifar10, cifar100, or portraits,
                         next 4 arguments disregarded for portraits
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of allowed shift types
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param seed: int, for np random generator
    @return: None
    '''
    if dataset_name in real_world_datasets:
        shift_sequence_with_sample_size = dataset_name
    else:
        shift_sequence_split            = shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
    
    output_dir       = config.output_dir + 'shift_quantities/' + shift_sequence_with_sample_size + '/seed' + str(seed) + '/'
    logging_dir      = output_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_filename = logging_dir + 'quantify_shifts_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
    logger           = set_up_logger('logger_main',
                                     logging_filename)
    logger.info('Quantifying shifts')
    logger.info('Dataset: '                + dataset_name)
    if dataset_name not in real_world_datasets:
        logger.info('Shift sequence: '         + shift_sequence)
        logger.info('Source sample size: '     + str(source_sample_size))
        logger.info('Target sample sizes: '    
                    + ', '.join([str(target_sample_size) for target_sample_size in target_sample_sizes]))
        logger.info('Final target test size: ' + str(final_target_test_size))
    logger.info('Seed: '                   + str(seed))
    
    if dataset_name == 'portraits':
        data_loaders_over_time = load_yearbook_data(logger)
    else:
        data_loaders_over_time = load_datasets_over_time(logger,
                                                         dataset_name,
                                                         shift_sequence,
                                                         source_sample_size,
                                                         target_sample_sizes,
                                                         final_target_test_size,
                                                         seed             = seed,
                                                         visualize_shifts = False,
                                                         output_dir       = output_dir)
    
    # Fit PCA on images from all time points
    max_pca_components     = 100
    pca_explained_variance = .95
    X_comps_over_time, final_test_X_comps = run_pca_on_all_train_images(data_loaders_over_time,
                                                                        logger,
                                                                        pca_explained_variance,
                                                                        max_pca_components)
    Y_over_time  = [data_loaders['train'].dataset.tensors[1] for data_loaders in data_loaders_over_time]
    final_test_Y = data_loaders_over_time[-1]['test'].dataset.tensors[1]
    if torch.cuda.is_available():
        Y_over_time  = [Y.cpu() for Y in Y_over_time]
        final_test_Y = final_test_Y.cpu()
    Y_over_time  = [Y.numpy() for Y in Y_over_time]
    final_test_Y = final_test_Y.numpy()
    
    # As a sanity check, is Wasserstein-2 distance close to 0 if we compare half of the source dist samples to the other half?
    num_t0_train_samples = len(Y_over_time[0])
    num_t0_samples_half  = int(num_t0_train_samples/2.)
    t0_train_sample_idxs = np.arange(num_t0_train_samples)
    np.random.shuffle(t0_train_sample_idxs)
    t0_s1_sample_idxs    = t0_train_sample_idxs[:num_t0_samples_half]
    t0_s2_sample_idxs    = t0_train_sample_idxs[num_t0_samples_half:]
    logger.info('Computing shift between two halves of source distribution')
    source_covariate_shift_W2, source_conditional_shift_W2 \
        = quantify_shift_between_distributions(X_comps_over_time[0][t0_s1_sample_idxs],
                                               Y_over_time[0][t0_s1_sample_idxs],
                                               X_comps_over_time[0][t0_s2_sample_idxs],
                                               Y_over_time[0][t0_s2_sample_idxs],
                                               logger)
    
    covariate_shift_W2s   = []
    conditional_shift_W2s = []
    for time_idx in range(1, len(data_loaders_over_time)):
        logger.info('Computing shift from time ' + str(time_idx - 1) + ' to time ' + str(time_idx))
        covariate_shift_W2, conditional_shift_W2 \
            = quantify_shift_between_distributions(X_comps_over_time[time_idx - 1],
                                                   Y_over_time[time_idx - 1],
                                                   X_comps_over_time[time_idx],
                                                   Y_over_time[time_idx],
                                                   logger)
        covariate_shift_W2s.append(covariate_shift_W2)
        conditional_shift_W2s.append(conditional_shift_W2)

    covariate_shift_rel_to_final_test_W2s   = []
    conditional_shift_rel_to_final_test_W2s = []
    for time_idx in range(len(data_loaders_over_time)):
        logger.info('Computing shift from time ' + str(time_idx) + ' train to final time test')
        covariate_shift_W2, conditional_shift_W2 \
            = quantify_shift_between_distributions(X_comps_over_time[time_idx],
                                                   Y_over_time[time_idx],
                                                   final_test_X_comps,
                                                   final_test_Y,
                                                   logger)
        covariate_shift_rel_to_final_test_W2s.append(covariate_shift_W2)
        conditional_shift_rel_to_final_test_W2s.append(conditional_shift_W2)
    
    with open(output_dir + 'quantify_shift_Wasserstein_dists.json', 'w') as f:
        json.dump({'Covariate shift W2s'  : covariate_shift_W2s,
                   'Conditional shift W2s': conditional_shift_W2s,
                   'Covariate shift relative to final test W2s'  : covariate_shift_rel_to_final_test_W2s,
                   'Conditional shift relative to final test W2s': conditional_shift_rel_to_final_test_W2s,
                   'Source covariate shift W2'  : source_covariate_shift_W2,
                   'Source conditional shift W2': source_conditional_shift_W2},
                  f)
    
def create_parser():
    '''
    Create argument parser
    @return: ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Quantify the degree of shift between 2 distributions.'))
    parser.add_argument('--dataset',
                        action  = 'store',
                        type    = str,
                        help    = 'Specify cifar10, cifar100, or portraits.')
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
    parser.add_argument('--seed',
                        action  = 'store',
                        type    = int,
                        default = 1007,
                        help    = 'Specify random seed.')
    return parser
    
if __name__ == '__main__':
    
    parser = create_parser()
    args   = parser.parse_args()
    
    real_world_datasets = {'portraits'}
    if args.dataset not in real_world_datasets:
        target_sample_sizes = [int(i) for i in args.target_sample_size_seq.split(':')]
    else:
        target_sample_sizes = []
    
    quantity_shifts_in_sequence(args.dataset,
                                args.shift_sequence,
                                args.source_sample_size,
                                target_sample_sizes,
                                args.target_test_size,
                                args.seed)