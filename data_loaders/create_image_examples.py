import os
import sys
import argparse

from datetime import datetime
from os.path import join, dirname, abspath

from load_image_data_with_shifts import load_datasets_over_time
from load_yearbook_data import load_yearbook_data

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def create_parser():
    '''
    Create argument parser
    @return: ArgumentParser
    '''
    parser = argparse.ArgumentParser(description = ('Visualize sequence of examples in dataset.'))
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
    
    output_dir  = config.output_dir + 'image_visualization/'
    logging_dir = output_dir + 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    
    if args.dataset == 'portraits':
        logging_filename = logging_dir + 'create_portraits_image_examples_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
        image_output_dir = output_dir + 'portraits/'
    else:
        shift_sequence_split            = args.shift_sequence.split(':')
        num_target_steps                = len(shift_sequence_split)
        target_sample_sizes             = [int(size) for size in args.target_sample_size_seq.split(':')]
        assert len(target_sample_sizes) == num_target_steps
        shifts_with_sample_size         = [shift_sequence_split[i] + str(target_sample_sizes[i])
                                           for i in range(num_target_steps)]
        shift_sequence_with_sample_size = 'source' + str(args.source_sample_size) + ':' + ':'.join(shifts_with_sample_size)
        logging_filename = logging_dir + 'create_' + args.dataset + '_image_examples_' + shift_sequence_with_sample_size + '_seed' + str(args.seed) \
                         + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.log'
        image_output_dir = output_dir + args.dataset + '_' + shift_sequence_with_sample_size + '_seed' + str(args.seed) + '/'
    
    logger = set_up_logger('logger_main',
                           logging_filename)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    if args.dataset == 'portraits':
        load_yearbook_data(logger,
                           image_output_dir + 'portraits_examples.pdf')
    else:
        load_datasets_over_time(logger,
                                args.dataset,
                                args.shift_sequence,
                                args.source_sample_size,
                                target_sample_sizes,
                                args.target_test_size,
                                seed             = args.seed,
                                visualize_shifts = True,
                                output_dir       = image_output_dir)