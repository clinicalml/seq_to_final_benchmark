import os
import sys

from PIL import Image
from itertools import product

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def load_yearbook_data(logger,
                       fig_file_name = None):
    '''
    Create PyTorch data loaders with yearbook photos from each decade starting in 1930s, after 2000s are grouped
    After 2000s is split 40-10-50 for train-val-test. Other times split 80-20.
    Y = 0 for M, Y = 1 for F
    Images are 32 x 32
    @param logger: logger, for INFO messages
    @param fig_file_name: str, path to save figure showing some examples, None to skip creating figure
    @return: list of dict mapping str to torch DataLoaders, list over time,
             dict maps train / valid /test split to loader containing images and outcomes,
             training samples in size-100 batches, validation and test samples in a single batch,
             test samples only at final time step
    '''
    np.random.seed(1007)
    data_dir   = config.data_dir + 'yearbook_pictures/faces_aligned_small_mirrored_co_aligned_cropped_cleaned/'
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()])
    data_loaders = []
    examples     = []
    for time_idx in range(8):
        time_dataset = torchvision.datasets.ImageFolder(data_dir + 'time' + str(time_idx))
        time_tuples = time_dataset.imgs
        time_images = torch.stack([transforms(Image.open(i[0])) for i in time_tuples])
        normalize_transform = torchvision.transforms.Normalize(255*torch.FloatTensor([0.485, 0.456, 0.406]),
                                                               255*torch.FloatTensor([0.229, 0.224, 0.225]))
        time_images = normalize_transform(time_images)
        time_labels = torch.LongTensor([i[1] for i in time_tuples])
        time_idxs   = np.arange(len(time_labels))
        np.random.shuffle(time_idxs)
        if time_idx != 7:
            train_idxs = time_idxs[:int(.8*len(time_idxs))]
            valid_idxs = time_idxs[int(.8*len(time_idxs)):]
        else:
            train_idxs = time_idxs[:int(.4*len(time_idxs))]
            valid_idxs = time_idxs[int(.4*len(time_idxs)):int(.5*len(time_idxs))]
            test_idxs  = time_idxs[int(.5*len(time_idxs)):]
        
        time_train_dataset = torch.utils.data.TensorDataset(time_images[train_idxs],
                                                            time_labels[train_idxs])
        time_train_loader  = torch.utils.data.DataLoader(time_train_dataset,
                                                         batch_size = 100,
                                                         shuffle    = True)
        
        time_valid_dataset = torch.utils.data.TensorDataset(time_images[valid_idxs],
                                                            time_labels[valid_idxs])
        time_valid_loader  = torch.utils.data.DataLoader(time_valid_dataset,
                                                         batch_size = len(valid_idxs),
                                                         shuffle    = False)
        time_data_loaders = {'train': time_train_loader,
                             'valid': time_valid_loader}
        logger.info(str(len(train_idxs)) + ' training samples at time ' + str(time_idx))
        logger.info(str(len(valid_idxs)) + ' validation samples at time ' + str(time_idx))
        
        if time_idx == 7:
            time_test_dataset = torch.utils.data.TensorDataset(time_images[test_idxs],
                                                               time_labels[test_idxs])
            final_test_loader = torch.utils.data.DataLoader(time_test_dataset,
                                                            batch_size = len(test_idxs),
                                                            shuffle    = False)
            time_data_loaders['test'] = final_test_loader
            logger.info(str(len(test_idxs)) + ' test samples at time ' + str(time_idx))
        
        data_loaders.append(time_data_loaders)
        
        time_female_examples = []
        time_male_examples   = []
        # add female examples first
        example_idx = 0
        while len(time_female_examples) < 3 or len(time_male_examples) < 3:
            if time_labels[train_idxs[example_idx]] == 0:
                if len(time_male_examples) < 3:
                    time_male_examples.append(time_images[train_idxs[example_idx]])
            else:
                if len(time_female_examples) < 3:
                    time_female_examples.append(time_images[train_idxs[example_idx]])
            example_idx += 1
        examples.append(time_female_examples + time_male_examples)

    if fig_file_name is not None:
        time_period_names = ['1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s']
        visualize_yearbook_examples(examples,
                                    time_period_names,
                                    fig_file_name)
    
    return data_loaders

def visualize_yearbook_examples(image_examples,
                                time_period_names,
                                file_name):
    '''
    Make a 6 x # time steps grid showing 3 examples per gender in each time period
    @param image_examples: list of lists of images, outer list over time period, inner list contains 3 female and 3 male
    @param time_period_names: list of str, name of each time period for row labels
    @param file_name: str, path to save figure
    @return: None
    '''
    num_time_steps = len(image_examples)
    assert len(time_period_names) == num_time_steps
    fig, ax = plt.subplots(nrows   = num_time_steps,
                           ncols   = 6,
                           figsize = (3, num_time_steps/2),
                           dpi     = 1000)
    for time_idx, example_idx in product(range(num_time_steps), range(6)):
        time_ex_image     = image_examples[time_idx][example_idx].numpy().transpose(1, 2, 0)
        reverse_norm_std  = np.array([[[0.229, 0.224, 0.225]]])*255
        reverse_norm_mean = np.array([[[0.485, 0.456, 0.406]]])*255
        time_ex_image     = np.multiply(time_ex_image, reverse_norm_std) + reverse_norm_mean
        ax[time_idx, example_idx].imshow(time_ex_image,
                                         interpolation = 'nearest')
        ax[time_idx, example_idx].set_xticks([])
        ax[time_idx, example_idx].set_yticks([])
        # make xaxis invisible
        ax[time_idx, example_idx].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[time_idx, example_idx].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[time_idx, example_idx].tick_params(left=False, labelleft=False, right=False)
        #remove background patch (only needed for non-white background)
        ax[time_idx, example_idx].patch.set_visible(False)
    for time_idx in range(num_time_steps):
        ax[time_idx, 0].set_ylabel(time_period_names[time_idx], fontsize=5)
    ax[0, 1].set_title('female', fontsize=5)
    ax[0, 4].set_title('male', fontsize=5)
    plt.subplots_adjust(wspace = .05,
                        hspace = .05)
    fig.savefig(file_name)