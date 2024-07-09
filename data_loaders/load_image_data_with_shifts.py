import sys
import time
import json

from os.path import join, dirname, abspath
from PIL import Image
from itertools import product

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import corruptions

sys.path.append(dirname(dirname(abspath(__file__))))
import config

def apply_corruption(images,
                     corruption_name):
    '''
    Apply corruptions with severity 5 defined in 
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    Note: These functions assume the images are not normalized, so be careful when using this function!
    @param images: torch FloatTensor, contains features (from images)
    @param corruption_name: str, name of corruption to apply to image
    @return: torch FloatTensor, contains corrupted features (from images)
    '''
    corruption_function = getattr(corruptions,
                                  corruption_name)
    corrupted_images = torch.FloatTensor(corruption_function(images, severity = 5))
    return corrupted_images

def apply_gaussian_noise(images,
                         scale = .38):
    '''
    Apply Gaussian noise to normalized image
    @param images: torch FloatTensor, contains features (from images)
    @param scale: float, standard deviation for Gaussian noise is drawn from (added to 0-1 scale)
    @return: torch FloatTensor, contains corrupted features (from images)
    '''
    return np.clip(images + np.random.normal(size=images.shape, scale=scale), 0, 1).float()

def apply_uniform_random_discrete_noise(images,
                                        scale = 3):
    '''
    Draw random noise for each pixel from discrete uniform distribution [-scale, scale]
    @param images: torch FloatTensor, contains features (from images)
    @param scale: int, range for drawing noise, on 0-255 scale
    @return: torch FloatTensor, contains corrupted features (from images)
    '''
    if float(torch.max(images)) <= 1:
        return np.clip(images + np.random.randint(int(-1 * scale), int(scale), size = images.shape)/256., 0, 1).float()
    return np.clip(images + np.random.randint(int(-1 * scale), int(scale), size = images.shape), 0, 255).float()

def apply_rotation(images,
                   rotation_angle):
    '''
    Rotate images
    @param images: torch FloatTensor, contains features (from images)
    @param rotation_angle: float, number of degrees to rotate image counter-clockwise
    @return: torch FloatTensor, contains rotated features (from images)
    '''
    rotation = torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.rotate(x, rotation_angle))
    rotated_images = [rotation(image)[None] # None adds dimension in front
                      for image in images]
    return torch.vstack(rotated_images)

def apply_conditional_rotation(images,
                               labels,
                               rotation_angle):
    '''
    Rotate images different angles for each label class
    @param images: torch FloatTensor, contains features (from images)
    @param labels: torch LongTensor, contains labels for each image
    @param rotation_angle: np array of floats, number of degrees to rotate image counter-clockwise for each label class
    @return: images modified
    '''
    assert images.shape[0] == len(labels)
    num_label_classes = len(rotation_angle)
    assert num_label_classes > torch.max(labels)
    for label_class in range(num_label_classes):
        if rotation_angle[label_class] != 0:
            label_idxs = torch.where(labels == label_class)[0]
            if len(label_idxs) > 0:
                images[label_idxs] = apply_rotation(images[label_idxs],
                                                    rotation_angle[label_class])
    return images

def apply_horizontal_flip(images):
    '''
    Horizontally flip images
    @param images: torch FloatTensor, contains features (from images)
    @return: torch FloatTensor, contains horizontally flipped features (from images)
    '''
    horizontal_flip = torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.hflip(x))
    flipped_images  = [horizontal_flip(image)[None]
                       for image in images]
    return torch.vstack(flipped_images)

def apply_center_crop(images,
                      new_dim):
    '''
    Crop images to center new_dim x new_dim pixels, then resize to fill original dimensions
    Effectively zooms into center of image
    @param images: torch FloatTensor, contains features (from images)
    @param new_dim: int, number of pixels to retain at center, must be less than image height and width
    @return: torch FloatTensor, contains zoomed-in images
    '''
    original_height = images.shape[2]
    assert images.shape[3] == original_height
    transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(new_dim),
                                                 torchvision.transforms.Resize(original_height)])
    
    zoomed_images = [transforms(image)[None]
                     for image in images]
    return torch.vstack(zoomed_images)

def apply_recoloring(images,
                     recoloring):
    '''
    Tint images by adding constant to each RGB channel as specified
    @param images: torch FloatTensor, contains features (from images), # samples x 3 x H x W or # samples x H x W x 3
    @param recoloring: np array of floats, length 3, number of pixels to add to each RGB channel, on 0-255 scale
    @return: images recolored
    '''
    assert len(recoloring) == 3
    if images.shape[1] == 3:
        recoloring_dim = np.expand_dims(np.expand_dims(np.expand_dims(recoloring, 0), 2), 2)
    else:
        assert images.shape[3] == 3
        recoloring_dim = np.expand_dims(np.expand_dims(np.expand_dims(recoloring, 0), 0), 0)
    if float(torch.max(images)) <= 1:
        return np.clip(images + recoloring_dim/256., 0, 1).float()
    return np.clip(images + recoloring_dim, 0, 255).float()

def apply_conditional_recoloring(images,
                                 labels,
                                 recoloring):
    '''
    Recolor images with different changes for each label class
    @param images: torch FloatTensor, contains features (from images)
    @param labels: torch LongTensor, contains labels for each image
    @param recoloring: np array of floats, # label classes x 3, number of pixels to add to each RGB channel for each label class,
                       on 0-255 scale
    @return: images modified
    '''
    assert images.shape[0] == len(labels)
    num_label_classes = recoloring.shape[0]
    assert num_label_classes > torch.max(labels)
    assert recoloring.shape[1] == 3
    for label_class in range(num_label_classes):
        if not np.all(recoloring[label_class] == 0):
            label_idxs = torch.where(labels == label_class)[0]
            if len(label_idxs) > 0:
                images[label_idxs] = apply_recoloring(images[label_idxs],
                                                      recoloring[label_class])
    return images

def manually_augment_images(augmentation,
                            train_images,
                            train_labels,
                            valid_images,
                            valid_labels):
    '''
    Augment images with horizontal flips, center crops, and rotations
    Creates multiple copies of each image. If draw is specified, a subset that matches the original size is selected.
    @param augmentation: str, multiple, multiple_with_center_crop, multiple_draw, multiple_with_center_crop_draw
    @param train_images: torch FloatTensor, # train samples x 3 x H x W or # samples x H x W x 3
    @param train_labels: torch LongTensor, labels
    @param valid_images: torch FloatTensor, # valid samples x 3 x H x W or # samples x H x W x 3
    @param valid_labels: torch LongTensor, labels for valid samples
    @return: 1. torch FloatTensor, augmented train images
             2. torch LongTensor, augmented train labels
             3. torch FloatTensor, augmented validation images
             4. torch LongTensor, augmented validation labels
    '''
    augmented_train_images = [train_images]
    augmented_valid_images = [valid_images]
    flipped_train_images   = apply_horizontal_flip(train_images)
    flipped_valid_images   = apply_horizontal_flip(valid_images)
    augmented_train_images.append(flipped_train_images)
    augmented_valid_images.append(flipped_valid_images)
    if not silent:
        logger.info('Added horizontal flip augmentation')

    if augmentation.startswith('multiple_with_center_crop'):
        centered_train_images         = []
        centered_valid_images         = []
        flipped_centered_train_images = []
        flipped_centered_valid_images = []
        center_sizes                  = [30, 28, 26, 24]
        for center_size in center_sizes:
            centered_train_images.append(apply_center_crop(train_images, center_size))
            centered_valid_images.append(apply_center_crop(valid_images, center_size))
            if not silent:
                logger.info('Added zoom in to ' + str(center_size) + 'x' + str(center_size) + ' augmentation')
            flipped_centered_train_images.append(apply_center_crop(flipped_train_images, center_size))
            flipped_centered_valid_images.append(apply_center_crop(flipped_valid_images, center_size))
            if not silent:
                logger.info('Added horizontal flip + zoom in to ' 
                            + str(center_size) + 'x' + str(center_size) + ' augmentation')
        augmented_train_images += centered_train_images
        augmented_valid_images += centered_valid_images
        augmented_train_images += flipped_centered_train_images
        augmented_valid_images += flipped_centered_valid_images

    for rotation_augment_angle in [-15, -10, -5, 5, 10, 15]:
        augmented_train_images.append(apply_rotation(train_images, rotation_augment_angle))
        augmented_valid_images.append(apply_rotation(valid_images, rotation_augment_angle))
        if not silent:
            logger.info('Added rotate ' + str(rotation_augment_angle) + ' degrees augmentation')
        augmented_train_images.append(apply_rotation(flipped_train_images, rotation_augment_angle))
        augmented_valid_images.append(apply_rotation(flipped_valid_images, rotation_augment_angle))
        if not silent:
            logger.info('Added horizontal flip + rotate ' + str(rotation_augment_angle) + ' degrees augmentation')

        if augmentation.startswith('multiple_with_center_crop') and rotation_augment_angle != -15:
            for center_idx in range(len(center_sizes)):
                augmented_train_images.append(apply_rotation(centered_train_images[center_idx], rotation_augment_angle))
                augmented_valid_images.append(apply_rotation(centered_valid_images[center_idx], rotation_augment_angle))
                if not silent:
                    logger.info('Added zoom in to ' + str(center_sizes[center_idx]) + 'x' + str(center_sizes[center_idx])
                                + ' + rotate ' + str(rotation_augment_angle) + ' degrees augmentation')

                augmented_train_images.append(apply_rotation(flipped_centered_train_images[center_idx], 
                                                             rotation_augment_angle))
                augmented_valid_images.append(apply_rotation(flipped_centered_valid_images[center_idx], 
                                                             rotation_augment_angle))
                if not silent:
                    logger.info('Added zoom in to ' + str(center_sizes[center_idx]) + 'x' + str(center_sizes[center_idx])
                                + ' + horizontal flip + rotate ' + str(rotation_augment_angle) + ' degrees augmentation')

    train_labels = train_labels.repeat(len(augmented_train_images))
    valid_labels = valid_labels.repeat(len(augmented_valid_images))
    train_images = torch.cat(augmented_train_images)
    valid_images = torch.cat(augmented_valid_images)

    if augmentation.endswith('_draw'):
        # randomly draw specified sample sizes from augmented images
        augmented_train_indices = np.arange(len(augmented_train_images))
        augmented_valid_indices = np.arange(len(augmented_valid_images))
        np.random.shuffle(augmented_train_indices)
        np.random.shuffle(augmented_valid_indices)
        train_labels = augmented_train_labels[augmented_train_indices[:int(.4*sample_size)]]
        valid_labels = augmented_valid_labels[augmented_valid_indices[:int(.1*sample_size)]]
        train_images = augmented_train_images[augmented_train_indices[:int(.4*sample_size)]]
        valid_images = augmented_valid_images[augmented_valid_indices[:int(.1*sample_size)]]
    
    return train_images, train_labels, valid_images, valid_labels

def aggregate_and_shuffle_samples(images,
                                  labels):
    '''
    Aggregate lists of samples into a single torch Tensor and shuffle the samples
    @param images: list of torch FloatTensors, # samples x 32 x H x W or # samples x H x W x 3
    @param labels: list of torch LongTensors, labels for samples, list and tensors of same length as images
    @return: 1. torch FloatTensor, aggregated images
             2. torch LongTensor, aggregated labels
    '''
    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    
    sample_idxs = np.arange(len(images))
    np.random.shuffle(sample_idxs)
    
    return images[sample_idxs], labels[sample_idxs]

def select_cifar10_samples(label_probs,
                           sample_size,
                           orig_train_images,
                           orig_train_labels,
                           orig_test_images = None,
                           orig_test_labels = None,
                           test_size        = 0):
    '''
    Select CIFAR-10 examples based on label probabilities
    @param label_probs: np array of floats, probabilities of each label class
    @param sample_size: int, total number of samples for train, valid, + test splits
    @param orig_train_images: torch FloatTensor, # samples x 3 x H x W or # samples x H x W x 3, used for train + val split
    @param orig_train_labels: torch LongTensor, labels for train + val split
    @param orig_test_images: torch FloatTensor, # samples x 3 x H x W or # samples x H x W x 3, used for test split, 
                             None if test_size is 0
    @param orig_test_labels: torch LongTensor, labels for test split, None if test_size is 0
    @param test_size: int, number of samples for test split
    @return: 1. torch FloatTensor, selected train images
             2. torch LongTensor, selected train labels
             3. torch FloatTensor, selected valid images
             4. torch LongTensor, selected valid labels
             5. torch FloatTensor, selected test images, None if test_size is 0
             6. torch LongTensor, selected test labels, None if test_size is 0
    '''
    if test_size > 0:
        assert orig_test_images is not None
        assert orig_test_labels is not None
    else:
        test_images = None
        test_labels = None
    num_classes = 10
    num_train_val_samples = sample_size - test_size
    num_train_val_samples_so_far = 0
    train_images = []
    valid_images = []
    train_labels = []
    valid_labels = []
    if test_size > 0:
        num_test_samples_so_far = 0
        test_images = []
        test_labels = []
    for label_class in range(num_classes):
        if label_class == num_classes - 1 or np.sum(label_probs[label_class + 1:]) == 0: # no more classes come after
            class_train_val_size = num_train_val_samples - num_train_val_samples_so_far
            assert class_train_val_size >= 0
            if test_size > 0:
                class_test_size  = test_size - num_test_samples_so_far
                assert class_test_size >= 0
        else:
            class_train_val_size = int(num_train_val_samples * label_probs[label_class])
            num_train_val_samples_so_far += class_train_val_size
            if test_size > 0:
                class_test_size  = int(test_size * label_probs[label_class])
                num_test_samples_so_far += class_test_size
        class_train_size         = int(.8 * class_train_val_size)
        class_orig_train_indices = torch.nonzero(torch.where(orig_train_labels == label_class, 1, 0), as_tuple = True)[0]
        shuffled_train_idxs      = np.arange(len(class_orig_train_indices))
        np.random.shuffle(shuffled_train_idxs)
        class_orig_train_indices = class_orig_train_indices[shuffled_train_idxs]
        assert class_train_val_size <= len(class_orig_train_indices)
        if test_size > 0:
            class_orig_test_indices = torch.nonzero(torch.where(orig_test_labels == label_class, 1, 0), 
                                                    as_tuple = True)[0]
            shuffled_test_idxs      = np.arange(len(class_orig_test_indices))
            np.random.shuffle(shuffled_test_idxs)
            class_orig_test_indices = class_orig_test_indices[shuffled_test_idxs]
            assert class_test_size <= len(class_orig_test_indices)
        class_train_indices = class_orig_train_indices[:class_train_size]
        class_train_images  = orig_train_images[class_train_indices]
        class_train_labels  = orig_train_labels[class_train_indices]
        class_valid_indices = class_orig_train_indices[class_train_size:class_train_val_size]
        class_valid_images  = orig_train_images[class_valid_indices]
        class_valid_labels  = orig_train_labels[class_valid_indices]
        if test_size > 0:
            class_test_indices  = class_orig_test_indices[:class_test_size]
            class_test_images   = orig_test_images[class_test_indices]
            class_test_labels   = orig_test_labels[class_test_indices]

        train_images.append(class_train_images)
        valid_images.append(class_valid_images)
        train_labels.append(class_train_labels)
        valid_labels.append(class_valid_labels)
        if test_size > 0:
            test_images.append(class_test_images)
            test_labels.append(class_test_labels)
            
    train_images, train_labels = aggregate_and_shuffle_samples(train_images, train_labels)
    valid_images, valid_labels = aggregate_and_shuffle_samples(valid_images, valid_labels)
    if test_size > 0:
        test_images, test_labels = aggregate_and_shuffle_samples(test_images, test_labels)
            
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def select_cifar100_samples(subpops,
                            label_probs,
                            sample_size,
                            orig_train_images,
                            orig_train_labels,
                            orig_test_images = None,
                            orig_test_labels = None,
                            test_size        = 0):
    '''
    Select CIFAR-100 examples based on label probabilities and sub-population proportions across all label classes
    Labels will be replaced by coarse-grained 1-20
    @param subpops: np array of floats, length 5, proportion of samples to take from each sub-class in CIFAR-100
    @param label_probs: np array of floats, probabilities of each label class
    @param sample_size: int, total number of samples for train, valid, + test splits
    @param orig_train_images: torch FloatTensor, # samples x 3 x H x W or # samples x H x W x 3, used for train + val split
    @param orig_train_labels: torch LongTensor, labels for train + val split
    @param orig_test_images: torch FloatTensor, # samples x 3 x H x W or # samples x H x W x 3, used for test split, 
                             None if test_size is 0
    @param orig_test_labels: torch LongTensor, labels for test split, None if test_size is 0
    @param test_size: int, number of samples for test split
    @return: 1. torch FloatTensor, selected train images
             2. torch LongTensor, selected train labels
             3. torch FloatTensor, selected valid images
             4. torch LongTensor, selected valid labels
             5. torch FloatTensor, selected test images, None if test_size is 0
             6. torch LongTensor, selected test labels, None if test_size is 0
    '''
    if test_size > 0:
        assert orig_test_images is not None
        assert orig_test_labels is not None
    else:
        test_images = None
        test_labels = None
    assert np.isclose(np.sum(label_probs), 1)
    assert np.isclose(np.sum(subpops), 1)
    num_classes = 20
    num_train_val_samples = sample_size - test_size
    num_train_val_samples_so_far = 0
    train_images  = []
    valid_images  = []
    train_labels  = []
    valid_labels  = []
    subpop_labels = [['otter', 'seal', 'beaver', 'dolphin', 'whale'],
                     ['flatfish', 'trout', 'aquarium fish', 'ray', 'shark'],
                     ['poppy', 'tulip', 'rose', 'orchid', 'sunflower'],
                     ['bottle', 'cup', 'can', 'bowl', 'plate'],
                     ['apple', 'pear', 'orange', 'sweet pepper', 'mushroom'],
                     ['clock', 'lamp', 'telephone', 'television', 'keyboard'],
                     ['chair', 'couch', 'bed', 'table', 'wardrobe'],
                     ['beetle', 'cockroach', 'bee', 'butterfly', 'caterpillar'],
                     ['leopard', 'tiger', 'lion', 'bear', 'wolf'],
                     ['road', 'bridge', 'house', 'castle', 'skyscraper'],
                     ['plain', 'forest', 'mountain', 'cloud', 'sea'],
                     ['camel', 'cattle', 'chimpanzee', 'kangaroo', 'elephant'],
                     ['skunk', 'possum', 'raccoon', 'porcupine', 'fox'],
                     ['crab', 'lobster', 'spider', 'snail', 'worm'],
                     ['man', 'woman', 'boy', 'girl', 'baby'],
                     ['lizard', 'snake', 'turtle', 'crocodile', 'dinosaur'],
                     ['mouse', 'shrew', 'hamster', 'rabbit', 'squirrel'],
                     ['oak', 'pine', 'maple', 'willow', 'palm'],
                     ['bicycle', 'motorcycle', 'pickup truck', 'bus', 'train'],
                     ['streetcar', 'tractor', 'tank', 'lawn mower', 'rocket']]
    with open('cifar100_labels.json', 'r') as f:
        orig_labels = json.load(f)
                     
    if test_size > 0:
        num_test_samples_so_far = 0
        test_images = []
        test_labels = []
    for label_class in range(num_classes):
        if label_class == num_classes - 1 or np.sum(label_probs[label_class + 1:]) == 0: # no more classes come after
            class_train_val_size = num_train_val_samples - num_train_val_samples_so_far
            assert class_train_val_size >= 0
            if test_size > 0:
                class_test_size  = test_size - num_test_samples_so_far
                assert class_test_size >= 0
        else:
            class_train_val_size = int(num_train_val_samples * label_probs[label_class])
            num_train_val_samples_so_far += class_train_val_size
            if test_size > 0:
                class_test_size  = int(test_size * label_probs[label_class])
                num_test_samples_so_far += class_test_size
        
        num_class_train_val_samples_so_far = 0
        num_class_test_samples_so_far      = 0
        for subpop_idx in range(len(subpops)):
            if subpop_idx == len(subpops) - 1 or np.sum(subpops[subpop_idx + 1:]) == 0: # no more subpop come after
                subpop_train_val_size = class_train_val_size - num_class_train_val_samples_so_far
                assert subpop_train_val_size >= 0
                if test_size > 0:
                    subpop_test_size = class_test_size - num_class_test_samples_so_far
                    assert subpop_test_size >= 0
            else:
                subpop_train_val_size = int(class_train_val_size * subpops[subpop_idx])
                num_class_train_val_samples_so_far += subpop_train_val_size
                if test_size > 0:
                    subpop_test_size = int(class_test_size * subpops[subpop_idx])
                    num_class_test_samples_so_far += subpop_test_size
                    
            subpop_label = orig_labels.index(subpop_labels[label_class][subpop_idx])
            subpop_train_size = int(.8 * subpop_train_val_size)
            subpop_orig_train_indices = torch.nonzero(torch.where(orig_train_labels == subpop_label, 1, 0), as_tuple = True)[0]
            shuffled_train_idxs = np.arange(len(subpop_orig_train_indices))
            np.random.shuffle(shuffled_train_idxs)
            subpop_orig_train_indices = subpop_orig_train_indices[shuffled_train_idxs]
            assert subpop_train_val_size <= len(subpop_orig_train_indices)
            if test_size > 0:
                subpop_orig_test_indices = torch.nonzero(torch.where(orig_test_labels == subpop_label, 1, 0), as_tuple = True)[0]
                shuffled_test_idxs = np.arange(len(subpop_orig_test_indices))
                np.random.shuffle(shuffled_test_idxs)
                subpop_orig_test_indices = subpop_orig_test_indices[shuffled_test_idxs]
                assert subpop_test_size <= len(subpop_orig_test_indices)
            subpop_train_indices = subpop_orig_train_indices[:subpop_train_size]
            subpop_train_images  = orig_train_images[subpop_train_indices]
            subpop_train_labels  = torch.LongTensor(label_class * np.ones(subpop_train_size))
            subpop_valid_indices = subpop_orig_train_indices[subpop_train_size:subpop_train_val_size]
            subpop_valid_images  = orig_train_images[subpop_valid_indices]
            subpop_valid_labels  = torch.LongTensor(label_class * np.ones(subpop_train_val_size - subpop_train_size))
            if test_size > 0:
                subpop_test_indices  = subpop_orig_test_indices[:subpop_test_size]
                subpop_test_images   = orig_test_images[subpop_test_indices]
                subpop_test_labels   = torch.LongTensor(label_class * np.ones(subpop_test_size))

            train_images.append(subpop_train_images)
            valid_images.append(subpop_valid_images)
            train_labels.append(subpop_train_labels)
            valid_labels.append(subpop_valid_labels)
            if test_size > 0:
                test_images.append(subpop_test_images)
                test_labels.append(subpop_test_labels)
                
    train_images, train_labels = aggregate_and_shuffle_samples(train_images, train_labels)
    valid_images, valid_labels = aggregate_and_shuffle_samples(valid_images, valid_labels)
    if test_size > 0:
        test_images, test_labels = aggregate_and_shuffle_samples(test_images, test_labels)
                
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def load_corrupted_dataset(logger,
                           dataset_name    = 'cifar10',
                           corruption_name = [],
                           rotation_angle  = np.zeros(10),
                           label_flip      = [],
                           label_probs     = .1*np.ones(10),
                           recolorings     = np.zeros((10,3)),
                           subpops         = .2*np.ones(5),
                           sample_size     = 20000,
                           seed            = 1007,
                           augmentation    = 'none',
                           batch_size      = 100,
                           silent          = True,
                           test_size       = 10000,
                           examples_for_visualization = False):
    '''
    Load corrupted CIFAR-10 or CIFAR-100 data
    CIFAR-100 set-up uses the 20 label super-classes
    Images are normalized to mean 0 in training set
    Images returned follow shape (# samples, # channels, height, width)
    @param logger: logger, for INFO messages
    @param dataset_name: str, cifar10 or cifar100
    @param corruption_name: list of str, sequence of types of corruption to apply to target images, considered input-level shift
    @param rotation_angle: np array of floats, number of degrees to rotate target images counterclockwise for each label class,
                           conditional rotation applied before flipping labels, considered mid-level shift
    @param label_flip: list of int, series of label swaps to apply, considered output-level shift,
                       swap labels to 9 - Y if 1,
                       swap labels as Y + 2 with Y = 9 -> Y = 1 and Y = 8 -> Y = 0 if 2
    @param label_probs: np array of floats, probabilities of each label class
    @param recolorings: np array of floats, # label classes x 3 channels, tints to add to RGB channels, on 0-255 scale,
                        conditional recoloring applied before flipping labels, considered input-level shift
    @param subpops: np array of floats, length 5, proportion of samples to take from each sub-class in CIFAR-100
    @param sample_size: int, total number of samples
    @param seed: int, seed for numpy random generator
    @param augmentation: str, 'none' applies no augmentations,
                         'random', applies random augmentations in transform pipeline when loading data
                         'multiple', creates 13 augmentations per train/valid sample:
                         1. 6 rotations: -15, -10, -5, 5, 10, 15 degrees
                         2. 7 horizontal flips w/ rotations at same angles above (and 0 degrees)
                         'multiple_with_center_crop', creates additional 18 augmentations per train/valid sample:
                         3. 8 center crops 30 x 30, 28 x 38, 26 x 26, and 24 x 24 w/ and w/o horizontal flips
                         4. 10 center crops w/ and w/o horizontal flips and w/ rotation -10, -5, 5, 10, 15 degrees
                         'multiple_draw'/'multiple_with_center_crop_draw', draws subset of original size from augmented set
    @param batch_size: int, training batch size in data loader
    @param silent: bool, whether to log augmentations
    @param test_size: int, number of test samples, remaining samples are split 80/20 for train/valid
    @param examples_for_visualization: bool, whether to return 1 training example per class un-normalized for plotting
    @return: 1. torch DataLoader, contains feature and outcomes for training samples in size-100 batches
             2. torch DataLoader, contains feature and outcomes for validation samples in a single batch
             3. torch DataLoader, contains feature and outcomes for test samples in a single batch,
                None if test_size is zero
             4. list of np arrays, 1 training example per class, None if class absent in data, 
                non-normalized images, 32 x 32 x 3, None if examples_for_visualization False
    '''
    assert dataset_name in {'cifar10', 'cifar100'}
    if dataset_name == 'cifar10':
        num_classes = 10
    else:
        num_classes = 20
    assert test_size < sample_size
    assert len(rotation_angle) == num_classes
    assert len(label_probs)  == num_classes
    assert np.isclose(np.sum(label_probs), 1)
    assert np.isclose(np.sum(subpops), 1)
    assert recolorings.shape == (num_classes, 3)
    assert augmentation in {'none', 'random', 'multiple', 'multiple_with_center_crop', 'multiple_draw', 
                            'multiple_with_center_crop_draw'}
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if test_size == 0:
        orig_test_images = None
        orig_test_labels = None
    
    if augmentation == 'random':
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=2, padding_mode='edge'),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # means and stds from https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
            torchvision.transforms.Normalize(255*[0.485, 0.456, 0.406],
                                             255*[0.229, 0.224, 0.225])])
    else:
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(255*[0.485, 0.456, 0.406],
                                             255*[0.229, 0.224, 0.225])])
    if dataset_name == 'cifar10':
        orig_train_dataset   = torchvision.datasets.CIFAR10(root      = config.data_dir + 'cifar10', 
                                                            train     = True,
                                                            download  = True,
                                                            transform = train_transforms)
    else:
        orig_train_dataset   = torchvision.datasets.CIFAR100(root      = config.data_dir + 'cifar100', 
                                                             train     = True,
                                                             download  = True,
                                                             transform = train_transforms)
        
    if test_size > 0:
        test_transforms   = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(255*[0.485, 0.456, 0.406],
                                             255*[0.229, 0.224, 0.225])])
        if dataset_name == 'cifar10':
            orig_test_dataset = torchvision.datasets.CIFAR10(root      = config.data_dir + 'cifar10',
                                                             train     = False,
                                                             download  = True,
                                                             transform = test_transforms)
        else:
            orig_test_dataset = torchvision.datasets.CIFAR100(root      = config.data_dir + 'cifar100',
                                                              train     = False,
                                                              download  = True,
                                                              transform = test_transforms)
    
    orig_train_images = torch.FloatTensor(orig_train_dataset.data).permute((0,3,1,2))
    orig_train_labels = torch.LongTensor(orig_train_dataset.targets)
    if test_size > 0:
        orig_test_images  = torch.FloatTensor(orig_test_dataset.data).permute((0,3,1,2))
        orig_test_labels  = torch.LongTensor(orig_test_dataset.targets)
    
    if dataset_name == 'cifar10':
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels \
            = select_cifar10_samples(label_probs,
                                     sample_size,
                                     orig_train_images,
                                     orig_train_labels,
                                     orig_test_images,
                                     orig_test_labels,
                                     test_size)
    else:
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels \
            = select_cifar100_samples(subpops,
                                      label_probs,
                                      sample_size,
                                      orig_train_images,
                                      orig_train_labels,
                                      orig_test_images,
                                      orig_test_labels,
                                      test_size)
    
    for corruption in corruption_name:
        assert corruption == 'discrete_noise'
        train_images = apply_uniform_random_discrete_noise(train_images)
        valid_images = apply_uniform_random_discrete_noise(valid_images)
        if test_size > 0:
            test_images  = apply_uniform_random_discrete_noise(test_images)
    
    train_images = apply_conditional_rotation(train_images,
                                              train_labels,
                                              rotation_angle)
    train_images = apply_conditional_recoloring(train_images,
                                                train_labels,
                                                recolorings)
    valid_images = apply_conditional_rotation(valid_images,
                                              valid_labels,
                                              rotation_angle)
    valid_images = apply_conditional_recoloring(valid_images,
                                                valid_labels,
                                                recolorings)
    if test_size > 0:
        test_images = apply_conditional_rotation(test_images,
                                                 test_labels,
                                                 rotation_angle)
        test_images = apply_conditional_recoloring(test_images,
                                                   test_labels,
                                                   recolorings)
        
    for flip in label_flip:
        if flip == 1:
            train_labels = num_classes - 1 - train_labels
            valid_labels = num_classes - 1 - valid_labels
            if test_size > 0:
                test_labels  = num_classes - 1 - test_labels
        elif flip == 2:
            train_labels = torch.where(train_labels == num_classes - 1, 1,
                                       torch.where(train_labels == num_classes - 2, 0, 
                                                   train_labels + 2))
            valid_labels = torch.where(valid_labels == num_classes - 1, 1,
                                       torch.where(valid_labels == num_classes - 2, 0, 
                                                   valid_labels + 2))
            if test_size > 0:
                test_labels  = torch.where(test_labels  == num_classes - 1, 1,
                                           torch.where(test_labels == num_classes - 2, 0, 
                                                       test_labels + 2))
    
    for label_class in range(num_classes):
        train_class_freq = np.sum(np.where(train_labels == label_class, 1, 0))/len(train_labels)
        valid_class_freq = np.sum(np.where(valid_labels == label_class, 1, 0))/len(valid_labels)
        if test_size > 0:
            test_class_freq  = np.sum(np.where(test_labels  == label_class, 1, 0))/len(test_labels)
            logger.info('Label class ' + str(label_class) + ' frequency: Train: ' + str(train_class_freq)
                        + ', Valid: ' + str(valid_class_freq) + ', Test: ' + str(test_class_freq))
        else:
            logger.info('Label class ' + str(label_class) + ' frequency: Train: ' + str(train_class_freq)
                        + ', Valid: ' + str(valid_class_freq))
        
    if augmentation.startswith('multiple'):
        train_images, train_labels, valid_images, valid_labels \
            = manually_augment_images(augmentation,
                                      train_images,
                                      train_labels,
                                      valid_images,
                                      valid_labels)
        
    train_images = torch.FloatTensor(train_images)
    valid_images = torch.FloatTensor(valid_images)
    train_labels = torch.LongTensor(train_labels)
    valid_labels = torch.LongTensor(valid_labels)
    if test_size > 0:
        test_images = torch.FloatTensor(test_images)
        test_labels = torch.LongTensor(test_labels)
    if examples_for_visualization:
        examples_for_visuals = []
        for label_class in range(num_classes):
            example_idx = torch.nonzero(torch.where(train_labels == label_class, 1, 0))[0][0]
            examples_for_visuals.append(np.transpose(train_images[example_idx].numpy(), (1, 2, 0)))
    else:
        examples_for_visuals = None
        
    train_dataset = torch.utils.data.TensorDataset(train_images,
                                                   train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_images,
                                                   valid_labels)
    if test_size > 0:
        test_dataset  = torch.utils.data.TensorDataset(test_images,
                                                       test_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size = batch_size,
                                               shuffle    = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size = batch_size,
                                               shuffle    = False)
    if test_size > 0:
        test_loader  = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size = batch_size,
                                                   shuffle    = False)
    else:
        test_loader  = None
    
    return train_loader, valid_loader, test_loader, examples_for_visuals

def visualize_images(dataset_name,
                     examples_over_time,
                     labels,
                     file_name):
    '''
    Visualize training examples from the first 5 label classes, 1 row per time point
    @param dataset_name: str, cifar10 or cifar100
    @param examples_from_classes: list of lists of np arrays, outer list over time, inner list over label classes, 
                                  1 example per class per time step
    @param labels: list of str, names of shifts at each time step for row label
    @param file_name: str, path to save visualization
    @return: None
    '''
    assert dataset_name in {'cifar10', 'cifar100'}
    num_time_steps = len(examples_over_time)
    assert len(labels) == num_time_steps
    fig, ax = plt.subplots(nrows   = num_time_steps,
                           ncols   = 5,
                           figsize = (2.5, num_time_steps/2),
                           dpi     = 1000)
    if dataset_name == 'cifar10':
        label_classes = [0, 1, 2, 3, 5]
        label_names   = ['plane', 'car', 'bird', 'cat', 'dog']
    else:
        label_classes = [9, 18, 2, 8, 12]
        label_names   = ['man-made', 'vehicle', 'flower', 'carnivore', 'mammal']
    for time_idx, label_class in product(range(num_time_steps), range(5)):
        # cifar is on 0-255 scale
        example_image = examples_over_time[time_idx][label_classes[label_class]].astype(int)
        ax[time_idx, label_class].imshow(example_image,
                                         interpolation = 'nearest')
        ax[time_idx, label_class].set_xticks([])
        ax[time_idx, label_class].set_yticks([])
        # make xaxis invisible
        ax[time_idx, label_class].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[time_idx, label_class].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[time_idx, label_class].tick_params(left=False, labelleft=False, right=False)
        #remove background patch (only needed for non-white background)
        ax[time_idx, label_class].patch.set_visible(False)
    for i in range(5):
        ax[0,i].set_title(label_names[i], fontsize=5)
    
    for time_idx in range(num_time_steps):
        ax[time_idx,0].set_ylabel(labels[time_idx], fontsize=5)
    
    plt.subplots_adjust(wspace = .05,
                        hspace = .05)
    fig.savefig(file_name)

def load_datasets_over_time(logger,
                            dataset_name,
                            shift_sequence,
                            source_sample_size,
                            target_sample_sizes,
                            final_target_test_size,
                            seed             = 1007,
                            visualize_shifts = False,
                            output_dir       = None,
                            test_set_for_all = False):
    '''
    Create data loaders for each time step
    @param logger: logger, for INFO messages
    @param dataset_name: str, cifar10 or cifar100
    @param shift_sequence: str, shift at each time step separated by colon, 
                           each time step is comma-separated combination of 
                           corruption, rotation, label_flip, label_shift, rotation_cond, recolor, recolor_cond
    @param source_sample_size: int, number of training/validation samples in source domain
    @param target_sample_sizes: list of int, number of training/validation samples for each target domain
    @param final_target_test_size: int, number of test samples in final domain
    @param seed: int, seed for data generation
    @param visualize_shifts: bool, whether to visualize images from shift sequence
    @param output_dir: str, directory to place plot of visualized images
    @param test_set_for_all: bool, whether to include a test set of size final_target_test_size at each target time step,
                             defaults to only a test set at the final time step if False
    @return: list of dict mapping str to torch DataLoaders, list over time, data split to loader
    '''
    assert dataset_name in {'cifar10', 'cifar100'}
    assert source_sample_size > 0
    assert np.all(np.array(target_sample_sizes) > 0)
    assert final_target_test_size > 0
    if visualize_shifts:
        assert output_dir is not None
    
    start_time = time.time()
    if dataset_name == 'cifar10':
        num_classes = 10
    else:
        num_classes = 20
    source_subpops = np.array([0.5, 0.5, 0, 0, 0])
    source_train_loader, source_valid_loader, _, source_examples \
        = load_corrupted_dataset(logger,
                                 dataset_name    = dataset_name,
                                 corruption_name = [],
                                 rotation_angle  = np.zeros(num_classes),
                                 label_flip      = [],
                                 label_probs     = np.ones(num_classes)/num_classes,
                                 recolorings     = np.zeros((num_classes, 3)),
                                 subpops         = source_subpops,
                                 sample_size     = source_sample_size,
                                 seed            = seed,
                                 augmentation    = 'random',
                                 test_size       = 0,
                                 examples_for_visualization = visualize_shifts)
    data_loaders_over_time = [{'train': source_train_loader,
                               'valid': source_valid_loader}]
    if visualize_shifts:
        examples_to_visualize = [source_examples]
        visual_labels         = ['Time 0\nsource']
    
    np.random.seed(seed)
    target_shifts         = shift_sequence.split(':')
    target_seeds          = np.random.randint(10000, size=len(target_shifts))
    target_corruptions    = []
    target_rotation_angle = np.zeros(num_classes)
    target_label_flip     = []
    target_recolorings    = np.zeros((num_classes, 3)) # RGB channel tinting
    target_label_probs    = np.ones(num_classes)/num_classes
    target_subpops        = source_subpops
    for target_idx in range(len(target_shifts)):
        if 'corruption' in target_shifts[target_idx]:
            target_corruptions.append('discrete_noise')
        if 'rotation_cond' in target_shifts[target_idx]:
            for label_class in range(num_classes):
                if label_class % 2 == 0:
                    target_rotation_angle[label_class] += 30
                else:
                    target_rotation_angle[label_class] -= 30
        elif 'rotation' in target_shifts[target_idx]:
            target_rotation_angle += 30
        if 'label_flip' in target_shifts[target_idx]:
            if len(target_label_flip) == 0:
                target_label_flip.append(1)
            elif target_label_flip[-1] == 1:
                target_label_flip.append(2)
            else:
                target_label_flip.append(1)
        if 'label_shift' in target_shifts[target_idx]:
            for label_class in range(num_classes):
                if label_class % 2 == 0:
                    target_label_probs[label_class] -= .3/num_classes
                else:
                    target_label_probs[label_class] += .3/num_classes
            assert np.all(target_label_probs >= 0)
            assert np.isclose(np.sum(target_label_probs), 1)
        if 'recolor_cond' in target_shifts[target_idx]:
            for label_class in range(num_classes):
                if label_class % 3 == 0:
                    target_recolorings[label_class,0] += 30
                elif label_class % 3 == 1:
                    target_recolorings[label_class,1] += 30
                else:
                    target_recolorings[label_class,2] += 30
        elif 'recolor' in target_shifts[target_idx]:
            target_recolorings[:,0] += 30
        if 'subpop' in target_shifts[target_idx]:
            assert dataset_name == 'cifar100', 'Sub-population shifts only supported for CIFAR-100'
            assert target_subpops[-1] == 0, 'No more sub-population shifts possible'
            if target_subpops[0] == 0.5:
                target_subpops = np.array([1./3, 1./3, 1./3, 0, 0])
            elif target_subpops[0] == 1./3:
                target_subpops = np.array([0, 1./3, 1./3, 1./3, 0])
            else:
                target_subpops = np.array([0, 0, 1./3, 1./3, 1./3])
        
        if target_idx == len(target_shifts) - 1 or test_set_for_all:
            time_step_sample_size = target_sample_sizes[target_idx] + final_target_test_size
            time_step_test_size   = final_target_test_size
        else:
            time_step_sample_size = target_sample_sizes[target_idx]
            time_step_test_size   = 0
        target_train_loader, target_valid_loader, target_test_loader, target_examples \
            = load_corrupted_dataset(logger,
                                     dataset_name    = dataset_name,
                                     corruption_name = target_corruptions,
                                     rotation_angle  = target_rotation_angle,
                                     label_flip      = target_label_flip,
                                     label_probs     = target_label_probs,
                                     recolorings     = target_recolorings,
                                     subpops         = target_subpops,
                                     sample_size     = time_step_sample_size,
                                     seed            = target_seeds[target_idx],
                                     augmentation    = 'random',
                                     test_size       = time_step_test_size,
                                     examples_for_visualization = visualize_shifts)
        if target_idx == len(target_shifts) - 1 or test_set_for_all:
            data_loaders_over_time.append({'train': target_train_loader,
                                           'valid': target_valid_loader,
                                           'test' : target_test_loader})
        else:
            data_loaders_over_time.append({'train': target_train_loader,
                                           'valid': target_valid_loader})
        
        if visualize_shifts:
            examples_to_visualize.append(target_examples)
            if target_shifts[target_idx] == 'rotation_cond':
                visual_labels.append('Time ' + str(target_idx + 1) + '\ncond rotat')
            else:
                visual_labels.append('Time ' + str(target_idx + 1) + '\n' + target_shifts[target_idx].replace('_', ' '))
    if visualize_shifts:
        visualize_images(dataset_name,
                         examples_to_visualize,
                         visual_labels,
                         output_dir + 'image_examples.pdf')
    return data_loaders_over_time

def combine_data_loaders_over_time(data_loaders_over_time,
                                   time_weights = np.ones(1)):
    '''
    Combine data loaders over time into a single data loader per data split
    @param data_loaders_over_time: list of dicts mapping str to torch DataLoader, list over time periods, data split to loader
    @param time_weights: np array of floats, weight for each time step, same weight given to all samples,
                         equal weights can be specified by len-1 array of 1
    @return: dict mapping str to torch DataLoader, data split to loader
    '''
    num_time_steps = len(data_loaders_over_time)
    if len(time_weights) == 1:
        assert time_weights[0] == 1
        time_weights = np.ones(num_time_steps)
    data_splits    = list(data_loaders_over_time[0].keys())
    data_tensors_over_time = {data_split: [[tensor] for tensor in data_loaders_over_time[0][data_split].dataset.tensors]
                              for data_split in data_splits}
    for data_split in data_splits:
        split_t0_num_samples = data_tensors_over_time[data_split][0][0].shape[0]
        data_tensors_over_time[data_split].append([time_weights[0] * torch.ones(split_t0_num_samples)])
    for time_idx in range(1, num_time_steps):
        # check same data splits at each time step
        assert set(data_loaders_over_time[time_idx].keys()) == set(data_splits)
        for data_split in data_splits:
            # check data loader has same number of tensors
            time_split_tensors = data_loaders_over_time[time_idx][data_split].dataset.tensors
            assert len(time_split_tensors) == len(data_tensors_over_time[data_split]) - 1
            for tensor_idx in range(len(time_split_tensors)):
                data_tensors_over_time[data_split][tensor_idx].append(time_split_tensors[tensor_idx])
            time_num_samples = time_split_tensors[0].shape[0]
            data_tensors_over_time[data_split][-1].append(time_weights[time_idx] * torch.ones(time_num_samples))
    combined_data_tensors = {data_split: [torch.cat(tensors) for tensors in data_tensors_over_time[data_split]]
                             for data_split in data_splits}
    combined_data_sets    = {data_split: torch.utils.data.TensorDataset(*combined_data_tensors[data_split])
                             for data_split in data_splits}
    batch_size = data_loaders_over_time[0][data_splits[0]].batch_size
    combined_data_loaders = {data_split: torch.utils.data.DataLoader(combined_data_sets[data_split],
                                                                     batch_size = batch_size,
                                                                     shuffle    = True if data_split == 'train' else False)
                             for data_split in data_splits}
    return combined_data_loaders