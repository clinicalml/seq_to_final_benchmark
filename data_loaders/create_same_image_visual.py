import sys

from copy import deepcopy
from os.path import join, dirname, abspath

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from load_image_data_with_shifts import (
    apply_uniform_random_discrete_noise,
    apply_rotation,
    apply_recoloring
)

sys.path.append(dirname(dirname(abspath(__file__))))
import config

np.random.seed(1007)

train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # means and stds from https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
                torchvision.transforms.Normalize(255*[0.485, 0.456, 0.406],
                                                 255*[0.229, 0.224, 0.225])])

orig_train_dataset   = torchvision.datasets.CIFAR10(root      = config.data_dir + 'cifar10', 
                                                    train     = True,
                                                    download  = True,
                                                    transform = train_transforms)
orig_train_images = torch.FloatTensor(orig_train_dataset.data).permute((0,3,1,2))
two_train_images  = orig_train_images[1:3]

# apply corruptions
two_corrupted_images = deepcopy(two_train_images)
first_corruption_images  = [np.transpose(two_corrupted_images[0].numpy(), (1, 2, 0))]
second_corruption_images = [np.transpose(two_corrupted_images[1].numpy(), (1, 2, 0))]
scale = 3
corruption_degrees = np.random.randint(int(-1 * scale), int(scale), size = two_train_images.shape)
for i in range(4):
    two_corrupted_images = np.clip(two_corrupted_images + corruption_degrees, 0, 255).float()
    first_corruption_images.append(np.transpose(two_corrupted_images[0].numpy(), (1, 2, 0)))
    second_corruption_images.append(np.transpose(two_corrupted_images[1].numpy(), (1, 2, 0)))

# apply rotations
first_rotation_images  = [np.transpose(two_train_images[0].numpy(), (1, 2, 0))]
second_rotation_images = [np.transpose(two_train_images[1].numpy(), (1, 2, 0))]
for i in range(4):
    two_rotated_images = apply_rotation(two_train_images, 30*(i+1))
    first_rotation_images.append(np.transpose(two_rotated_images[0].numpy(), (1, 2, 0)))
    second_rotation_images.append(np.transpose(two_rotated_images[1].numpy(), (1, 2, 0)))

# apply recolorings
recoloring_degree     = np.zeros(3)
recoloring_degree[0]  = 30
first_recolor_images  = [np.transpose(two_train_images[0].numpy(), (1, 2, 0))]
second_recolor_images = [np.transpose(two_train_images[1].numpy(), (1, 2, 0))]
for i in range(4):
    two_recolored_images  = apply_recoloring(two_train_images, recoloring_degree*(i+1))
    first_recolor_images.append(np.transpose(two_recolored_images[0].numpy(), (1, 2, 0)))
    second_recolor_images.append(np.transpose(two_recolored_images[1].numpy(), (1, 2, 0)))

example_images = [first_corruption_images,
                  second_corruption_images,
                  first_rotation_images,
                  second_rotation_images,
                  first_recolor_images,
                  second_recolor_images]
ylabels = ['Corruption\nEx. 1', 'Corruption\nEx. 2', 'Rotation\nEx. 1', 'Rotation\nEx. 2', 'Recoloring\nEx. 1', 'Recoloring\nEx. 2']
fig, ax = plt.subplots(nrows   = 6,
                       ncols   = 5,
                       figsize = (2.5, 3),
                       dpi     = 1000)
for label_class in range(6):
    for example_idx in range(5):
        ax[label_class, example_idx].imshow(example_images[label_class][example_idx].astype(int),
                                            interpolation = 'nearest')
        ax[label_class, example_idx].set_xticks([])
        ax[label_class, example_idx].set_yticks([])
        # make xaxis invisible
        ax[label_class, example_idx].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[label_class, example_idx].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[label_class, example_idx].tick_params(left=False, labelleft=False, right=False)
        #remove background patch (only needed for non-white background)
        ax[label_class, example_idx].patch.set_visible(False)
    ax[label_class, 0].set_ylabel(ylabels[label_class], fontsize=5)
plt.subplots_adjust(wspace = .05,
                    hspace = .05)
fig_filename = config.output_dir + 'image_visualization/two_examples_repeat_CRT.pdf'
fig.savefig(fig_filename)