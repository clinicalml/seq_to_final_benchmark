import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import config

label_names = ['aquatic\nmammals', 'fish', 'flowers', 'food\ncontainers', 'fruits &\nveggies', 'electric\ndevices',
               'furniture', 'insects', 'carnivores', 'man-made\noutdoor', 'natural\noutdoor', 'omnivores &\nherbivores', 
               'medium\nmammals', 'invertebrates', 'people', 'reptiles', 'small\nmammals', 'trees', 'vehicles 1', 'vehicles 2']
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

train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # means and stds from https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
                torchvision.transforms.Normalize(255*[0.485, 0.456, 0.406],
                                                 255*[0.229, 0.224, 0.225])])

orig_train_dataset   = torchvision.datasets.CIFAR100(root      = config.data_dir + 'cifar100', 
                                                     train     = True,
                                                     download  = True,
                                                     transform = train_transforms)
orig_train_images = torch.FloatTensor(orig_train_dataset.data).permute((0,3,1,2))
orig_train_labels = torch.LongTensor(orig_train_dataset.targets)

example_images = []
for label_class in range(20):
    label_images = []
    for subpop_idx in range(5):
        subpop_label = orig_labels.index(subpop_labels[label_class][subpop_idx])
        subpop_first_index = torch.nonzero(torch.where(orig_train_labels == subpop_label, 1, 0), as_tuple = True)[0][0]
        label_images.append(np.transpose(orig_train_images[subpop_first_index].numpy(), (1, 2, 0)))
    example_images.append(label_images)
    
# plot first 10 label classes
fig, ax = plt.subplots(nrows   = 10,
                       ncols   = 5,
                       figsize = (2.5, 5),
                       dpi     = 1000)
for label_class in range(10):
    for subpop_idx in range(5):
        ax[label_class, subpop_idx].imshow(example_images[label_class][subpop_idx].astype(int),
                                           interpolation = 'nearest')
        ax[label_class, subpop_idx].set_xticks([])
        ax[label_class, subpop_idx].set_yticks([])
        # make xaxis invisible
        ax[label_class, subpop_idx].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[label_class, subpop_idx].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[label_class, subpop_idx].tick_params(left=False, labelleft=False, right=False)
        #remove background patch (only needed for non-white background)
        ax[label_class, subpop_idx].patch.set_visible(False)
        subpop_title = subpop_labels[label_class][subpop_idx]
        if subpop_title == 'sweet pepper':
            subpop_title = 'pepper'
        ax[label_class, subpop_idx].set_title(subpop_title, fontsize=5, pad=0)
    ax[label_class, 0].set_ylabel(label_names[label_class], fontsize=5)
plt.subplots_adjust(wspace = .05,
                    hspace = .25)
fig_filename = '/afs/csail.mit.edu/group/clinicalml/users4/cji/image_visualization/cifar100_source6000:subpop4000:subpop6000:subpop4000_seed1007/subpops_ordered_1.pdf'
fig.savefig(fig_filename)

# plot second 10 label classes
plt.clf()
fig, ax = plt.subplots(nrows   = 10,
                       ncols   = 5,
                       figsize = (2.5, 5),
                       dpi     = 1000)
for label_class in range(10):
    for subpop_idx in range(5):
        ax[label_class, subpop_idx].imshow(example_images[label_class+10][subpop_idx].astype(int),
                                           interpolation = 'nearest')
        ax[label_class, subpop_idx].set_xticks([])
        ax[label_class, subpop_idx].set_yticks([])
        # make xaxis invisible
        ax[label_class, subpop_idx].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[label_class, subpop_idx].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[label_class, subpop_idx].tick_params(left=False, labelleft=False, right=False)
        #remove background patch (only needed for non-white background)
        ax[label_class, subpop_idx].patch.set_visible(False)
        ax[label_class, subpop_idx].set_title(subpop_labels[label_class+10][subpop_idx], fontsize=5, pad=0)
    ax[label_class, 0].set_ylabel(label_names[label_class+10], fontsize=5)
plt.subplots_adjust(wspace = .05,
                    hspace = .25)
fig_filename = '/afs/csail.mit.edu/group/clinicalml/users4/cji/image_visualization/cifar100_source6000:subpop4000:subpop6000:subpop4000_seed1007/subpops_ordered_2.pdf'
fig.savefig(fig_filename)