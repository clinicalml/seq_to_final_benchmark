# Seq-to-Final: A Benchmark for Tuning from Sequential Distributions to a Final Time Point

Distribution shift over time occurs in many settings. Leveraging historical data is necessary to learn a model for the last time point when limited data is available in the final period, yet few methods have been developed specifically for this purpose. We construct this benchmark Seq-to-Final so that methods for leveraging sequential historical data can be evaluated against existing methods for handling distribution shift. This benchmark focuses on image classification tasks using CIFAR-10 and CIFAR-100 as the base images for the synthetic sequences. There are benchmarks for defining the types of shifts at each step in the sequence:

- Input-level shifts: corruption, rotation, recoloring
- Intermediate-level shifts: conditional rotation, sub-population shift
- Output-level shifts: label flips

Users can also define other building blocks or add other datasets to be used as the base dataset. The sequences can be constructed by specifying the shifts and sample size at each time step. A test set is created at the final step for evaluation. The benchmark also includes the Portraits dataset so results on synthetic sequences can be compared to results on this real-world sequence. Users can also add other real-world datasets.

The benchmark includes 3 classes of methods for comparison:
- Methods that learn from all data without adapting to the final period
- Methods that learn from historical data with no regard to the sequential nature and then adapt to the final period
- Methods that leverage the sequential nature of historical data when tailoring a model to the final period.

This benchmark also includes code for creating several visualizations:
- Linear interpolation paths plot the test accuracies of models where the weights are linearly interpolated from the initialization at the last step to the final model at the last step. These plots can show whether a method is good at leveraging historical data to find an initialization that is within the same loss basin as the final model.
- We also include two methods for plotting the models over time. The first projects the weights onto directions that reflect historical shift and limited final sample size. The second uses singular vector canonical correlation analysis to compare the intermediate outputs from each method to the oracle (a model that is learned from a large amount of data at the final step). Both of these methods are limited by their dependence on the oracle model, which is somewhat arbitrary due to random initialization.

In our paper, we demonstrate this benchmark by comparing the 3 classes of methods on 5 synthetic sequences and the real-world Portraits sequence. Our results suggest that, for the sequences in our benchmark, methods that disregard the sequential structure and adapt to the final time point tend to perform well. The approaches we evaluate that leverage the sequential nature do not offer any improvement. We hope that this benchmark will inspire the development of new algorithms that are better at leveraging sequential historical data or a deeper understanding of why methods that disregard the sequential nature are able to perform well.

## Setting up datasets

Run `conda create --prefix NEWENV --file fine_tuning_env.yml` to set up the conda environment.

The CIFAR-10 and CIFAR-100 datasets will be automatically downloaded when running the benchmark, so no preparation is needed for the synthetic sequences. To prepare the portraits dataset,
1. Download from https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0
2. Move `create_yearbook_folders.sh` to the unzipped folder containing `M` and `F` sub-directories and run it there. This will create two label folders with images within the folder for each time period.
3. The experiment scripts will be able to read in the data by calling the `load_yearbook_data` method in `load_yearbook_data.py`.

To recreate the figures in the paper, run these scripts in `data_loaders`:
1. Figure 2: `python3 create_same_image_visual.py`
2. Figures 3-4: `python3 create_subpop_visual.py`
3. Figures 5-7: `./create_all_image_examples.sh`

To add other datasets for constructing synthetic sequences,
1. Create a function like `select_cifar10_samples` or `select_cifar100_samples` in `load_image_data_with_shifts.py` that splits the dataset into training, validation, and test.
2. Modify `load_corrupted_dataset` to load images for the new dataset.

To add other real-world datasets, create functions similar to those in `load_yearbook_data.py`.

## Running benchmark methods

There are three classes of methods in our benchmark:
1. Methods that learn a single model for all time steps
2. Methods that learn a model from all historical data and then adapt to the final step
3. Methods that leverage the sequential nature of historical data and target the final step

The first two classes and the sequential fine-tuning sub-class of the third class are implemented in `run_erm_fine_tuning.py`. The joint model sub-class of the third class is implemented in `run_joint_model_learning.py`

### Running methods that learn a single model or are based on fine-tuning 

To run one of these methods, run `python3 run_erm_fine_tuning.py` in the `benchmark` directory with the following arugments:
- `--dataset`: `cifar10`, `cifar100`, or `portraits`
- `--model_type`: `densenet`, `resnet`, or `convnet`
- `--num_blocks`: number of blocks in each of the model architectures above
- `--shift_sequence`: shifts separated by commas, time steps separated by colons, choices: `corruption`, `rotation`, `label_flip`, `label_shift`, `recolor`, `rotation_cond`, `recolor_cond`, `subpop`
- `--source_sample_size`: # samples for training/validation in first domain
- `--target_sample_size`: # samples for training/validation in subsequent domains
- `--target_test_size`: # samples in test set for final domain
- `--target_sample_size_seq`: colon-separated sequence of sample sizes for subsequent domains if different per time step
- `--gpu_num`: which GPU device to use
- `--seed`: for random generation
- `--method`: `erm_final`, `erm_all`, `erm_all_weighted`, `erm_fine_tune`, `sequential_fine_tune`, `dro_all`, `irm_all`, `dro_fine_tune`, or `irm_fine_tune`. `erm_final` is used for the baseline or the oracle that only trains on data from the final step. `erm_all`, `irm_all`, `dro_all`, and `erm_all_weighted` learn a single model with the corresponding method on data from all steps. `erm_all_weighted` puts more weight on later time steps in the training objective. `erm_fine_tune`, `dro_fine_tune`, and `irm_fine_tune` fit a model with the corresponding method on all historical data and then fine-tunes at the last step. `sequential_fine_tune` fits a model on the source distribution and then fine-tunes at each step.
- `--fine_tune_layers`: specify the combination of layers that is fine-tuned. Multiple combinations of layers can be tuned in a sequence. These are the options:
    - `all` (default): tune all layers
    - `search`: try all combinations of layers at each step
    - `surgical_rgn` and `surgical_snr`: order layers by relative gradient norm and signal-to-noise ratio from surgical fine-tuning paper and try tuning different numbers of layers
    - Dash-separated list of comma-separated list of layers: Comma-separated list of layers are tuned together. Dash-separated list corresponds to tuning first layer set in list until convergence, then tuning second layer set etc.
    - Colon-separated list of different options at each time step in sequential fine-tuning
    - `linear_probe_fine_tune`: shortcut for `fc-all`
    - `gradual_unfreeze_first_to_last`: shortcut for `conv1-conv1,layer1-conv1,layer1,layer2-conv1,layer1,layer2,layer3-conv1,layer1,layer2,layer3,layer4-all`
    - `gradual_unfreeze_last_to_first`: shortcut for `fc-fc,layer4-fc,layer4,layer3-fc,layer4,layer3,layer2-fc,layer4,layer3,layer2,layer1-all`
- `--start_with_imagenet`: specify to start a 4-block ResNet with the pre-trained ImageNet model instead of starting from scratch
- `--learning_rate_decay`: `none`, `linear`, or `exponential` for how to decay the learning rate over time steps in sequential fine-tuning
- `--fine_tune_mode`: Variant of fine-tuning to run. Some of these options are only for debugging the joint model as in Table 14 of the paper.
  - `standard`: fine-tune the layers specified in `--fine_tune_layers`
    `side_tune`: use side modules for blocks
  - `low_rank_adapt`: use low-rank adapters for blocks
  - `standard_using_joint`: fine-tune all layers by freezing previous time steps and fine-tuning the last time step in a joint model with no adjacent regularization, use only for debugging to see if ablated joint model matches ERM final
  - `standard_using_joint_with_init`: same as above but also initialize last time step with previous parameters, use only for debugging to see if ablated joint model matches standard fine-tuning
  - `standard_using_joint_with_prev_reg`: same as above but with regularization towards previous weights instead, use only to see if ablated joint model matches standard with previous regularization and ablated initialization
  - `standard_using_joint_with_init_and_prev_reg`: also to see if ablated joint model matches standard with previous regularization.
- `--adapter_rank`: specify rank of each adapter pair A_o x A_i
- `--adapter_mode`: specify whether to multiply or add low-rank adapters
- `--side_layer_sizes`: specify comma-separated list of sizes of intermediate layers in side modules, number of layers is 1 more than list length, default side module size is 1 convolution layer, can also specify `block` to use side modules that are the same size as the original modules
- `--ablate_init`: specify to initialize fine-tuned model from scratch
- `--all_layers_efficient`: specify to make side modules or low-rank adapters for non-block layers (input convolution and output fully connected)
- `--test_all_steps`: specify to record test accuracy at all steps of sequential fine-tuning

### Running the joint model

To fit a variant of the joint model, run `python run_joint_model_learning.py` in the `benchmark` directory with the following arguments:

- `--dataset`: `cifar10`, `cifar100`, or `portraits`
- `--model_type`: `densenet`, `resnet`, or `convnet`
- `--num_blocks`: number of blocks in each of the model architectures above
- `--shift_sequence`: shifts separated by commas, time steps separated by colons, choices: `corruption`, `rotation`, `label_flip`, `label_shift`, `recolor`, `rotation_cond`, `recolor_cond`, `subpop`
- `--source_sample_size`: # samples for training/validation in first domain
- `--target_sample_size`: # samples for training/validation in subsequent domains
- `--target_test_size`: # samples in test set for final domain
- `--target_sample_size_seq`: colon-separated sequence of sample sizes for subsequent domains if different per time step
- `--gpu_num`: which GPU device to use
- `--seed`: for random generation
- `--visualize_images`: specify to plot examples from 5 classes at each time step
- `--separate_layers`: specify `all` (default) for all layers to be separate at each time step, `search` to find the best combination of layers to share across all time steps, or a comma-separated list of layers that are new at each time step with time steps separated by colons. Example: `conv1,layer1:fc` means the `conv1` and `layer1` layers are new at the 2nd time step and the `fc` layer is new at the 3rd time step.
- `--resume`: specify to resume fitting for best hyperparameters
- `--weight_l2_by_time`: specify to weight L2 regularization by 1/(t+1) and adjacent L2 regularization by 1/(T-t)
- `--loss_samples`: `all` (default) to use all previous samples, `nearest neighbors` to use nearest neighbor path among previous samples, `optimal matching` to use optimal matching that minimizes total distance at each time step
- `--loss_weight_increasing`: specify to weight loss at each time step by 1/(T - t + 1) instead of alpha at T and 1 elsewhere
- `--adjacent_reg_type`: specify how to implement regularization on difference between parameters at adjacent time steps. Options: `none`, `l2`, `l2_fisher`, `l2_high`, `l2_very_high`, `l1`
- `--start_with_imagenet`: specify to initialize model with ImageNet pretrained weights
- `--mode`: specify `separate` to have separate modules at different time steps, `side_tune` to add output from a side module instead of using new blocks at each time step, or `low_rank_adapt` to form adapter pairs A_o x A_i. Original weights will be multiplied or added by sequence of low-rank factors.
- `--adapter_rank`: specify rank of each adapter pair A_o x A_i
- `--adapter_mode`: specify whether to multiply or add low-rank adapters
- `--side_layer_sizes`: specify comma-separated list of sizes of intermediate layers in side modules, number of layers is 1 more than list length, default side module size is 1 convolution layer, can also specify `block` to use side modules that are the same size as the original modules 
- `--ablate`: specify to ablate side modules or low-rank adapters, i.e. set joint model to be the same at all time steps and replicate ERM all
- `--sweep_efficient_reg`: specify to tune regularization on side modules or low-rank adapters separately and plot validation accuracy for each regularization constant
- `--all_layers_efficient`: specify to use side modules or low-rank adapters also in non-block layers, e.g. input convolution and output fully connected layers. These are maximum possible rank that is still efficient, so may be smaller than `--adapter_rank`.

To examine the effect of varying adjacent regularization on the norm of the parameter changes and on the validation accuracy at the final time step, run `python3 plot_effect_of_adjacent_reg_on_joint_model.py` with the following arguments:

- `--dataset`: `cifar10` or `cifar100`
- `--shift_sequence_str`: name of directory for shift sequence, e.g. `shift6000:corruption4000`
- `--plot`: `param_norm` for the norm of the parameter changes or `val_acc` for the validation accuracy
- `--gpu_num`: which GPU device to use if plotting norms
- `--seed`: for random generation
- `--include_very_high`: use this flag to include the very high adjacent regularization settings in the plot

### Summarizing benchmark metrics

Run `python3 summarize_metric_csvs.py` to create tables that show the performance of all experiments that have been run. The tables will have 1 row per model architecture and 1 column per method. If multiple seeds have been run, each entry will be the mean across seeds. The mean and standard deviation across all architectures will also be included in each column.

To recreate Table 8 in the paper, run `./run_convnet_RCL.sh`, `./run_densenet_RCL.sh`, `./run_resnet_RCL.sh`, and `python3 summarize_metric_csvs.py`. Tables 7-13 can be recreated by running similar scripts and then summarizing the metrics into tables.

## Creating visualizations

We include code for creating three types of plots:
1. Linear interpolation paths: To visualize the loss landscape from the initial to the final model at the last time step for each method, we linearly interpolate the model weights and plot the accuracy of the interpolated model on the test set at the last time step against the interpolation factor.
2. Projection of weights onto directions that reflect historical shift and limited final sample size: The historical shift direction is defined by the difference between the weights from running ERM on all historical data and the oracle. The limited final sample size direction is defined by the difference between the weights from running ERM on the limited final data and the oracle.
3. CCA coefficients: This visualization plots the mean of the top SVCCA coefficients explaining 50% and 90% of the variance when comparing each model to the oracle.

All three of these plots can be created by running `python3 visualize_models.py` in the `benchmark` directory with the following arguments:

- `--dataset`: `cifar10`, `cifar100`, or `portraits`. The latter is only supported for the linear interpolation visualiation since the other two plots depend on an oracle
- `--model_type`: `convnet`, `densenet`, `resnet`, or `merge`. The latter can only be run for the linear interpolation visualization after the visuals have already been created for all 3 architectures
- `--num_blocks`: number of blocks in architecture
- `--shift_sequence`: specified in the same way as above for running the models
- `--representation`: `linear_interpolate` for visualization 1, `weight_proj` for visualization 2, or `cca` for visualization 3
- `--model_dirs`: colon-separated list of directory names for models to plot
- `--model_names`: colon-separated list of model names for legend
- `--plot_title`: title for plots

To recreate Figure 1 and Figures 10-17, run `visualize_models.sh` in the `experiment_scripts` directory.

## Miscellaneous

### Quantifying shift

To quantify the amount of shift from each time step to the final time step, run `python3 quantify_shifts.py` in the `benchmark` directory with the `--dataset`, `--shift_sequence`, `--source_sample_size`, `--target_sample_size_seq`, and `--target_test_size` arguments.

To reproduce the metrics in Tables 4 and 5, run `./quantify_shifts.sh` in the `experiment_scripts` directory.

### Model architectures

To compute the number of parameters in each model architecture and reproduce Table 6, run `python3 compute_num_params_in_different_model_sizes.py` in the `model_classes` directory.