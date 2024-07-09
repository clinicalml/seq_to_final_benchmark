import os
import sys
import time
import json

from copy import deepcopy
from collections import defaultdict
from itertools import product
from functools import partial
from os.path import dirname, abspath, join
from datetime import datetime

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(dirname(dirname(abspath(__file__))))
import config

sys.path.append(join(dirname(dirname(abspath(__file__))), 'utils'))
from logging_utils import set_up_logger
from pytorch_model_utils import save_state_dict_to_gz, load_state_dict_from_gz

def plot_losses_over_epochs(train_losses,
                            valid_losses,
                            plot_filename,
                            logger,
                            plot_title = None,
                            accuracy   = False):
    '''
    Plot training and validation losses over epochs
    @param train_losses: list of floats, training losses over epochs
    @param valid_losses: list of floats, validation losses over epochs
    @param plot_filename: str, path to save plot
    @param logger: logger, for INFO messages
    @param plot_title: str, plot title
    @param accuracy: bool, if True, then assumes function is plotting accuracies instead
    @return: None
    '''
    start_time = time.time()
    plt.clf()
    plt.rc('font', 
           size = 14)
    fig, ax = plt.subplots(figsize = (6.4, 4.8))
    ax.plot(train_losses,
            label = 'Train')
    ax.plot(valid_losses,
            label = 'Valid')
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

def compute_loss(model,
                 loader,
                 return_num_samples = False):
    '''
    Compute cross entropy loss
    @param model: PyTorch model, __call__ must output predictions for samples 
    @param loader: torch DataLoader, contains FloatTensors of features and outcomes
    @param return_num_samples: bool, whether to also return number of samples in loader
    @return: 1. float, average loss
             2. int, number of samples if return_num_samples is True
    '''
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'none')
    total_loss  = 0.
    num_samples = 0
    model.eval()
    for batch_idx, batch_samples in enumerate(loader):
        if len(batch_samples) == 3:
            batch_x, batch_y, batch_weight = batch_samples
            if torch.cuda.is_available():
                batch_weight = batch_weight.cuda()
        else:
            batch_x, batch_y = batch_samples
        if torch.cuda.is_available():
            batch_x  = batch_x.cuda()
            batch_y  = batch_y.cuda()
        with torch.no_grad():
            outputs  = model(batch_x)
            loss     = loss_fn(outputs, batch_y)
        if len(batch_samples) == 3:
            loss    *= batch_weight
        loss         = torch.sum(loss)
        if torch.cuda.is_available():
            loss     = float(loss.detach().cpu().numpy())
        else:
            loss     = float(loss.detach().numpy())
        batch_size   = len(batch_y)
        total_loss  += loss
        num_samples += batch_size
    model.train()
    if return_num_samples:
        return total_loss/num_samples, num_samples
    return total_loss/num_samples

def compute_average_loss(model,
                         loaders):
    '''
    Compute average loss across all samples
    @param model: PyTorch model, __call__ must output predictions for samples 
    @param loaders: list of torch DataLoaders, each contains FloatTensors of features and outcomes
    @return: float, average loss
    '''
    total_num_samples = 0.
    total_loss        = 0.
    for loader in loaders:
        avg_loss, num_samples = compute_loss(model,
                                             loader,
                                             return_num_samples = True)
        total_num_samples    += num_samples
        total_loss           += avg_loss * num_samples
    return total_loss/total_num_samples
    
def compute_accuracy(model,
                     loader,
                     return_num_samples = False):
    '''
    Compute 0-1 accuracy
    @param model: block_model, model to fit
    @param loader: torch DataLoader, contains FloatTensors of features and outcomes
    @param return_num_samples: bool, whether to also return number of samples in loader
    @return: 1. float, average accuracy
             2. int, number of samples if return_num_samples is True
    '''
    num_correct = 0
    num_samples = 0
    model.eval()
    for batch_idx, batch_samples in enumerate(loader):
        batch_x = batch_samples[0]
        batch_y = batch_samples[1]
        if torch.cuda.is_available():
            batch_x  = batch_x.cuda()
        with torch.no_grad():
            outputs  = torch.argmax(model(batch_x), dim = 1)
        if torch.cuda.is_available():
            outputs  = outputs.detach().cpu().numpy()
        else:
            outputs  = outputs.detach().numpy()
        batch_y      = batch_y.detach().numpy()
        num_correct += np.sum(np.where(outputs.astype(int) == batch_y.astype(int), 1, 0))
        batch_size   = len(batch_y)
        num_samples += batch_size
    model.train()
    if return_num_samples:
        return num_correct/float(num_samples), num_samples
    return num_correct/float(num_samples)

def compute_average_accuracy(model,
                             loaders):
    '''
    Compute average accuracy across all samples
    @param model: PyTorch model, __call__ must output predictions for samples 
    @param loaders: list of torch DataLoaders, each contains FloatTensors of features and outcomes
    @return: float, average accuracy
    '''
    total_num_samples = 0.
    total_num_correct = 0.
    for loader in loaders:
        avg_accuracy, num_samples = compute_accuracy(model,
                                                     loader,
                                                     return_num_samples = True)
        total_num_samples        += num_samples
        total_num_correct        += avg_accuracy * num_samples
    return total_num_correct/total_num_samples

def get_predictions(model,
                    loader):
    '''
    Get predictions for all samples in loader
    @param model: block_model, model to fit
    @param loader: torch DataLoader, contains FloatTensors of features and outcomes
    @return: np array of ints, prediction classes
    '''
    batch_preds = []
    model.eval()
    for batch_idx, batch_samples in enumerate(loader):
        batch_x = batch_samples[0]
        if torch.cuda.is_available():
            batch_x  = batch_x.cuda()
        with torch.no_grad():
            outputs  = torch.argmax(model(batch_x), dim = 1)
        if torch.cuda.is_available():
            outputs  = outputs.detach().cpu().numpy()
        else:
            outputs  = outputs.detach().numpy()
        batch_preds.append(outputs)
    model.train()
    return np.concatenate(batch_preds, axis = None).astype(int)

def set_parameters_to_update(model,
                             layers_to_tune,
                             logger):
    '''
    Set whether parameter requires gradient
    @param model: block_model
    @param layers_to_tune: str, which layers to tune, 
                           options: all or combination of layer names, e.g. conv1,layer1,layer2,layer3,layer4,fc
    @param logger: logger, for INFO messages
    @return: list of torch parameters, those that require tuning
    '''
    if layers_to_tune != 'all':
        layers = set(layers_to_tune.split(','))
        assert layers.issubset(set(model.layer_names))
        for layer in model.layer_names:
            layer_idx = model.layer_names.index(layer)
            if layer in layers:
                for param in model.layers[layer_idx].parameters():
                    param.requires_grad = True
            else:
                for param in model.layers[layer_idx].parameters():
                    param.requires_grad = False

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            logger.info(name + ' will be updated')
        else:
            logger.info(name + ' will not be updated')
    
    return params_to_update

def load_imagenet_pretrained_resnet18(resnet18):
    '''
    Load imagenet pre-trained weights from pytorch for resnet18
    If there are 1000 output classes, pre-trained weights are also loaded for the fully connected layer
    @param resnet18: resnet_without_dropout, must be ResNet-18 architecture, will be modified and returned
    @return: resnet_without_dropout
    '''
    assert resnet18.num_blocks       == 4
    assert resnet18.layers_per_block == [2, 2, 2, 2]
    assert resnet18.input_dims       == [64, 64, 128, 256]
    assert resnet18.output_dims      == [64, 128, 256, 512]
    assert resnet18.strides          == [1, 2, 2, 2]
    resnet18_pretrained = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True)
    resnet18.conv1.load_state_dict(resnet18_pretrained.conv1.state_dict())
    resnet18.bn1.load_state_dict(resnet18_pretrained.bn1.state_dict())
    resnet18.layer1.load_state_dict(resnet18_pretrained.layer1.state_dict())
    resnet18.layer2.load_state_dict(resnet18_pretrained.layer2.state_dict())
    resnet18.layer3.load_state_dict(resnet18_pretrained.layer3.state_dict())
    resnet18.layer4.load_state_dict(resnet18_pretrained.layer4.state_dict())
    if resnet18.num_classes == 1000:
        resnet18.fc.load_state_dict(resnet18_pretrained.fc.state_dict())
    return resnet18

def compute_batch_loss(batch_samples,
                       model,
                       loss_fn,
                       dropout,
                       irm_weight = None):
    '''
    Compute loss for batch
    @param batch_samples: tuple of torch FloatTensors, features, labels, and optionally sample weights
    @param model: block_model
    @param loss_fn: torch loss that takes in prediction and label
    @param dropout: float
    @param irm_weight: torch FloatTensor, use for IRM loss, otherwise None
    @return: torch FloatTensor, loss for batch
    '''
    loss_train = 0.
    if len(batch_samples) == 3:
        batch_x, batch_y, batch_weight = batch_samples
        if torch.cuda.is_available():
            batch_weight = batch_weight.cuda()
    else:
        batch_x, batch_y = batch_samples
    if torch.cuda.is_available():
        batch_x  = batch_x.cuda()
        batch_y  = batch_y.cuda()

    batch_pred       = model(batch_x, dropout)
    if irm_weight is not None:
        batch_pred  *= irm_weight
    batch_train_loss = loss_fn(batch_pred, batch_y)
    if len(batch_samples) == 3:
        batch_train_loss *= batch_weight

    if irm_weight is not None:
        batch_idxs_shuffled = torch.randperm(len(batch_y))
        grad1 = torch.autograd.grad(batch_train_loss[batch_idxs_shuffled[0::2]].mean(), 
                                    irm_weight,
                                    create_graph = True)[0]
        grad2 = torch.autograd.grad(batch_train_loss[batch_idxs_shuffled[1::2]].mean(),
                                    irm_weight,
                                    create_graph = True)[0]
        loss_train += (grad1 * grad2).sum()

    loss_train += torch.mean(batch_train_loss)
    return loss_train
        
def fit_single_model(layers_to_tune,
                     model,
                     train_loader,
                     valid_loader,
                     learning_rate,
                     weight_decay,
                     dropout,
                     n_epochs,
                     fileheader,
                     logger,
                     save_model                  = True,
                     model_name                  = None,
                     resume_from_n_epochs        = 0,
                     early_stopping_n_epochs     = float('inf'),
                     regularization              = 'standard',
                     regularization_prev_models  = [],
                     regularization_fisher_infos = [],
                     remove_uncompressed_model   = True,
                     method                      = 'erm'):
    '''
    Fit a block model
    Save model, load from file if exists
    Save train and valid losses, load if exists, plot over epochs
    @param layers_to_tune: str, which layers to tune, 
                           options: all or combination of layer names, e.g. conv1,layer1,layer2,layer3,layer4,fc
    @param model: block_model, model to fit
    @param train_loader: torch DataLoader, contains training features, outcomes, and optionally sample weights in batches,
                         if using IRM or group DRO loss, this is a list of data loaders
    @param valid_loader: torch DataLoader, contains validation features and outcomes in batches
    @param learning_rate: float, learning rate
    @param weight_decay: float, regularization constant
    @param dropout: float, probability each output entry after each layer/block is zero'd during training
    @param n_epochs: int, number of epochs
    @param fileheader: str, start of paths to files, ends in _
    @param logger: logger, for INFO messages
    @param save_model: bool, whether to save model
    @param model_name: str, used for plot titles if provided
    @param resume_from_n_epochs: int, number of epochs to resume training from
    @param early_stopping_n_epochs: int or inf, number of epochs before stopping if validation accuracy no longer improves
    @param regularization: str, options: standard for L2 regularization towards 0,
                           previous for L2 regularization towards previous model,
                           fisher for L2 regularization towards previous models weighted by Fisher info,
                           previous_l1 for L1 regularization towards previous model
    @param regularization_prev_models: list of block_models, previous models to regularize towards,
                                       length 0 for standard, length 1 for previous, at least length 1 for Fisher,
                                       must be same architecture as model
    @param regularization_fisher_infos: list of dict mapping str to FloatTensor, list over previous models,
                                        dict maps layer name to Fisher info for each parameter, only used for Fisher
    @param remove_uncompressed_model: bool, whether to remove uncompressed .pt file when loading model, usually can set to True 
                                      unless expecting multiple threads to load the same model at the same time
    @param method: str, erm (empirical risk minimization), irm (invariant risk minimization), 
                   or dro (group distributionally robust optimization)
    @return: 1. block_model, fitted and set to best parameters
             2. validation accuracy
             3. block_model, from final epoch
             4. Adam optimizer, from final epoch
             5. int, epoch when fitting was stopped, may be less than n_epochs if early stopping applied
    '''
    assert regularization in {'standard', 'previous', 'fisher', 'previous_l1'}
    if regularization == 'standard':
        assert len(regularization_prev_models)  == 0
        assert len(regularization_fisher_infos) == 0
    elif regularization.startswith('previous'):
        assert len(regularization_prev_models)  == 1
        assert len(regularization_fisher_infos) == 0
    else:
        assert len(regularization_prev_models) >= 1
        assert len(regularization_prev_models) == len(regularization_fisher_infos)
    if regularization == 'previous_l1':
        reg_type = 'l1'
    else:
        reg_type = 'l2'
    assert dropout >= 0 and dropout < 1
    assert learning_rate > 0
    assert weight_decay >= 0
    assert method in {'erm', 'irm', 'dro'}
    if method in {'irm', 'dro'}:
        assert isinstance(train_loader, list)
        multi_domain_method = True
    else:
        multi_domain_method = False
    
    assert fileheader[-1] == '_'
    best_model_filename     = fileheader + str(n_epochs) + 'epochs_best_model.pt.gz'
    best_model_exists       = os.path.exists(best_model_filename)
    last_model_filename     = fileheader + str(n_epochs) + 'epochs_last_model.pt.gz'
    last_model_exists       = os.path.exists(last_model_filename)
    last_optimizer_filename = fileheader + str(n_epochs) + 'epochs_last_adam_optimizer.pt.gz'
    last_optimizer_exists   = os.path.exists(last_optimizer_filename)
    losses_filename         = fileheader + str(n_epochs) + 'epochs_losses.json'
    losses_exist            = os.path.exists(losses_filename)
    loaded_from_files       = False
    
    best_model = deepcopy(model)
    adam_optimizer = torch.optim.Adam(model.parameters(),
                                      lr           = learning_rate,
                                      weight_decay = 0.)
    if torch.cuda.is_available():
        model = model.cuda()
        best_model = best_model.cuda()
    
    if best_model_exists and last_model_exists and last_optimizer_exists and losses_exist:
        loaded_from_files = True
        
        best_model = load_state_dict_from_gz(best_model,
                                             best_model_filename,
                                             remove_uncompressed_model)
        logger.info('Loaded best model from ' + best_model_filename)
        
        model = load_state_dict_from_gz(model,
                                        last_model_filename,
                                        remove_uncompressed_model)
        logger.info('Loaded last model from ' + last_model_filename)
        
        adam_optimizer = load_state_dict_from_gz(adam_optimizer,
                                                 last_optimizer_filename,
                                                 remove_uncompressed_model)
        logger.info('Loaded last optimizer from ' + last_optimizer_filename)
        
        with open(losses_filename, 'r') as f:
            losses = json.load(f)
        logger.info('Loaded losses and accuracies from ' + losses_filename)
        
    else:
        overall_start_time = time.time()
        
        if method == 'irm':
            irm_weight = torch.nn.Parameter(torch.ones(model.num_classes))
            if torch.cuda.is_available():
                irm_weight = irm_weight.cuda()
        else:
            irm_weight = None
        
        assert resume_from_n_epochs <= n_epochs
        if resume_from_n_epochs > 0:
            resume_last_model_filename     = fileheader + str(resume_from_n_epochs) + 'epochs_last_model.pt.gz'
            resume_last_optimizer_filename = fileheader + str(resume_from_n_epochs) + 'epochs_last_adam_optimizer.pt.gz'
            resume_losses_filename         = fileheader + str(resume_from_n_epochs) + 'epochs_losses.json'
            resume_irm_weight_filename     = fileheader + str(resume_from_n_epochs) + 'epochs_last_irm_weight.pt.gz'
            assert os.path.exists(resume_last_model_filename)
            assert os.path.exists(resume_last_optimizer_filename)
            assert os.path.exists(resume_losses_filename)
            
            model = load_state_dict_from_gz(model,
                                            resume_last_model_filename,
                                            remove_uncompressed_model)
            logger.info('Loaded model to resume from ' + resume_last_model_filename)
            
            adam_optimizer = load_state_dict_from_gz(adam_optimizer,
                                                     resume_last_optimizer_filename,
                                                     remove_uncompressed_model)
            logger.info('Loaded Adam optimizer to resume from ' + resume_last_optimizer_filename)
            
            if method == 'irm':
                irm_weight = load_state_dict_from_gz(irm_weight,
                                                     resume_irm_weight_filename,
                                                     remove_uncompressed_model)
                logger.info('Loaded IRM weight to resume from ' + resume_irm_weight_filename)
            
            with open(resume_losses_filename, 'r') as f:
                losses     = json.load(f)
            best_valid_acc = max(losses['valid_step' + str(num_time_steps - 1) + '_accuracies'])
            found_better   = False
        else:
            if method == 'erm':
                train_loss = compute_loss(model,
                                          train_loader)
                train_acc  = compute_accuracy(model,
                                              train_loader)
            else:
                train_loss = compute_average_loss(model,
                                                  train_loader)
                train_acc  = compute_average_accuracy(model,
                                                      train_loader)
            losses         = {'train_losses'    : [train_loss],
                              'train_accuracies': [train_acc],
                              'valid_losses'    : [compute_loss(model, valid_loader)],
                              'valid_accuracies': [compute_accuracy(model, valid_loader)]}
            best_valid_acc = losses['valid_accuracies'][0]
            best_epoch     = 0
            
        logger.info('Starting at train loss ' + str(losses['train_losses'][-1])
                    + ', train acc ' + str(losses['train_accuracies'][-1])
                    + ', val loss ' + str(losses['valid_losses'][-1]) 
                    + ', val acc ' + str(losses['valid_accuracies'][-1]))
        
        params_to_update = set_parameters_to_update(model,
                                                    layers_to_tune,
                                                    logger)
        
        best_params = deepcopy(model.state_dict())
        loss_fn     = torch.nn.CrossEntropyLoss(reduction = 'none')
        
        n_epochs_stop = n_epochs
        n_epochs_no_improvement = 0
        if multi_domain_method:
            max_batches_domain_idx = np.argmax(np.array([0 if train_loader[i] is None
                                                         else len(train_loader[i]) 
                                                         for i in range(len(train_loader))]))
            all_domain_loaders     = train_loader
            all_domain_generators  = [iter(loader)
                                      for loader in all_domain_loaders]
            train_loader           = all_domain_loaders[max_batches_domain_idx]
        for epoch in range(resume_from_n_epochs, n_epochs):
            
            start_time = time.time()
            # update weights
            for batch_idx, batch_samples in enumerate(train_loader):
                loss_train = compute_batch_loss(batch_samples,
                                                model,
                                                loss_fn,
                                                dropout,
                                                irm_weight)

                if multi_domain_method:
                    # compute loss for each domain
                    for domain_idx in range(len(all_domain_loaders)):
                        if domain_idx == max_batches_domain_idx:
                            continue
                        try:
                            batch_samples = next(all_domain_generators[domain_idx])
                        except StopIteration:
                            # restart data loader for this time step if reach end
                            all_domain_generators[domain_idx]  = iter(all_domain_loaders[domain_idx])
                            batch_samples = next(all_domain_generators[domain_idx])
                        
                        batch_loss = compute_batch_loss(batch_samples,
                                                        model,
                                                        loss_fn,
                                                        dropout,
                                                        irm_weight)
                        
                        if method == 'dro':
                            if batch_loss > loss_train:
                                loss_train = batch_loss
                        else:
                            loss_train += batch_loss
                
                if regularization == 'standard':
                    loss_train += weight_decay * model.compute_param_norm(reg_type)
                else:
                    for prev_model_idx in range(len(regularization_prev_models)):
                        loss_train += weight_decay \
                                    * model.compute_param_norm_to_another_model(regularization_prev_models[prev_model_idx],
                                                                                reg_type,
                                                                                regularization_fisher_infos[prev_model_idx])
                    
                if torch.cuda.is_available():
                    loss_train = loss_train.cuda()
                loss_train.backward(retain_graph = True)
                
                #gradient_clip_size = 1
                #torch.nn.utils.clip_grad_norm_(params_to_update, gradient_clip_size)
                adam_optimizer.step()
                adam_optimizer.zero_grad()
                
            # track losses
            if multi_domain_method:
                train_loss = compute_average_loss(model,
                                                  all_domain_loaders)
                train_acc  = compute_average_accuracy(model,
                                                      all_domain_loaders)
            else:
                train_loss = compute_loss(model,
                                          train_loader)
                train_acc  = compute_accuracy(model,
                                              train_loader)
            losses['train_losses'].append(train_loss)
            losses['train_accuracies'].append(train_acc)
            valid_loss = compute_loss(model, valid_loader)
            losses['valid_losses'].append(valid_loss)
            valid_acc  = compute_accuracy(model, valid_loader)
            losses['valid_accuracies'].append(valid_acc)
            if valid_acc > best_valid_acc:
                best_valid_acc   = valid_acc
                best_params      = deepcopy(model.state_dict())
                best_epoch       = epoch + 1
                if resume_from_n_epochs > 0:
                    found_better = True
            
            logger.info('Epoch ' + str(epoch + 1) + ' achieved train loss ' + str(losses['train_losses'][-1])
                        + ', train acc ' + str(losses['train_accuracies'][-1])
                        + ', val loss ' + str(losses['valid_losses'][-1]) 
                        + ', val acc ' + str(losses['valid_accuracies'][-1])
                        + ' in ' + str(time.time() - start_time) + ' seconds')
            
            if losses['valid_accuracies'][-1] <= losses['valid_accuracies'][-2]:
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
        
        if resume_from_n_epochs > 0:
            if found_better:
                logger.info('Best val acc at epoch ' + str(best_epoch))
            else:
                logger.info('Resuming fitting did not improve val acc')
        else:
            logger.info('Best val acc at epoch ' + str(best_epoch))
        
        best_model.load_state_dict(best_params)
        
        logger.info('Fitted model in ' + str(time.time() - overall_start_time) + ' seconds')
        
        if save_model:
            save_state_dict_to_gz(best_model,
                                  best_model_filename)
            logger.info('Saved best model to ' + best_model_filename)
            
            save_state_dict_to_gz(model,
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
        plot_losses_over_epochs(losses['train_losses'],
                                losses['valid_losses'],
                                losses_plot_filename,
                                logger,
                                plot_title = model_name)
    
    if not os.path.exists(acc_plot_filename):
        plot_losses_over_epochs(losses['train_accuracies'],
                                losses['valid_accuracies'],
                                acc_plot_filename,
                                logger,
                                plot_title = model_name,
                                accuracy   = True)
    
    best_val_acc = max(losses['valid_accuracies'])
    logger.info('Best validation accuracy for ' + best_model_filename + ': ' + str(best_val_acc))
    return best_model, best_val_acc, model, adam_optimizer, n_epochs_stop

def tune_hyperparameters_for_model(layers_to_tune,
                                   init_model,
                                   train_loader,
                                   valid_loader,
                                   learning_rates,
                                   weight_decays,
                                   dropouts,
                                   n_epochs,
                                   fileheader,
                                   logger,
                                   allow_load_best_from_disk   = True,
                                   regularization              = 'standard',
                                   regularization_prev_models  = [],
                                   regularization_fisher_infos = [],
                                   remove_uncompressed_model   = True,
                                   method                      = 'erm'):
    '''
    Find best hyperparameters for model
    @param layers_to_tune: str, which layers to tune, 
                           options: all or combination of conv1,layer1,layer2,layer3,layer4,fc
    @param init_model: block_model, model to fit
    @param train_loader: torch DataLoader, contains training features, outcomes, and optionally sample weights in batches
    @param valid_loader: torch DataLoader, contains validation features and outcomes in batches
    @param learning_rates: list of float, learning rates to try
    @param weight_decays: list of float, regularization constants to try
    @param dropouts: list of float, dropout rates to try
    @param n_epochs: int, number of epochs
    @param logger: logger, for INFO messages
    @param fileheader: str, start of paths to files, ends in _, will append comma-separated layers to tune
    @param allow_load_best_from_disk: bool, whether best model can be loaded from disk, 
                                      otherwise will overwrite best model on disk,
                                      set to False if expanding set of hyperparameters
    @param regularization: str, options: standard for L2 regularization towards 0,
                                         previous for L2 regularization towards previous model,
                                         fisher for L2 regularization towards previous models weighted by Fisher info,
                                         previous_l1 for L1 regularization towards previous model
    @param regularization_prev_models: list of block_models, previous models to regularize towards,
                                       length 0 for standard, length 1 for previous, at least length 1 for Fisher,
                                       must be same architecture as init_model
    @param regularization_fisher_infos: list of dict mapping str to FloatTensor, list over previous models,
                                        dict maps layer name to Fisher info for each parameter, only used for Fisher
    @param remove_uncompressed_model: bool, whether to remove uncompressed .pt file when loading model, usually can set to True 
                                      unless expecting multiple threads to load the same model at the same time
    @param method: str, erm, irm, or dro
    @return: 1. block_model, fitted and set to best parameters
             2. validation accuracy
    '''
    assert fileheader[-1] == '_'
    assert regularization in {'standard', 'previous', 'fisher', 'previous_l1'}
    assert method in {'erm', 'irm', 'dro'}
    if method == 'erm':
        combo_fileheader      = fileheader + layers_to_tune + '_'
    else:
        combo_fileheader      = fileheader + method + '_' + layers_to_tune + '_'
    best_hyperparams_filename = combo_fileheader + 'best_hyperparams.json'
    best_model_filename       = combo_fileheader + 'best_model.pt.gz'
    if regularization == 'previous_l1':
        reg_type = 'l1'
    else:
        reg_type = 'l2'
    
    if allow_load_best_from_disk and os.path.exists(best_hyperparams_filename) and os.path.exists(best_model_filename):
        best_model = deepcopy(init_model)
        best_model = load_state_dict_from_gz(best_model,
                                             best_model_filename,
                                             remove_uncompressed_model)
        if torch.cuda.is_available():
            best_model = best_model.cuda()
        
        with open(best_hyperparams_filename, 'r') as f:
            best_hyperparams = json.load(f)
        if best_hyperparams['learning rate'] in learning_rates \
        and best_hyperparams['weight decay'] in weight_decays \
        and best_hyperparams['dropout'] in dropouts \
        and best_hyperparams['regularization'] == regularization \
        and best_hyperparams['early stopping epochs'] <= n_epochs:
            # best hyperparams are among options given
            losses_filename = combo_fileheader + 'lr' + str(best_hyperparams['learning rate']) \
                            + '_' + regularization + '_reg' + str(best_hyperparams['weight decay']) \
                            + '_dropout' + str(best_hyperparams['dropout']) \
                            + '_' + str(best_hyperparams['early stopping epochs']) + 'epochs_losses.json'
            if os.path.exists(losses_filename):
                with open(losses_filename, 'r') as f:
                    losses = json.load(f)
                best_valid_acc = max(losses['valid_accuracies'])
                
                logger.info('Loaded best model from ' + best_model_filename)
                return best_model, best_valid_acc
        
    best_val_acc = -1
    for learning_rate, weight_decay, dropout in product(learning_rates, weight_decays, dropouts):
        model                 = deepcopy(init_model)
        hyperparam_fileheader = combo_fileheader + 'lr' + str(learning_rate) \
                              + '_' + regularization + '_reg' + str(weight_decay) \
                              + '_dropout' + str(dropout) + '_'
        
        best_epoch_model, val_acc, last_model, last_optimizer, n_epochs_stop \
            = fit_single_model(layers_to_tune,
                               model,
                               train_loader,
                               valid_loader,
                               learning_rate,
                               weight_decay,
                               dropout,
                               n_epochs,
                               hyperparam_fileheader,
                               logger,
                               save_model                  = False,
                               early_stopping_n_epochs     = 5,
                               regularization              = regularization,
                               regularization_prev_models  = regularization_prev_models,
                               regularization_fisher_infos = regularization_fisher_infos,
                               remove_uncompressed_model   = remove_uncompressed_model,
                               method                      = method)
        
        logger.info('Block model fitting '     + layers_to_tune
                    + ' with learning rate ' + str(learning_rate)
                    + ', ' + regularization + ' regularization ' + str(weight_decay)
                    + ', dropout '           + str(dropout)
                    + ' val acc: '           + str(val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc        = val_acc
            best_model          = best_epoch_model
            best_learning_rate  = learning_rate
            best_weight_decay   = weight_decay
            best_dropout        = dropout
            best_last_model     = last_model
            best_last_optimizer = last_optimizer
            best_n_epochs_stop  = n_epochs_stop
    
    logger.info('Best hyperparameters when fitting ' + layers_to_tune
                + ': learning rate '     + str(best_learning_rate)
                + ', ' + regularization + ' regularization ' + str(best_weight_decay)
                + ', val acc '           + str(best_val_acc))
    
    best_hyperparams = {'learning rate'        : best_learning_rate,
                        'weight decay'         : best_weight_decay,
                        'dropout'              : best_dropout,
                        'regularization'       : regularization,
                        'early stopping epochs': best_n_epochs_stop}
    with open(best_hyperparams_filename, 'w') as f:
        json.dump(best_hyperparams, f)
    
    save_state_dict_to_gz(best_model,
                          best_model_filename)
    logger.info('Saved best model when fitting ' + layers_to_tune
                + ' to ' + best_model_filename)
    
    last_model_filename = combo_fileheader + str(best_n_epochs_stop) + 'epochs_last_model.pt.gz'
    save_state_dict_to_gz(best_last_model,
                          last_model_filename)
    logger.info('Saved last model from fitting ' + layers_to_tune
                + ' to ' + last_model_filename)
    
    last_optimizer_filename = combo_fileheader + str(best_n_epochs_stop) + 'epochs_last_adam_optimizer.pt.gz'
    save_state_dict_to_gz(best_last_optimizer,
                          last_optimizer_filename)
    logger.info('Saved last optimizer from fitting ' + layers_to_tune
                + ' to ' + last_optimizer_filename)
    
    return best_model, best_val_acc