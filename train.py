import os
import json
import argparse
import torch

import data_loader as data_loaders
import model as models
import trainer.loss as loss_functions
import trainer.metric as metric_functions
from trainer import Trainer, FineTuner

import utils.util as util
from utils import Logger
from utils import color_print as cp

def train_base_model(config):
    cp.print_progress('Training base model')
    train_logger = Logger()

    # setup data_loader instances
    data_loader = util.get_instance(data_loaders, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = util.get_instance(models, 'model', config)

    # get function handles of loss and metrics
    loss_fn = getattr(loss_functions, config['loss'])
    metrics = [getattr(metric_functions, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = util.get_instance(torch.optim, 'optimizer', config, trainable_params)

    lr_scheduler = util.get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    util.print_setting(data_loader, valid_data_loader, model, loss_fn, metrics,  optimizer, lr_scheduler)

    trainer = Trainer(model, loss_fn, metrics, optimizer,
                      resume=None,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    cp.print_progress('TRAINER\n', trainer)

    trainer.train()

    cp.print_progress('Training base model completed')

    return os.path.join(trainer.checkpoint_dir, 'model_best.pth')

def fine_tune_model(config, base_model):
    target_class = config['target_class']
    cp.print_progress('Fine tune model with', target_class)

    config['data_loader']['args']['target_class'] = target_class
    config['data_loader']['args']['unknown'] = True

    train_logger = Logger()

    # setup data_loader instances
    data_loader = util.get_instance(data_loaders, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = util.get_instance(models, 'model', config)

    # get function handles of loss and metrics
    loss_fn = getattr(loss_functions, config['loss'])
    metrics = [getattr(metric_functions, met) for met in config['metrics']]

    util.print_setting(data_loader, valid_data_loader, model, loss_fn, metrics,  None, None)

    # build base model
    trainer = FineTuner(model, loss_fn, metrics,
                        base_model=base_model,
                        config=config,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        train_logger=train_logger,
                        target_class=target_class)

    trainer.train()

    cp.print_progress('Fine tuning is completed for ', target_class)

    return os.path.join(trainer.checkpoint_dir, 'model_best.pth')

def main(base_config, fine_tune_config, base_model, target_class, seed):
    base_config['data_loader']['args']['seed'] = seed

    if not base_model:
        base_model = train_base_model(base_config)

    fine_tune_config['trainer']['epochs'] = fine_tune_config['trainer']['epochs'] + base_config['trainer']['epochs']

    fine_tune_config['data_loader']['args']['seed'] = seed

    for target in target_class:
        fine_tune_config['target_class'] = [target]

        fine_tune_model(fine_tune_config, base_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keyword spotting convrnn')
    parser.add_argument('-bc', '--base_config', default=None, type=str,
                        help='base model config file path (default: None)')
    parser.add_argument('-fc', '--fine_tune_config', default=None, type=str,
                        help='fine tunning config file path (default: None)')
    parser.add_argument('-b', '--base_model', default=None, type=str,
                        help='path to base_model (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--target_class', nargs='+', type=int,
                        help="target class to fine tune (default: all 10 classes)",
                        default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('-s', '--seed', default=0, type=int, help="random seed")
    args = parser.parse_args()

    if args.base_config:
        # load config file
        base_config = json.load(open(args.base_config))
    elif args.base_model:
        # load base_config file from base model checkpoint
        base_config = torch.load(args.base_model)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.fine_tune_config:
        # load config file
        fine_tune_config = json.load(open(args.fine_tune_config))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(base_config, fine_tune_config, args.base_model, args.target_class, args.seed)
