import os
import json
import argparse
import torch

import data_loader as data_loaders
import model as models
import trainer.loss as loss_functions
import trainer.metric as metric_functions
from trainer import Trainer

import utils.util as util
from utils import Logger
from utils import color_print as cp

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = util.get_instance(data_loaders, 'data_loader', config)
    cp.print_progress('TRAIN DATASET\n', data_loader)

    valid_data_loader = data_loader.split_validation()
    cp.print_progress('VALID DATASET\n', valid_data_loader)

    # build model architecture
    model = util.get_instance(models, 'model', config)
    cp.print_progress('MODEL\n', model)

    # get function handles of loss and metrics
    loss_fn = getattr(loss_functions, config['loss'])
    cp.print_progress('LOSS FUNCTION\n', loss_fn.__name__)

    metrics = [getattr(metric_functions, met) for met in config['metrics']]
    cp.print_progress('METRICS\n', [metric.__name__ for metric in metrics])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = util.get_instance(torch.optim, 'optimizer', config, trainable_params)
    cp.print_progress('OPTIMIZER\n', optimizer)

    lr_scheduler = util.get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    cp.print_progress('LR_SCHEDULER\n', type(lr_scheduler).__name__)

    trainer = Trainer(model, loss_fn, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    cp.print_progress('TRAINER\n', trainer)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keyword spotting convrnn')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
