import os
import json
import argparse
import torch
import data_loader as data_loaders
import model as models
import trainer.loss as loss_functions
import trainer.metric as metric_functions

from utils import Logger
from utils import color_print as cp

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(data_loaders, 'data_loader', config)
    cp.print_progress('DATASET\n', data_loader)

    # build model architecture
    model = get_instance(models, 'model', config)
    cp.print_progress('MODEL\n', model)

    # get function handles of loss and metrics
    loss = getattr(loss_functions, config['loss'])
    cp.print_progress('LOSS FUNCTION\n', loss.__name__)

    metrics = [getattr(metric_functions, met) for met in config['metrics']]
    cp.print_progress('METRICS\n', [metric.__name__ for metric in metrics])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    cp.print_progress('OPTIMIZER\n', optimizer)

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    cp.print_progress('LR_SCHEDULER\n', type(lr_scheduler).__name__)

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
