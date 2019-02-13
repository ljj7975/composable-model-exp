import os
import json
import argparse
import torch
# import data_loader.data_loaders as module_data
import data_loader as data_loaders

from utils import Logger

def get_instance(data_loaders, name, config):
    return getattr(data_loaders, config[name]['type'])(config['keywords'], **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(data_loaders, 'data_loader', config)

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
