import os
import argparse
import torch
from tqdm import tqdm

import data_loader as data_loaders
import model as models
import trainer.loss as loss_functions
import trainer.metric as metric_functions

import utils.util as util
from utils import color_print as cp


def main(config, resume):
    # # setup data_loader instances
    # data_loader = get_instance(data_loaders, 'data_loader', config)

    # setup data_loader instances
    data_loader = getattr(data_loaders, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # TODO :: use generic function for printing out model setting
    cp.print_progress('test DATASET\n', data_loader)

    # build model architecture
    model = util.get_instance(models, 'model', config)
    cp.print_progress('MODEL\n', model)

    # get function handles of loss and metrics
    loss_fn = getattr(loss_functions, config['loss'])
    cp.print_progress('LOSS FUNCTION\n', loss_fn.__name__)

    metrics = [getattr(metric_functions, met) for met in config['metrics']]
    cp.print_progress('METRICS\n', [metric.__name__ for metric in metrics])

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metrics)})

    test_result_str = 'TEST RESULTS\n'
    for key, val in log.items():
        test_result_str += ('\t' + str(key) + ' : ' + str(val) + '\n')

    cp.print_progress(test_result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
