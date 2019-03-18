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

def main(config, resume, fine_tuned_model_dir, target_class):

    # build model architecture
    model = util.get_instance(models, 'model', config)

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict)

    weight_list = []
    bias_list = []
    for target in target_class:

        dir_path = os.path.join(fine_tuned_model_dir, str(target))
        trained_models = os.listdir(dir_path)

        latest_best_model = os.path.join(dir_path, max(trained_models), "model_best.pth");

        checkpoint = torch.load(latest_best_model)
        state_dict = checkpoint['state_dict']

        weight_list.append(state_dict["fc2.weight"][0])
        bias_list.append(state_dict["fc2.bias"][0])

    weight = torch.stack(weight_list)
    bias = torch.stack(bias_list)

    layer_id = model.swap_fc(len(target_class))

    model.fc2.weight = torch.nn.Parameter(weight)
    model.fc2.bias = torch.nn.Parameter(bias)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # setup data_loader instances
    data_loader = getattr(data_loaders, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # get function handles of loss and metrics
    loss_fn = getattr(loss_functions, config['loss'])

    metrics = [getattr(metric_functions, met) for met in config['metrics']]

    util.print_setting(data_loader, None, model, loss_fn, metrics,  None, None)

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
    parser.add_argument('-ft', '--fine_tuned_model_dir', type=str,
                        default="saved/mnist_fine_tune",
                        help='path to dir contnaining fine tuned model (default: saved/mnist_fine_tune)')
    parser.add_argument('-t', '--target_class', nargs='+', type=int,
                        help="target class to fine tune (default: all 10 classes)", default=[1,2,3,4,5,6,7,8,9,10])

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume, args.fine_tuned_model_dir, args.target_class)
