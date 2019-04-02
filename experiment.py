import os
import argparse
import torch
import random
import shutil
from tqdm import tqdm
import json
import numpy as np

# import data_loader as data_loaders
# import model as models
# import trainer.loss as loss_functions
# import trainer.metric as metric_functions

import train
import evaluate

import utils.util as util
from utils import color_print as cp

EXP_LOSS = ['softmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss']
TARGET_CLASS = [0,1,2,3,4,5,6,7,8,9]

def train_models(num_model, saved_model_dir):

    base_config = json.load(open('config/mnist_base.json'))
    fine_tune_config = json.load(open('config/mnist_fine_tune.json'))

    fine_tune_config['trainer']['epochs'] = fine_tune_config['trainer']['epochs'] + base_config['trainer']['epochs']

    for i in range(num_model):
        seed = random.randint(0, 200)

        for loss in EXP_LOSS:
            model_dir = os.path.join(saved_model_dir, loss) 
            next_index = str(int(max(os.listdir(model_dir))) + 1)
            dest_dir = os.path.join(saved_model_dir, loss, next_index)

            cp.print_warning("training ", i+1, "th model with loss : ", loss)
            cp.print_warning("destiation :", dest_dir)

            base_config['data_loader']['args']['seed'] = seed
            fine_tune_config['data_loader']['args']['seed'] = seed

            base_model = train.train_base_model(base_config)

            for target in TARGET_CLASS:
                fine_tune_config['target_class'] = [target]
                train.fine_tune_model(fine_tune_config, base_model)

            shutil.move('saved', dest_dir)

def evaluate_base_model(saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        acc = []

        model_dir = os.path.join(saved_model_dir, loss)
        for i in os.listdir(model_dir):
            base_model_dir = os.path.join(saved_model_dir, loss, i, 'mnist_base')
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model : ", base_model)

            config = torch.load(base_model)['config']

            model, data_loader, loss_fn, metrics = evaluate.load_model(config, base_model, TARGET_CLASS)

            config['metrics'] = ["pred_acc"]
            log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

            acc.append(log['pred_acc'])

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = round(np.array(acc).mean(), 3)
        mapping[loss]['raw_accuracy'] = acc

        cp.print_warning('average base model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning(mapping[loss]['raw_accuracy'])

    return mapping

def evaluate_combined_model(saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        acc = []

        model_dir = os.path.join(saved_model_dir, loss)
        for i in os.listdir(model_dir):
            base_model_dir = os.path.join(saved_model_dir, loss, i, 'mnist_base')
            fine_tuned_model_dir = os.path.join(saved_model_dir, loss, i, 'mnist_fine_tune')
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model : ", base_model)

            config = torch.load(base_model)['config']

            model, data_loader, loss_fn, metrics = evaluate.load_model(config, base_model, TARGET_CLASS)

            config['metrics'] = ["pred_acc"]

            model = evaluate.combine_model(model, fine_tuned_model_dir, TARGET_CLASS)

            log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

            acc.append(log['pred_acc'])

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = round(np.array(acc).mean(), 3)
        mapping[loss]['raw_accuracy'] = acc

        cp.print_warning('average base model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning(mapping[loss]['raw_accuracy'])

    return mapping


def evaluate_models(saved_model_dir):
    base_model_acc = evaluate_base_model(saved_model_dir)
    combined_model_acc = evaluate_combined_model(saved_model_dir)

    print(json.dumps(base_model_acc, indent=4, separators=(': ')))
    print(json.dumps(combined_model_acc, indent=4, separators=(': ')))


def main(train_flag, saved_model_dir, num_model):
    util.makedir_exist_ok(saved_model_dir)

    for loss in EXP_LOSS:
        dest_dir = os.path.join(saved_model_dir, loss)
        util.makedir_exist_ok(dest_dir)

    if train_flag:
        train_models(num_model, saved_model_dir)

    evaluate_models(saved_model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-m', '--models', type=str,
                        default="trained",
                        help='path to dir contnaining trained models (default: saved)')
    parser.add_argument('-t', '--train', help="train models", action='store_true')
    parser.add_argument('-n', '--num_model', default=1, type=int, help="number of models to train")

    args = parser.parse_args()
    main(args.train, args.models, args.num_model)
