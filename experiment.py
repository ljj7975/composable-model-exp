import os
import argparse
import random
import shutil
import json
import pprint
import torch
import numpy as np
from tqdm import tqdm

import train
import evaluate

import utils.util as util
from utils import color_print as cp


EXP_LOSS = ['logsoftmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss']
TARGET_CLASS = []

def train_models(exp_type, num_model, saved_model_dir):

    base_config = json.load(open('config/{}_base.json'.format(exp_type)))
    fine_tune_config = json.load(open('config/{}_fine_tune.json'.format(exp_type)))

    fine_tune_config['trainer']['epochs'] = fine_tune_config['trainer']['epochs'] + base_config['trainer']['epochs']

    for i in tqdm(range(num_model)):
        seed = random.randint(0, 200)

        for loss in EXP_LOSS:
            model_dir = os.path.join(saved_model_dir, loss)
            indice = list(map(int, os.listdir(model_dir)))
            next_index = max(indice) + 1 if indice else 0

            dest_dir = os.path.join(saved_model_dir, loss, str(next_index))

            cp.print_warning("training ", i+1, "th model with loss : ", loss)
            cp.print_warning("destiation :", dest_dir)

            base_config['data_loader']['args']['seed'] = seed
            fine_tune_config['data_loader']['args']['seed'] = seed

            base_config['loss'] = loss
            fine_tune_config['loss'] = loss

            if next_index > 0:
                prev_index = next_index - 1
                prev_model_dir = os.path.join(saved_model_dir, loss, str(prev_index))

                copy_src_dir = os.path.join(prev_model_dir, base_config['name'])
                copy_dest_dir = os.path.join('saved', '{}_base'.format(exp_type))

                cp.print_warning("copy the base model from ", copy_src_dir, "to", copy_dest_dir)
                shutil.copytree(copy_src_dir, copy_dest_dir)

                best_base_model = max(os.listdir(copy_dest_dir))
                base_model = os.path.join(copy_dest_dir, best_base_model, 'model_best.pth')

                cp.print_warning("loaded base model", base_model)

            else:
                base_model = train.train_base_model(base_config)

            for target in TARGET_CLASS:
                fine_tune_config['target_class'] = [target]
                train.fine_tune_model(fine_tune_config, base_model)

            cp.print_warning("moving saved folder to", dest_dir)
            shutil.move('saved', dest_dir)

def evaluate_base_model(exp_type, saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        acc = []

        model_dir = os.path.join(saved_model_dir, loss)
        for i in tqdm(os.listdir(model_dir)):
            base_model_dir = os.path.join(saved_model_dir, loss, i, '{}_base'.format(exp_type))
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model : ", base_model)

            if not torch.cuda.is_available():
                config = torch.load(base_model, map_location='cpu')['config']
            else:
                config = torch.load(base_model)['config']

            config['metrics'] = ["pred_acc"]
            model, data_loader, loss_fn, metrics = evaluate.load_model(config, base_model, TARGET_CLASS)

            log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

            acc.append(log['pred_acc'])

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = round(np.array(acc).mean() * 100, 2)
        mapping[loss]['raw_accuracy'] = acc

        cp.print_warning('average base model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning(mapping[loss]['raw_accuracy'])

    return len(os.listdir(model_dir)), mapping

def evaluate_fine_tuned_model(exp_type, saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        sum = [0] * len(TARGET_CLASS)

        model_dir = os.path.join(saved_model_dir, loss)
        for i in tqdm(os.listdir(model_dir)):
            fine_tuned_model_dir = os.path.join(saved_model_dir, loss, i, '{}_fine_tune'.format(exp_type))

            for c in os.listdir(fine_tuned_model_dir):
                class_model_dir = os.path.join(fine_tuned_model_dir, c)
                latest_model = max(os.listdir(class_model_dir))
                fine_tuned_model = os.path.join(class_model_dir, latest_model, 'model_best.pth')
                cp.print_warning("fine tuned model : ", fine_tuned_model)

                if not torch.cuda.is_available():
                    config = torch.load(fine_tuned_model, map_location='cpu')['config']
                else:
                    config = torch.load(fine_tuned_model)['config']

                target_class = [int(c)]
                config['model']['args']['num_classes'] = len(target_class) + 1

                model, data_loader, loss_fn, metrics = evaluate.load_model(config, fine_tuned_model, target_class)

                log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

                sum[int(c)] += log['pred_acc']

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = []
        num_model = len(os.listdir(model_dir))

        for acc in sum:
            mapping[loss]['average_accuracy'].append(round((acc/num_model) * 100, 2))
        mapping[loss]['summed_accuracy'] = sum

        cp.print_warning('average fine tune model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning('summed fine tune model accuracy :', sum)

    return num_model, mapping

def evaluate_combined_model(exp_type, saved_model_dir, num_iter):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        sum = [0] * len(TARGET_CLASS)

        model_dir = os.path.join(saved_model_dir, loss)
        for i in tqdm(os.listdir(model_dir)):
            base_model_dir = os.path.join(saved_model_dir, loss, i, '{}_base'.format(exp_type))
            fine_tuned_model_dir = os.path.join(saved_model_dir, loss, i, '{}_fine_tune'.format(exp_type))
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model for combined model : ", base_model)
            cp.print_warning("fine tuned model model for combined model : ", base_model)

            if not torch.cuda.is_available():
                config = torch.load(base_model, map_location='cpu')['config']
            else:
                config = torch.load(base_model)['config']

            config['metrics'] = ["pred_acc"]
            ordered_class = TARGET_CLASS.copy()

            for _ in tqdm(range(num_iter)):
                random.shuffle(ordered_class)

                target_class = []
                for c in ordered_class:
                    target_class.append(int(c))
                    print('target_class', target_class)

                    model, data_loader, loss_fn, metrics = evaluate.load_model(config, base_model, target_class)

                    model = evaluate.combine_model(model, fine_tuned_model_dir, target_class)

                    log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

                    sum[len(target_class)-1] += log['pred_acc']

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = []
        num_model = len(os.listdir(model_dir))

        for acc in sum:
            mapping[loss]['average_accuracy'].append(round((acc / (num_model * num_iter)) * 100, 2))
        mapping[loss]['summed_accuracy'] = sum

        cp.print_warning('average combined model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning('summed combined model accuracy :', sum)

    return num_model, mapping


def evaluate_models(exp_type, saved_model_dir, num_iter):
    num_base_model, base_model_acc = evaluate_base_model(exp_type, saved_model_dir)
    cp.print_progress("< base model acc >")
    cp.print_progress(json.dumps(base_model_acc, indent=4, separators=(': ')))

    num_fine_tuned_model, fine_tuned_model_acc = evaluate_fine_tuned_model(exp_type, saved_model_dir)
    cp.print_progress("< fine tuned model acc >")
    cp.print_progress(json.dumps(fine_tuned_model_acc, indent=4, separators=(': ')))

    num_combined_model, combined_model_acc = evaluate_combined_model(exp_type, saved_model_dir, num_iter)
    cp.print_progress("< combined model acc >")
    cp.print_progress(json.dumps(combined_model_acc, indent=4, separators=(': ')))

    assert num_base_model == num_combined_model
    assert num_base_model == num_fine_tuned_model

    results = {}

    for loss in EXP_LOSS:
        results[loss] = {}
        results[loss]['base_model'] = base_model_acc[loss]
        results[loss]['fine_tuned_model'] = fine_tuned_model_acc[loss]
        results[loss]['combined_model'] = combined_model_acc[loss]

    results['num_model'] = num_base_model
    return results

def main(exp_type, train_flag, saved_model_dir, num_model, num_iter):
    global TARGET_CLASS
    util.makedir_exist_ok(saved_model_dir)

    base_config = json.load(open('config/{}_base.json'.format(exp_type)))
    TARGET_CLASS = np.arange(base_config['n_class']).tolist()
    cp.print_warning("target class :", TARGET_CLASS)

    for loss in EXP_LOSS:
        dest_dir = os.path.join(saved_model_dir, loss)
        util.makedir_exist_ok(dest_dir)

    if train_flag:
        train_models(exp_type, num_model, saved_model_dir)

    results = evaluate_models(exp_type, saved_model_dir, num_iter)

    cp.print_warning("< EXP RESULTS >")
    for loss in EXP_LOSS:
        cp.print_warning("loss : ", loss)
        cp.print_warning(json.dumps(results[loss], indent=4))

    summary_file = os.path.join(saved_model_dir, 'summary.txt')
    with open(summary_file, 'w') as file:
        pprint.pprint(results, stream=file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate composing algorithm')

    parser.add_argument('-m', '--models', type=str,
                        default="trained",
                        help='path to dir contnaining trained models (default: trained)')
    parser.add_argument('-t', '--train', help="train models", action='store_true')
    parser.add_argument('-nm', '--num_model', default=5, type=int,
                        help="number of models to train (default: 5)")
    parser.add_argument('-ni', '--num_iter', default=10, type=int,
                        help="number of iteration for combined model evaluation (default: 10)")
    parser.add_argument('-e', '--exp_type', default="mnist", type=str,
                        choices=["mnist", "cifar10", "cifar100"],
                        help="type of exp (default: mnist)")
    parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args.exp_type, args.train, args.models, args.num_model, args.num_iter)
