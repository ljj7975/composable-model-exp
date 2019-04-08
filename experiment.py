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

TARGET_CLASS = []

def train_models(num_model, saved_model_dir):

    base_config = json.load(open('config/{}_base.json'.format(EXP_TYPE)))
    fine_tune_config = json.load(open('config/{}_fine_tune.json'.format(EXP_TYPE)))

    fine_tune_config['trainer']['epochs'] = fine_tune_config['trainer']['epochs'] + base_config['trainer']['epochs']
    # if "media" not in base_config['data_loader']['args']['data_dir']:
    #     base_config['data_loader']['args']['data_dir'] = "/media/brandon/SSD" + base_config['data_loader']['args']['data_dir']

    # if "media" not in fine_tune_config['data_loader']['args']['data_dir']:
    #     fine_tune_config['data_loader']['args']['data_dir'] = "/media/brandon/SSD" + fine_tune_config['data_loader']['args']['data_dir']

    for i in tqdm(range(num_model)):
        seed = random.randint(0, 200)

        for loss in EXP_LOSS:
            base_config["trainer"]["save_dir"] = EXP_TYPE + "_" + loss
            fine_tune_config["trainer"]["save_dir"] = EXP_TYPE + "_" + loss
            base_config["trainer"]["log_dir"] = EXP_TYPE + "_" + loss + '/runs'
            fine_tune_config["trainer"]["log_dir"] = EXP_TYPE + "_" + loss + '/runs'

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

            # if next_index > 0:
            #     prev_index = next_index - 1
            #     prev_model_dir = os.path.join(saved_model_dir, loss, str(prev_index))

            #     copy_src_dir = os.path.join(prev_model_dir, base_config['name'])
            #     copy_dest_dir = os.path.join(base_config["trainer"]["save_dir"], '{}_base'.format(EXP_TYPE)

            #     cp.print_warning("copy the base model from ", copy_src_dir, "to", copy_dest_dir)
            #     shutil.copytree(copy_src_dir, copy_dest_dir)

            #     best_base_model = max(os.listdir(copy_dest_dir))
            #     base_model = os.path.join(copy_dest_dir, best_base_model, 'model_best.pth')

            #     cp.print_warning("loaded base model", base_model)
            # else:
            base_model = train.train_base_model(base_config)

            for target in TARGET_CLASS:
                fine_tune_config['target_class'] = [target]
                train.fine_tune_model(fine_tune_config, base_model)

            cp.print_warning("moving saved folder to", dest_dir)
            shutil.move(base_config["trainer"]["save_dir"], dest_dir)

def evaluate_base_model(saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        acc = []

        model_dir = os.path.join(saved_model_dir, loss)
        for i in tqdm(os.listdir(model_dir)):
            base_model_dir = os.path.join(saved_model_dir, loss, i, '{}_base'.format(EXP_TYPE))
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model : ", base_model)

            if not torch.cuda.is_available():
                config = torch.load(base_model, map_location='cpu')['config']
            else:
                config = torch.load(base_model)['config']
            
            # if "media" not in config['data_loader']['args']['data_dir']:
            #     config['data_loader']['args']['data_dir'] = "/media/brandon/SSD" + config['data_loader']['args']['data_dir']

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

def evaluate_fine_tuned_model(saved_model_dir):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        sum = [0] * len(TARGET_CLASS)

        model_dir = os.path.join(saved_model_dir, loss)
        for i in os.listdir(model_dir):
            fine_tuned_model_dir = os.path.join(saved_model_dir, loss, i, '{}_fine_tune'.format(EXP_TYPE))

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
                
                # if "media" not in config['data_loader']['args']['data_dir']:
                #     config['data_loader']['args']['data_dir'] = "/media/brandon/SSD" + config['data_loader']['args']['data_dir']

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

def evaluate_combined_model(saved_model_dir, num_iter, step_size):
    mapping = {}

    for loss in EXP_LOSS:
        cp.print_warning("loss function : ", loss)
        sum = [0] * int((len(TARGET_CLASS)/step_size))

        model_dir = os.path.join(saved_model_dir, loss)
        for i in tqdm(os.listdir(model_dir)):
            base_model_dir = os.path.join(saved_model_dir, loss, i, '{}_base'.format(EXP_TYPE))
            fine_tuned_model_dir = os.path.join(saved_model_dir, loss, i, '{}_fine_tune'.format(EXP_TYPE))
            latest_model = max(os.listdir(base_model_dir))
            base_model = os.path.join(base_model_dir, latest_model, 'model_best.pth')
            cp.print_warning("base model for combined model : ", base_model)
            cp.print_warning("fine tuned model model for combined model : ", base_model)

            if not torch.cuda.is_available():
                config = torch.load(base_model, map_location='cpu')['config']
            else:
                config = torch.load(base_model)['config']

            # if "media" not in config['data_loader']['args']['data_dir']:
            #     config['data_loader']['args']['data_dir'] = "/media/brandon/SSD" + config['data_loader']['args']['data_dir']

            config['metrics'] = ["pred_acc"]
            ordered_class = TARGET_CLASS.copy()

            for _ in range(num_iter):
                random.shuffle(ordered_class)

                target_class = []

                index = 0

                while len(ordered_class) > 0:
                    target_class += ordered_class[:step_size]
                    ordered_class = ordered_class[step_size:]

                    print('target_class', target_class)

                    model, data_loader, loss_fn, metrics = evaluate.load_model(config, base_model, target_class)

                    model = evaluate.combine_model(model, fine_tuned_model_dir, target_class)

                    log = evaluate.evaluate(model, data_loader, loss_fn, metrics)

                    sum[index] += log['pred_acc']
                    index += 1

        mapping[loss] = {}
        mapping[loss]['average_accuracy'] = []
        num_model = len(os.listdir(model_dir))

        for acc in sum:
            mapping[loss]['average_accuracy'].append(round((acc / (num_model * num_iter)) * 100, 2))
        mapping[loss]['summed_accuracy'] = sum

        cp.print_warning('average combined model accuracy :', mapping[loss]['average_accuracy'])
        cp.print_warning('summed combined model accuracy :', sum)

    return num_model, mapping


def evaluate_models(saved_model_dir, num_iter, step_size):
    cp.print_progress("< evaluate base model >")
    num_base_model, base_model_acc = evaluate_base_model(saved_model_dir)
    cp.print_progress("< base model acc >")
    cp.print_progress(json.dumps(base_model_acc, indent=4, separators=(': ')))

    cp.print_progress("< evaluate fine tuned model >")
    num_fine_tuned_model, fine_tuned_model_acc = evaluate_fine_tuned_model(saved_model_dir)
    cp.print_progress("< fine tuned model acc >")
    cp.print_progress(json.dumps(fine_tuned_model_acc, indent=4, separators=(': ')))

    cp.print_progress("< evaluate combined model >")
    num_combined_model, combined_model_acc = evaluate_combined_model(saved_model_dir, num_iter, step_size)
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

def main(train_flag, saved_model_dir, num_model, num_iter, step_size):
    global TARGET_CLASS

    cp.print_warning("loss functions :", EXP_LOSS)
    util.makedir_exist_ok(saved_model_dir)

    base_config = json.load(open('config/{}_base.json'.format(EXP_TYPE)))
    TARGET_CLASS = np.arange(base_config['n_class']).tolist()
    cp.print_warning("target class :", TARGET_CLASS)
    cp.print_warning("step size :", step_size)

    assert len(TARGET_CLASS) % step_size == 0

    for loss in EXP_LOSS:
        dest_dir = os.path.join(saved_model_dir, loss)
        util.makedir_exist_ok(dest_dir)

    if train_flag:
        train_models(num_model, saved_model_dir)

    results = evaluate_models(saved_model_dir, num_iter, step_size)
    results['num_class'] = base_config['n_class']
    results['target_class'] = TARGET_CLASS
    results['step_size'] = step_size

    cp.print_warning("< EXP RESULTS >")
    for loss in EXP_LOSS:
        cp.print_warning("loss : ", loss)
        cp.print_warning(json.dumps(results[loss], indent=4))

    summary_file = os.path.join(saved_model_dir, 'summary.txt')
    with open(summary_file, 'w') as file:
        results_json = json.dumps(results)
        file.write(results_json)

if __name__ == '__main__':
    global EXP_LOSS
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
                        choices=["mnist", "cifar10", "cifar100", "kws_res8_narrow", "kws_res15_narrow", "kws_res26_narrow"],
                        help="type of exp (default: mnist)")
    parser.add_argument('-l', '--loss_fn', nargs='+', type=str,
                        default=['logsoftmax_nll_loss', 'softmax_bce_loss', 'sigmoid_bce_loss'],
                        help='list of loss functions to include (default: logsoftmax_nll_loss, softmax_bce_loss, sigmoid_bce_loss)')
    parser.add_argument('-s', '--step', default=1, type=int,
                        help="step size for composed mode evaluation (default: 1)")
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    EXP_LOSS = args.loss_fn
    EXP_TYPE = args.exp_type

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args.train, args.models, args.num_model, args.num_iter, args.step)
