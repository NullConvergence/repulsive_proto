###
# This code is taken for test purposes from:
# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
##


import apex.amp as amp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cifar_data import mean, std, get_no_norm_trans, get_datasets, get_text_classes
from model_preproc import PreprocessingModel
from hyper_proto import HyperProto, HyperProtoPGD
from logger import Logger
import utils


mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()

upper_limit, lower_limit = 1, 0


def normalize(X):
    return (X - mu)/std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()

        delta.uniform_(-epsilon, epsilon)
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*epsilon

        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            d = torch.clamp(d + alpha * torch.sign(g),
                            min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(
            model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./cnfg_pgd.yml', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data

    tr_loader, tst_loader = get_datasets(cnfg['data']['flag'],
                                         cnfg['data']['dir'],
                                         cnfg['data']['batch_size'],
                                         apply_transform=False)

    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])

    model = utils.get_model(cnfg['model']).cuda()
    if cnfg['data']['flag'] == '10':
        model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(cnfg['resume']['path'])
    state = checkpoint
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss()

    epsilon = (cnfg['pgd']['epsilon'] / 255.)
    pgd_alpha = (cnfg['pgd']['alpha'] / 255.)

    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    misclassified_samples, true_labels, predicted_labels = [], [], []
    conf_matrix = torch.zeros(
        int(cnfg['data']['flag']), int(cnfg['data']['flag'])).to(device)
    for _, (x, y) in enumerate(tqdm(tst_loader)):
        X, y = x.cuda(), y.cuda()

        delta = attack_pgd(model, X, y, epsilon, pgd_alpha, cnfg['pgd']['iter'],
                           cnfg['pgd']['restarts'], 'l_inf', early_stop=False)
        delta = delta.detach()

        robust_output = model(normalize(torch.clamp(
            X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss = criterion(robust_output, y)

        output = model(normalize(X))
        loss = criterion(output, y)

        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output.max(1)
                            [1] == y).sum().item() / y.size(0)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item() / y.size(0)

        ind = np.where(
            (robust_output.max(1)[1] == y).cpu().data.numpy() == False)[0]
        misclassified_samples.append(X[ind].cpu().data.numpy())
        true_labels.append(y[ind].cpu().data.numpy())
        predicted_labels.append(robust_output.max(1)[1].cpu().data.numpy())

        conf_matrix = utils.confusion_matrix(robust_output, y, conf_matrix)

    print('test_robust_acc', test_robust_acc / len(tst_loader))
    print('test_acc', test_acc / len(tst_loader))

    np.save('./misclassified_overfit.npy', np.array(misclassified_samples))
    np.save('./truelabels_overfit.npy', np.array(true_labels))
    np.save('./predicted_labels_overfit', np.array(predicted_labels))

    if cnfg['data']['flag'] == '10':
        rng = get_text_classes()
    else:
        rng = range(0, int(cnfg['data']['flag']))
    utils.plot_confusion_matrix(
        conf_matrix.cpu().data.numpy(), rng)


if __name__ == '__main__':
    main()
