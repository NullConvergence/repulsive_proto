import apex.amp as amp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from adv.pgd import attack
from adv.pgd_clever import projected_gradient_descent
from cifar_data import mean, std, get_no_norm_trans, get_datasets
from model_preproc import PreprocessingModel
from hyper_proto import HyperProto, HyperProtoPGD
from logger import Logger
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./cnfg_clean.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data

    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'],
                                 cnfg['data']['trans'])

    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])

    model = utils.get_model(cnfg['model']).to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state)
    model.float()

    preproc_model = model  # PreprocessingModel(model)
    preproc_model.eval()

    tr_acc, tr_loss = 0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(tst_loader)):
            x, y = x.to(device), y.to(device)
            output = preproc_model(x)
            loss = F.cross_entropy(output, y)
            tr_acc += (output.max(1)[1] == y).sum().item() / len(y)
            tr_loss += loss.item()
        print('Loss \t{0}\nAcc \t {1}'.format(
            tr_loss/len(tst_loader), tr_acc/len(tst_loader)*100))

    pgd_conf = cnfg['pgd']
    eps = pgd_conf['epsilon'] / 255
    alpha = pgd_conf['alpha'] / 255
    rand_init = pgd_conf['restarts']
    iter = pgd_conf['iter']
    clip_min = 0
    clip_max = 1

    tst_loss, tst_acc = 0, 0

    for _, (x, y) in enumerate(tqdm(tst_loader)):
        x, y = x.to(device), y.to(device)

        x_ = projected_gradient_descent(preproc_model, x, eps, alpha,
                                        iter, np.inf,
                                        clip_min, clip_max,
                                        sanity_checks=False)

        output = preproc_model(x_)
        loss = F.cross_entropy(output, y)
        tst_acc += (output.max(1)[1] == y).sum().item() / len(y)
        tst_loss += loss.item()

    print('Adv Loss \t{0}\n Adv Acc \t {1}'.format(
        tst_loss/len(tst_loader), tst_acc/len(tst_loader)*100))


if __name__ == "__main__":
    main()
