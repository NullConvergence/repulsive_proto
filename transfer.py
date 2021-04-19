import apex.amp as amp
import argparse
import numpy as np
import torch
from tqdm import tqdm

from adv.pgd_clever import projected_gradient_descent
from cifar_data import mean, std, get_no_norm_trans, get_datasets
from model_preproc import PreprocessingModel
from hyper_proto import HyperProto, HyperProtoPGD
from logger import Logger
from test_pgd_overfit import *
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./cnfg_transfer.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data
    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'],
                                 'no_norm')

    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])

    # normal model is the regular model
    model = utils.get_model(cnfg['model']).cuda()  # .to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state)
    model.float()

    # normal model is the pgd overfit model
    # model = utils.get_model(cnfg['model']).cuda()
    # if cnfg['data']['flag'] == '10':
    #     model = nn.DataParallel(model).cuda()
    # checkpoint = torch.load(cnfg['resume']['path'])
    # state = checkpoint
    # model.load_state_dict(state)

    preproc_model = model

    # test model is the proto model
    test_model = utils.get_model(cnfg['test_model']).cuda()  # to(device)
    checkpoint = torch.load(cnfg['test_model_resume']['path'])
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint

    test_model.load_state_dict(state)
    test_model.float()

    preproc_model_test = PreprocessingModel(test_model)

    protos = torch.from_numpy(np.load(cnfg['proto_path'])).float()
    loss_reduction = cnfg['train']['loss_reduction'] if cnfg['train']['loss_reduction'] is not None else "mean"
    proto = HyperProtoPGD(protos, device,
                          preproc_model_test,
                          cnfg['pgd'],
                          None, None, loss_reduction)

    # test model is the pgd overfit model
    # test_model = utils.get_model(cnfg['test_model']).cuda()
    # if cnfg['data']['flag'] == '10':
    #     test_model = nn.DataParallel(test_model).cuda()
    # checkpoint = torch.load(cnfg['test_model_resume']['path'])
    # state = checkpoint
    # test_model.load_state_dict(state)

    preproc_model.eval()
    preproc_model_test.eval()

    loss, acc = 0, 0
    epsilon = (cnfg['pgd']['epsilon'] / 255.)
    pgd_alpha = (cnfg['pgd']['alpha'] / 255.)

    for _, (x, y) in enumerate(tqdm(tst_loader)):
        # x, y = x.to(device), y.to(device)
        x, y = x.cuda(), y.cuda()
        # vec_t = proto.get_vec_targets(y)
        # get adv. examples from normal model
        x_ = projected_gradient_descent(preproc_model, x, cnfg['pgd']['epsilon']/255,
                                        cnfg['pgd']['alpha']/255,
                                        cnfg['pgd']['iter'],
                                        np.inf,
                                        0, 1,
                                        sanity_checks=False)

        # get adv. examples from pgd overfit
        # delta = attack_pgd(model, x, y, epsilon, pgd_alpha, cnfg['pgd']['iter'],
        #                    cnfg['pgd']['restarts'], 'l_inf', early_stop=False)
        # delta = delta.detach()

        # # test with prototype model
        # x_ = normalize(torch.clamp(
        #     x + delta[:x.size(0)], min=lower_limit, max=upper_limit))

        output = proto(x_)
        true_classes = proto.get_normal_targets(output)
        pred = true_classes.max(1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item() / len(y)

        # output = test_model(x_)
        # acc += (output.max(1)[1] == y).sum().item() / len(y)

    print('Adv Loss \t{0}\n Adv Acc \t {1}'.format(
        loss/len(tst_loader), acc/len(tst_loader)*100))


if __name__ == "__main__":
    main()
