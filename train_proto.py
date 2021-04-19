import apex.amp as amp
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm

from cifar_data import mean, std, get_no_norm_trans, get_datasets
from model_preproc import PreprocessingModel
from hyper_proto import HyperProto, HyperProtoPGD
from logger import Logger
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./cnfg.yml', type=str)
    return parser.parse_args()


def main():
    # config
    args = parse_args()
    cnfg = utils.parse_config(args.config)
    # data
    tr_loader, tst_loader = get_datasets(cnfg['data']['flag'],
                                         cnfg['data']['dir'],
                                         cnfg['data']['batch_size'],
                                         'no_norm')

    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])
    logger = Logger(cnfg)
    model = utils.get_model(cnfg['model']).to(device)

    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])
    amp_args = dict(opt_level=cnfg['opt']['level'],
                    loss_scale=cnfg['opt']['loss_scale'], verbosity=False)
    if cnfg['opt']['level'] == '02':
        amp_args['master_weights'] = cnfg['opt']['store']
    model, opt = amp.initialize(model, opt, **amp_args)
    scheduler = utils.get_scheduler(
        opt, cnfg['train'], cnfg['train']['epochs']*len(tr_loader))

    preproc_model = PreprocessingModel(model)
    proto_path = os.path.join(os.path.dirname(
        os.path.abspath("train_proto.py")), cnfg['proto_path'])
    protos = torch.from_numpy(np.load(proto_path)).float()
    logger.log_file(proto_path)
    loss_reduction = cnfg['train']['loss_reduction'] if cnfg['train']['loss_reduction'] is not None else "mean"
    proto = HyperProtoPGD(protos, device,
                          preproc_model,
                          cnfg['pgd'],
                          opt, scheduler, loss_reduction)

    for epoch in range(cnfg['train']['epochs']):
        preproc_model.train()
        acc, loss = 0, 0
        for _, (x, y) in enumerate(tqdm(tr_loader)):
            x, y = x.to(device), y.to(device)
            l, a = proto.optimize(x, y)
            loss += l
            acc += a
        utils.log_lr(logger, opt, epoch)
        logger.log_train(epoch, loss/len(tr_loader),
                         (acc/len(tr_loader))*100, "clean_training")
        if (epoch+1) % cnfg['save']['epochs'] == 0 and epoch > 0:
            pth = 'models/' + cnfg['logger']['project'] + '_' + \
                cnfg['logger']['run'] + '_' + str(epoch) + '.pth'
            utils.save_model(model, cnfg, epoch, pth)
            logger.log_model(pth)

        # testing
        if (epoch+1) % cnfg['test'] == 0 or epoch == 0:
            preproc_model.eval()
            t_acc, t_loss = 0, 0

            for _, (x, y) in enumerate(tqdm(tst_loader)):
                with torch.no_grad():
                    x, y = x.to(device), y.to(device)
                    l, a = proto.test(x, y)
                    t_loss += l
                    t_acc += a
            logger.log_test(epoch, t_loss/len(tst_loader),
                            (t_acc/len(tst_loader))*100, "clean_testing")

            # test pgd
            t_loss, t_acc = 0, 0
            for _, (x, y) in enumerate(tqdm(tst_loader)):
                x, y = x.to(device), y.to(device)
                l, a = proto.test_pgd(x, y)
                t_loss += l
                t_acc += a
            logger.log_test_adversarial(epoch, t_loss/len(tst_loader),
                                        (t_acc/len(tst_loader))*100, "pgd_testing")


if __name__ == '__main__':
    main()
