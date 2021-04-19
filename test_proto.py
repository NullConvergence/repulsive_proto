import apex.amp as amp
import argparse
import numpy as np
import torch
from tqdm import tqdm

from adv.pgd_clever import projected_gradient_descent
from cifar_data import mean, std, get_no_norm_trans, get_datasets, get_text_classes
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
    _, tst_loader = get_datasets(cnfg['data']['flag'],
                                 cnfg['data']['dir'],
                                 cnfg['data']['batch_size'],
                                 apply_transform=True)

    utils.set_seed(cnfg['seed'])
    device = torch.device(
        'cuda:0') if cnfg['gpu'] is None else torch.device(cnfg['gpu'])

    model = utils.get_model(cnfg['model']).to(device)
    checkpoint = torch.load(cnfg['resume']['path'])
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint

    model.load_state_dict(state)
    model.float()

    opt = torch.optim.SGD(model.parameters(),
                          lr=cnfg['train']['lr'],
                          momentum=cnfg['train']['momentum'],
                          weight_decay=cnfg['train']['weight_decay'])
    amp_args = dict(opt_level=cnfg['opt']['level'],
                    loss_scale=cnfg['opt']['loss_scale'], verbosity=False)
    if cnfg['opt']['level'] == '02':
        amp_args['master_weights'] = cnfg['opt']['store']

    model, opt = amp.initialize(model, opt, **amp_args)
    preproc_model = PreprocessingModel(model)

    protos = torch.from_numpy(np.load(cnfg['proto_path'])).float()
    loss_reduction = cnfg['train']['loss_reduction'] if cnfg['train']['loss_reduction'] is not None else "mean"
    proto = HyperProtoPGD(protos, device,
                          preproc_model,
                          cnfg['pgd'],
                          opt, None, loss_reduction)

    preproc_model.eval()
    t_acc, t_loss = 0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(tst_loader)):
            x, y = x.to(device), y.to(device)
            l, a = proto.test(x, y)
            t_loss += l
            t_acc += a
    print('Loss \t{0}\nAcc \t {1}'.format(
        t_loss/len(tst_loader), t_acc/len(tst_loader)*100))

    loss, acc = 0, 0
    conf_matrix = torch.zeros(
        int(cnfg['data']['flag']), int(cnfg['data']['flag'])).to(device)

    misclassified_samples, true_labels, predicted_labels = [], [], []
    for _, (x, y) in enumerate(tqdm(tst_loader)):
        x, y = x.to(device), y.to(device)

        vec_t = proto.get_vec_targets(y)
        x_ = projected_gradient_descent(proto, x, cnfg['pgd']['epsilon']/255,
                                        cnfg['pgd']['alpha']/255,
                                        cnfg['pgd']['iter'],
                                        np.inf,
                                        0, 1,
                                        y=vec_t,
                                        sanity_checks=False,
                                        model_loss=True)

        output = proto(x_)
        true_classes = proto.get_normal_targets(output)
        pred = true_classes.max(1, keepdim=True)[1]

        acc += pred.eq(y.view_as(pred)).sum().item() / len(y)

        ind = np.where(pred.eq(y.view_as(pred)).cpu().data.numpy() == True)[0]
        misclassified_samples.append(x[ind].cpu().data.numpy())
        true_labels.append(y[ind].cpu().data.numpy())
        predicted_labels.append(pred[ind].cpu().data.numpy())

        conf_matrix = utils.confusion_matrix(true_classes, y, conf_matrix)

    print('Adv Loss \t{0}\n Adv Acc \t {1}'.format(
        loss/len(tst_loader), acc/len(tst_loader)*100))

    # np.save('./correctclassified_proto.npy', np.array(misclassified_samples))
    # np.save('./correcttruelabels_proto.npy', np.array(true_labels))
    # np.save('./correctpredicted_labels', np.array(predicted_labels))

    if cnfg['data']['flag'] == '10':
        rng = get_text_classes()
    else:
        rng = range(0, int(cnfg['data']['flag']))
    utils.plot_confusion_matrix(
        conf_matrix.cpu().data.numpy(), rng)


if __name__ == "__main__":
    main()
