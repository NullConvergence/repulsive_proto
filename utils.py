import models
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def parse_config(path):
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def get_model(cnfg):
    if cnfg['custom'] is True:
        kwargs = cnfg['kwargs'] if 'kwargs' in cnfg else {}
        return models.get_from_zoo(cnfg['arch'], kwargs)
    else:
        return models.get_tvision(cnfg['tvision']['name'], cnfg['tvision']['args'])


def get_scheduler(opt, cnfg, steps):
    if cnfg['lr_scheduler'] == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(opt,
                                                 base_lr=cnfg['lr_min'],
                                                 max_lr=cnfg['lr_max'],
                                                 step_size_up=cnfg['cyclic_step_size'],
                                                 step_size_down=cnfg['cyclic_step_size'])

    elif cnfg['lr_scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(opt,
                                               step_size=cnfg['step'],
                                               gamma=cnfg['gamma'])

    elif cnfg['lr_scheduler'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(opt,
                                                    milestones=cnfg['milestones'],
                                                    gamma=cnfg['gamma'])
    else:
        raise NotImplementedError(
            "[ERROR] The selected scheduler is not implemented")


def save_model(model, cnf, epoch, path, proto=None):
    state = {
        'epoch': epoch,
        'cnf': cnf,
        'arch': type(model).__name__,
        'model': model.state_dict(),
        'proto': proto
    }
    torch.save(state, path)


def get_lr(opt):
    lrs = []
    for param_group in opt.param_groups:
        lrs.append(param_group["lr"])
    return lrs


def adjust_lr(opt, sc, log, stp, do_log=True):
    sc.step()
    if do_log is True:
        log_lr(log, opt, stp)


def log_lr(log, opt, stp):
    lr = get_lr(opt)
    log.log_lr(lr, stp)


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_confusion_matrix(cf, classes):
    fig, ax = plt.subplots()
    im = ax.imshow(cf)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cf[i, j],
                           ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()
