import torch
import torch.nn.functional as F


def attack(x, y_, model, criterion=F.cross_entropy, iter=20,
           alpha=2., eps=8., rand_init=None,
           clip_min=0., clip_max=1.):
    with torch.no_grad():
        x_ = x.detach()
        if rand_init == 'random':
            x_ = x_ + torch.zeros_like(x_).uniform_(-eps, eps)
            x_ = torch.clamp(x_, clip_min, clip_max)

        for _ in range(iter):
            x_.requires_grad_()
            with torch.enable_grad():
                logits = model(x_)
                loss = criterion(logits, y_)
            grad = torch.autograd.grad(loss, [x_])[0]
            x_ = x_.detach() + alpha*torch.sign(grad.detach())
            x_ = torch.min(torch.max(x_, x-eps), x+eps)
            x_ = torch.clamp(x_, clip_min, clip_max)

        return x_
