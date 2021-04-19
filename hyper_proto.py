import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class HyperProto(nn.Module):
    def __init__(self, proto, device, net, opt=None, sch=None, loss_reduction="mean"):
        super().__init__()
        # self.proto = proto.cuda()  # to(device)
        self.proto = proto.to(device)
        self.dev = device
        self.net = net
        self.opt = opt
        self.sch = sch
        self.loss_reduction = loss_reduction

    def forward(self, x):
        return self.net(x)

    def optimize(self, x, y, add_ce=False):
        out = self.forward(x)

        vec_t = self.get_vec_targets(y)
        loss = self.loss(out, vec_t)

        # This only works when the output of the class prototypess
        # is equal with the target shape
        if add_ce is True:
            assert out.size() == y.size(), "The prototype size must equal the targets size"
            ce_loss = F.cross_entropy(out, y)
            loss = loss + ce_loss

        self.opt.zero_grad()
        with amp.scale_loss(loss, self.opt) as scaled_loss:
            scaled_loss.backward()
        self.opt.step()

        true_classes = self.get_normal_targets(out)
        pred = true_classes.max(1, keepdim=True)[1]
        acc = pred.eq(y.view_as(pred)).sum().item() / true_classes.size()[0]
        self.sch.step()

        return loss.item(), acc

    def predict(self, x):
        out = self.forward(x)
        pred = self.get_normal_targets(out)
        return pred

    def test(self, x, y):
        with torch.no_grad():
            out = self.net(x)
            vec_t = self.get_vec_targets(y)
            loss = self.loss(out,  vec_t)
            true_classes = self.get_normal_targets(out)
            pred = true_classes.max(1, keepdim=True)[1]
            acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

            return loss, acc

    def get_vec_targets(self, targets):
        """this method assumes that the classes are ordered
         e.g. [0, 1, 2, ..] = text_classes order"""
        return self.proto[targets]

    def get_normal_targets(self, out):
        if self.loss_reduction == 'inf':
            p = float('inf')
        else:
            p = 2

        out = F.normalize(out, p=p, dim=1)
        out = torch.mm(out, self.proto.t().to(self.dev))
        # out = torch.mm(out, self.proto.t().cuda())  # to(self.dev))
        return out

    def loss_with_conversion(self, x, y, dim=0, eps=1e-8,):
        y = self.get_vec_targets(y)
        return self.loss(x, y, dim, eps)

    def loss(self, x, y, dim=0, eps=1e-8):
        if self.loss_reduction == 'inf':
            linfd = torch.nn.PairwiseDistance(float('inf'), 1e-06)
            res = linfd(x, y)
            return res.mean()
        if self.loss_reduction == 'inf_sum':
            linfd = torch.nn.PairwiseDistance(float('inf'), 1e-06)
            res = linfd(x, y)
            return res.sum()
        if self.loss_reduction == 'inf_pow':
            linfd = torch.nn.PairwiseDistance(float('inf'), 1e-06)
            res = linfd(x, y)
            return (1 - res.sum()).pow(2)
        if self.loss_reduction == "mean":
            sim = 1 - F.cosine_similarity(x, y, dim, eps)
            return sim.mean()
        elif self.loss_reduction == "power_sum":
            return (1 - F.cosine_similarity(x, y, dim, eps)).pow(2).sum()
        else:
            raise "The selected reduction does not work"


class HyperProtoPGD(HyperProto):
    def __init__(self, proto, device, net, pgd_conf, opt=None, sch=None, loss_reduction="mean", norm=255):
        super().__init__(proto, device, net, opt, sch, loss_reduction)
        self.eps = pgd_conf['epsilon'] / norm
        self.alpha = pgd_conf['alpha'] / norm
        self.rand_init = pgd_conf['restarts']
        self.iter = pgd_conf['iter']
        self.clip_min = 0
        self.clip_max = 1

    def pgd_forward(self, x, y, forward=False):
        vec_targets = self.get_vec_targets(y)
        x_ = self.attack(x, vec_targets)
        if not forward:
            return x_
        else:
            return self.net(x_), x_

    def optimize_pgd(self, x,  y):
        x_ = self.pgd_forward(x, y)
        return super().optimize(x_, y)

    def attack(self, x, vec_targets):
        x_ = x.detach()
        if self.rand_init == 'random':
            x_ = x_ + torch.zeros_like(x_).uniform_(-self.eps, self.eps)
            x_ = torch.clamp(x_, self.clip_min, self.clip_max)

        for _ in range(self.iter):
            x_.requires_grad_()
            with torch.enable_grad():
                logits = self.net(x_)
                loss = self.loss(logits, vec_targets)
            grad = torch.autograd.grad(loss, [x_])[0]
            x_ = x_.detach() + self.alpha*torch.sign(grad.detach())
            x_ = torch.min(torch.max(x_, x-self.eps), x+self.eps)
            x_ = torch.clamp(x_, self.clip_min, self.clip_max)
        return x_

    def test_pgd(self, x, y):
        with torch.no_grad():
            vec_t = self.get_vec_targets(y)
            x_ = self.attack(x, vec_t)
            out = self.net(x_)
            loss = self.loss(out,  vec_t)
            true_classes = self.get_normal_targets(out)
            pred = true_classes.max(1, keepdim=True)[1]
            acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
            return loss.item(), acc
