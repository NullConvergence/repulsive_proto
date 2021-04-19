import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LOG = True
K = 1
LOG = True
N = 100
CLASSES = 100
EPOCHS = 50

torch.set_printoptions(threshold=10_000)


def get_proto(dim, n):
    return torch.Tensor(np.random.uniform(low=-1, high=1, size=(n, dim)))


def norm(vec):
    return torch.nn.functional.pdist(vec, float('inf'))


def loss(vec, r_s, k):
    norm_ = norm(vec)
    if LOG:
        print('Norm: \t', norm_)
    loss = torch.sum(r_s) + (k * torch.sum(norm_))

    return - loss


def distance_proto(dim, n, epc):
    centres = get_proto(dim, n)
    cent = nn.Parameter(centres, requires_grad=True)
    r_s = torch.Tensor(np.full([n], 1))
    k = torch.Tensor([1])
    opt = optim.SGD([cent], lr=0.01, momentum=.9)
    for _ in range(epc):
        opt.zero_grad()
        out = loss(cent, r_s, k)
        out.backward()
        opt.step()

    return cent


if __name__ == '__main__':
    cent = distance_proto(N, CLASSES, EPOCHS)
    # print(cent)
    proto = cent.data.numpy()  # np.array([x.data.numpy() for x in cent])
    np.save("./prototypes/prototypes-n-%dd-1k-%de-%dcl.npy" % (N, EPOCHS, CLASSES),
            proto)
