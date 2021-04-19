from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm


class T:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.t = TSNE(**kwargs)
        self.repr = None

    def fit(self, x, redo=False):
        if self.repr is not None and redo is False:
            return self.repr
        else:
            self.repr = self.t.fit_transform(x)
            return self.repr

    def normalize(self):
        assert self.repr is not None, "[ERROR] Please call fit first!"
        tx, ty = self.repr[:, 0], self.repr[:, 1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
        return tx, ty

    def viz_features(self, y, classes, save=False, path='./', name='fc1_features_tsne_default_pts.jpg'):

        # save proto
        # np.save("embeddings.npy", self.repr)
        # self.repr = np.load("embeddings.npy")

        y_test = np.asarray(y)
        tx, ty = self.normalize()

        print('y_test', y_test)
        print('y', y)
        fig = plt.figure(figsize=(16, 12))
        for i in range(len(classes)):
            y_i = y_test == i
            # print(y_i)
            print(tx[y_i].shape)
            plt.scatter(tx[y_i], ty[y_i], label=classes[i])
        fig.legend(loc=4)
        fig.gca().invert_yaxis()
        if save:
            fig.savefig(os.path.join(
                path, name), bbox_inches='tight')
        return fig, plt

        # plt.figure(figsize=(16, 12))
        # cmap = cm.get_cmap('jet', 10)
        # plt.scatter(x=self.repr[:, 0], y=self.repr[:, 1],
        #             c=y_test.reshape(50000), s=35, cmap=cmap)
        # # plt.scatter(x=self.repr[:, 0], y=self.repr[:, 1])
        # # plt.title("t-sne on vgg16 features cifar10")
        # plt.colorbar()
        # plt.show()
