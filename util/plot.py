import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from .functions import *

from .utils import get_y, save_fig


def plot_weight(FolderName, R, Rw):
    weight_R = np.stack(Rw["weight_R"])
    plt.figure()
    for i_sub in range(R["n_fixed"]):
        # print(i_sub)
        for ji in range(3):
            # print('%s'%(3*i_sub+ji))
            ax = plt.subplot(R["n_fixed"], 3, 3 * i_sub + ji + 1)
            ax.plot(abs(weight_R[:, ji * R["n_fixed"] + i_sub]))
            plt.title("%s" % (3 * i_sub + ji))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim([5e-2, 1e1])
            # ax.axis('off')
            # ax.text(-0.5,1,'%.2f'%(output_weight[i_sub]))

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.legend(fontsize=18)
    # plt.title('loss',fontsize=15)
    # fntmp = '%shiddeny%s'%(FolderName,epoch)
    fntmp = "%sweightevolve" % (FolderName)
    save_fig(plt, fntmp, iseps=0)


def plot_loss(FolderName, R):

    plt.figure()
    ax = plt.gca()
    # y1 = R['loss_test']
    y2 = np.asarray(R["loss_train"])
    # plt.plot(y1,'ro',label='Test')
    plt.plot(y2, "k-", label="Train")
    if len(R["tuning_ind"]) > 0:
        plt.plot(R["tuning_ind"], y2[R["tuning_ind"]], "r*")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(fontsize=18)
    plt.title("loss", fontsize=15)
    fntmp = "%sloss" % (FolderName)
    save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)


# def plot_tuning(self, FolderName, R):
#     plt.figure()
#     ax = plt.gca()
#     y2 = R["y_true_train"]
#     plt.plot(train_inputs, y2, "b*", label="True")
#     for iit in range(len(R["y_tuning"])):
#         plt.plot(
#             test_inputs, R["y_tuning"][iit], "-", label="%.3f" % (R["loss_tuning"][iit])
#         )
#     plt.title("turn points", fontsize=15)
#     plt.legend(fontsize=18)
#     fntmp = "%sturn" % (FolderName)
#     save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_y(FolderName, R, train_inputs, test_inputs, name=""):
    train_inputs = train_inputs.cpu()
    test_inputs = test_inputs.cpu()

    if R["input_dim"] == 2:
        X = np.arange(R["x_start"], R["x_end"], 0.1)
        Y = np.arange(R["x_start"], R["x_end"], 0.1)
        X, Y = np.meshgrid(X, Y)
        xy = np.concatenate((np.reshape(X, [-1, 1]), np.reshape(Y, [-1, 1])), axis=1)
        Z = np.reshape(get_y(xy), [len(X), -1])

        fp = plt.figure()
        ax = fp.gca(projection="3d")
        surf = ax.plot_surface(
            X, Y, Z - np.min(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        fp.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(
            train_inputs[:, 0], train_inputs[:, 1], R["y_train"] - np.min(R["y_train"])
        )
        fntmp = "%s2du%s" % (FolderName, name)
        save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)

    if R["input_dim"] == 1:
        plt.figure()
        ax = plt.gca()
        y1 = R["y_test"]
        y2 = R["y_true_train"]
        plt.plot(test_inputs, y1, "r-", label="Test")
        plt.plot(train_inputs, y2, "b*", label="True")
        plt.title("g2u", fontsize=15)
        plt.legend(fontsize=18)
        fntmp = "%su_m%s" % (FolderName, name)
        fntmp = "%su_m%s" % (FolderName, "")
        save_fig(plt, fntmp, ax=ax, isax=1, iseps=0)


def save_file(FolderName, R, Ry, Rw):
    with open("%s/objs.pkl" % (FolderName), "wb") as f:
        pickle.dump(R, f, protocol=4)
    with open("%s/objsy.pkl" % (FolderName), "wb") as f:
        pickle.dump(Ry, f, protocol=4)
    with open("%s/objsw.pkl" % (FolderName), "wb") as f:
        pickle.dump(Rw, f, protocol=4)
    text_file = open("%s/Output.txt" % (FolderName), "w")
    for para in R:
        if np.size(R[para]) > 20:
            continue
        text_file.write("%s: %s\n" % (para, R[para]))
    text_file.write("loss end: %s\n" % (R["loss_train"][-1]))
    # text_file.write('weight ini: %s\n' % (Rw['weight_R'][0]))
    text_file.close()


def plot_result(FolderName, R, Ry):
    y_pred = R["y_train"]
    y_fft = my_fft(R["y_true_train"]) / R["train_size"]
    plt.semilogy(y_fft + 1e-5, label="real")
    idx = SelectPeakIndex(y_fft, endpoint=False)
    plt.semilogy(idx, y_fft[idx] + 1e-5, "o")
    y_fft_pred = my_fft(y_pred) / R["train_size"]
    plt.semilogy(y_fft_pred + 1e-5, label="train")
    plt.semilogy(idx, y_fft_pred[idx] + 1e-5, "o")
    plt.legend()
    plt.xlabel("freq idx")
    plt.ylabel("freq")
    plt.savefig(FolderName + "fft.png")

    y_pred_epoch = np.squeeze(Ry["y_all"])
    idx1 = idx[:3]
    abs_err = np.zeros([len(idx1), len(Ry["y_all"])])
    y_fft = my_fft(R["y_true_train"])
    tmp1 = y_fft[idx1]
    for i in range(len(y_pred_epoch)):
        tmp2 = my_fft(y_pred_epoch[i])[idx1]
        abs_err[:, i] = np.abs(tmp1 - tmp2) / (1e-5 + tmp1)

    plt.figure()
    plt.pcolor(abs_err, cmap="RdBu", vmin=0.1, vmax=1)
    plt.colorbar()
    plt.savefig(FolderName + "/hot.png")
