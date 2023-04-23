import sys

import numpy as np
import torch

from .utils import get_y


def init_para():
    R = {}
    R["times"] = 0.5  # initial
    R["input_dim"] = 1
    R["output_dim"] = 1
    R["ActFuc"] = 1  # 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
    R["hidden_units"] = [100, 100]

    R["learning_rate"] = 2e-4
    R["learning_rateDecay"] = 5e-8

    R["train_size"] = 100

    R["test_size"] = 100
    R["x_start"] = -5
    R["x_end"] = 5
    R["device"] = "0"
    R["asi"] = 0
    R["tuning_points"] = []
    R["check_epoch"] = 10  # find the tuning point
    R["tuning_ind"] = []
    Ry = {}
    Ry["y_all"] = []
    Rw = {}
    Rw["weight_R"] = []
    lenarg = np.shape(sys.argv)[
        0
    ]  # Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
    if lenarg > 1:
        ilen = 1
        while ilen < lenarg:
            if sys.argv[ilen] == "-m":
                R["hidden_units"] = [np.int32(sys.argv[ilen + 1])]
            if sys.argv[ilen] == "-g":
                R["device"] = np.int32(sys.argv[ilen + 1])
            if sys.argv[ilen] == "-t":
                R["times"] = np.float32(sys.argv[ilen + 1])
            if sys.argv[ilen] == "-s":
                R["train_size"] = np.int32(sys.argv[ilen + 1])
            # if sys.argv[ilen]=='-lr':
            #     R['learning_rate']=np.float32(sys.argv[ilen+1])
            # if sys.argv[ilen]=='-dir':
            #     sBaseDir=sys.argv[ilen+1]
            ilen = ilen + 2

    R["hidden_units"] = [200, 200, 200, 100]
    R["batch_size"] = R["train_size"]
    R["astddev"] = 1 / (R["hidden_units"][0] ** R["times"])
    R["bstddev"] = 1 / (R["hidden_units"][0] ** R["times"])
    R["full_net"] = [R["input_dim"]] + R["hidden_units"] + [R["output_dim"]]

    if R["input_dim"] == 1:
        R["test_inputs"] = np.reshape(
            np.linspace(
                R["x_start"] - 0.5, R["x_end"] + 0.5, num=R["test_size"], endpoint=True
            ),
            [R["test_size"], 1],
        )
        R["train_inputs"] = np.reshape(
            np.linspace(R["x_start"], R["x_end"], num=R["train_size"], endpoint=True),
            [R["train_size"], 1],
        )
    else:
        R["test_inputs"] = (
            np.random.rand(R["test_size"], R["input_dim"]) * (R["x_end"] - R["x_start"])
            + R["x_start"]
        )
        R["train_inputs"] = (
            np.random.rand(R["train_size"], R["input_dim"])
            * (R["x_end"] - R["x_start"])
            + R["x_start"]
        )
    R["y_true_train"] = get_y(R["train_inputs"])
    R["n_fixed"] = 0
    min_n = np.min([R["n_fixed"], R["hidden_units"][0]])
    R["n_fixed"] = min_n
    R["loss_train"] = []

    return R, Rw, Ry
