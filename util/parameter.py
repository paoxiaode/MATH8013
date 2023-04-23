import sys

import numpy as np
from .utils import get_y


def init_para(hidden_size=200, num_layer=4, ActFuc=1):
    R = {}
    R["times"] = 0.5  # initial
    R["input_dim"] = 1
    R["output_dim"] = 1
    
    R["ActFuc"] = ActFuc
    R["ActFre"] = 1

    R["num_layers"] = num_layer
    R["hidden_dim"] = hidden_size

    R["learning_rate"] = 2e-4
    R["learning_rateDecay"] = 5e-8

    R["train_size"] = 100
    R["test_size"] = 100

    R["log_dir"] = "tmp"


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
    lenarg = np.shape(sys.argv)[0]
    if lenarg > 1:
        ilen = 1
        while ilen < lenarg:
            if sys.argv[ilen] == "--num_layers":
                R["num_layers"] = np.int32(sys.argv[ilen + 1])
            if sys.argv[ilen] == "--hidden_dim":
                R["hidden_dim"] = np.int32(sys.argv[ilen + 1])
            if sys.argv[ilen] == "--act":
                R["ActFuc"] = np.int32(sys.argv[ilen + 1])
            if sys.argv[ilen] == "--dir":
                R["log_dir"] = sys.argv[ilen + 1]
            if sys.argv[ilen] == "--fre":
                R["ActFre"] = np.float32(sys.argv[ilen + 1])
            ilen = ilen + 2

    R["hidden_units"] = [R["hidden_dim"]] * R["num_layers"]
    R["batch_size"] = R["train_size"]

    R["astddev"] = 1 / (R["hidden_units"][0] ** R["times"])
    R["bstddev"] = 1 / (R["hidden_units"][0] ** R["times"])
    if R["input_dim"] == 1:
        R["full_net"] = [R["input_dim"]] + R["hidden_units"] + [R["output_dim"]]
    else:
        R["full_net"] = [60] + R["hidden_units"] + [R["output_dim"]]

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
    R["loss_test"] = []

    return R, Rw, Ry
