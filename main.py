import time
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
from Network import DNNNetwork
from util import init_para, plot_loss, plot_y, save_file, plot_result, make_folder
from util.functions import getWini

# Make folders for output
FolderName = make_folder()

# All parameters
R, Rw, Ry = init_para()
plot_epoch = 500

test_inputs = torch.FloatTensor(R["test_inputs"]).to("cuda:0")
train_inputs = torch.FloatTensor(R["train_inputs"]).to("cuda:0")
y_true_train = torch.FloatTensor(R["y_true_train"]).to("cuda:0")

# print(min_n)
w_Univ0, b_Univ0 = getWini(
    hidden_units=R["hidden_units"],
    input_dim=R["input_dim"],
    output_dim_final=R["output_dim"],
    astddev=R["astddev"],
    bstddev=R["bstddev"],
)

print(np.shape(w_Univ0[0]))
print(np.shape(b_Univ0[0]))


def evaluate(model, loss, test_inputs, train_inputs, y_true_train):
    model.eval()
    y_test = model(test_inputs)
    # loss_test = float(self.loss(y_test, torch.FloatTensor(R['y_true_test']).to(device)).cpu())
    y_train = model(train_inputs)
    loss_train = loss(y_train, y_true_train)
    return y_test, y_train, loss_train


def run(network, step_n=1):
    # Load paremeters
    # nametmp = "%smodel/model.ckpt" % (FolderName)
    # network.load_state_dict(torch.load(nametmp))
    network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=2e-4)
    loss_fcn = nn.MSELoss(reduction="mean")

    for epoch in range(step_n):
        y_test, y_train, loss_train = evaluate(
            network, loss_fcn, test_inputs, train_inputs, y_true_train
        )
        R["y_train"] = y_train.cpu().detach().numpy()
        R["y_test"] = y_test.cpu().detach().numpy()
        R["loss_train"].append(loss_train.cpu().detach().numpy())

        t0 = time.time()
        for i in range(R["train_size"] // R["batch_size"] + 1):  # bootstrap
            mask = torch.randperm(R["train_size"])[: R["batch_size"]]
            y_train = network(train_inputs[mask])
            loss = loss_fcn(y_train, y_true_train[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # self.record_weight()
        Ry["y_all"].append(R["y_train"])
        R["learning_rate"] = R["learning_rate"] * (1 - R["learning_rateDecay"])
        if epoch % plot_epoch == 0:
            print("time elapse: %.3f" % (time.time() - t0))
            print("model, epoch: %d, train loss: %f" % (epoch, R["loss_train"][-1]))
            plot_loss(FolderName, R)
            plot_y(FolderName, R, train_inputs, test_inputs, name="%s" % (epoch))
            save_file(FolderName, R, Ry, Rw)

        if R["loss_train"][-1] < 1e-5:
            break


def main():
    device = torch.device(
        "cuda:%s" % (R["device"]) if torch.cuda.is_available() else "cpu"
    )
    # device = torch.device("cpu")
    print(device)

    network = DNNNetwork(R, w_Univ0, b_Univ0).to(device)
    print(network)

    run(network, 3000)
    plot_result(FolderName, R, Ry)


if __name__ == "__main__":
    main()
