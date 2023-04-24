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

# Load all parameters
R, Rw, Ry = init_para()
plot_epoch = 500

# Make folders for output
FolderName = make_folder(R)

# Load all inputs/labels
test_inputs = torch.FloatTensor(R["test_inputs"]).to("cuda:0")
train_inputs = torch.FloatTensor(R["train_inputs"]).to("cuda:0")
y_true_train = torch.FloatTensor(R["y_true_train"]).to("cuda:0")

# Load the linear layer weight and bias
w_Univ0, b_Univ0 = getWini(
    hidden_units=R["hidden_units"],
    input_dim=R["input_dim"] if R["input_dim"] == 1 else 60,
    output_dim_final=R["output_dim"],
    astddev=R["astddev"],
    bstddev=R["bstddev"],
)

def evaluate(model, loss, test_inputs, train_inputs, y_true_train):
    model.eval()
    y_test = model(test_inputs)
    y_train = model(train_inputs)
    loss_train = loss(y_train, y_true_train)
    loss_test = loss(y_test, y_true_train)
    return y_test, y_train, loss_train, loss_test


def run(network, step_n=1):
    optimizer = torch.optim.Adam(network.parameters(), lr=R["learning_rate"], weight_decay=R["learning_rateDecay"])
    loss_fcn = nn.MSELoss(reduction="mean")
    loss_train_old = 1e5
    for epoch in range(step_n):
        # Evaluate the model
        y_test, y_train, loss_train, loss_test = evaluate(
            network, loss_fcn, test_inputs, train_inputs, y_true_train
        )
        R["y_train"] = y_train.cpu().detach().numpy()
        R["y_test"] = y_test.cpu().detach().numpy()
        R["loss_train"].append(loss_train.cpu().detach().numpy())
        R["loss_test"].append(loss_test.cpu().detach().numpy())

        t0 = time.time()
        # Train the model on each batch
        for _ in range(R["train_size"] // R["batch_size"] + 1):  # bootstrap
            mask = torch.randperm(R["train_size"])[: R["batch_size"]]
            y_train = network(train_inputs[mask])
            loss = loss_fcn(y_train, y_true_train[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Ry["y_all"].append(R["y_train"])

        # Plot result every plot_epoch
        if epoch % plot_epoch == 0:
            print("time elapse: %.3f" % (time.time() - t0))
            print("model in epoch: %d, train loss: %f" % (epoch, loss_train))
            plot_loss(FolderName, R)
            plot_y(FolderName, R, train_inputs, test_inputs, name="%s" % (epoch))
            save_file(FolderName, R, Ry, Rw)
            loss_train_old = loss_train

        # Terminal training
        # if loss_train < 1e-5:
        #     break

        


def main():
    # Init device
    device = torch.device(
        "cuda:%s" % (R["device"]) if torch.cuda.is_available() else "cpu"
    )
    print(device)

    # Init model
    network = DNNNetwork(R, w_Univ0, b_Univ0).to(device)
    print(network)
    
    # Kick off training
    run(network, 2000)

    # Plot the result
    plot_result(FolderName, R, Ry)


if __name__ == "__main__":
    main()
