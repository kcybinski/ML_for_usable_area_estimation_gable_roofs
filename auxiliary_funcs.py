from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

import sklearn

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy import diag, interp
from itertools import cycle

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from termcolor import cprint

import plotly.graph_objects as go

import plotly.express as px

import time

class DeepNet(nn.Module):   
    def __init__(self, input_size, output_size, hidden_size_1st, hidden_size_2nd = None, hidden_size_3rd = None, hidden_size_4th = None, dropProb = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_1st = hidden_size_1st
        self.hidden_size_2nd = hidden_size_2nd
        self.hidden_size_3rd = hidden_size_3rd
        self.hidden_size_4th = hidden_size_4th
        super(DeepNet, self).__init__()
        self.l1 = nn.Linear(self.input_size, self.hidden_size_1st, bias=True)
        self.l1_drop = nn.Dropout(p = dropProb)

        if self.hidden_size_2nd is None and self.hidden_size_3rd is None and self.hidden_size_4th is None:
            self.l2 = nn.Linear(self.hidden_size_1st, self.output_size, bias=True)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is None and self.hidden_size_4th is None:
            self.l2 = nn.Linear(self.hidden_size_1st, self.hidden_size_2nd, bias=True)
            self.l2_drop = nn.Dropout(p = dropProb)
            self.l3 = nn.Linear(self.hidden_size_2nd, self.output_size, bias=True)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is not None and self.hidden_size_4th is None:
            self.l2 = nn.Linear(self.hidden_size_1st, self.hidden_size_2nd, bias=True)
            self.l2_drop = nn.Dropout(p = dropProb)
            self.l3 = nn.Linear(self.hidden_size_2nd, self.hidden_size_3rd, bias=True)
            self.l3_drop = nn.Dropout(p = dropProb)
            self.l4 = nn.Linear(self.hidden_size_3rd, self.output_size, bias=True)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is not None and self.hidden_size_4th is not None:
            self.l2 = nn.Linear(self.hidden_size_1st, self.hidden_size_2nd, bias=True)
            self.l2_drop = nn.Dropout(p = dropProb)
            self.l3 = nn.Linear(self.hidden_size_2nd, self.hidden_size_3rd, bias=True)
            self.l3_drop = nn.Dropout(p = dropProb)
            self.l4 = nn.Linear(self.hidden_size_3rd, self.hidden_size_4th, bias=True)
            self.l4_drop = nn.Dropout(p = dropProb)
            self.l5 = nn.Linear(self.hidden_size_4th, self.output_size, bias=True)

    def forward(self, x):
        x = F.relu(self.l1_drop(self.l1(x)))

        if self.hidden_size_2nd is None and self.hidden_size_3rd is None and self.hidden_size_4th is None:
            out = self.l2(x)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is None and self.hidden_size_4th is None:
            x = F.relu(self.l2_drop(self.l2(x)))
            out = self.l3(x)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is not None and self.hidden_size_4th is None:
            x = F.relu(self.l2_drop(self.l2(x)))
            x = F.relu(self.l3_drop(self.l3(x)))
            out = self.l4(x)
        if self.hidden_size_2nd is not None and self.hidden_size_3rd is not None and self.hidden_size_4th is not None:
            x = F.relu(self.l2_drop(self.l2(x)))
            x = F.relu(self.l3_drop(self.l3(x)))
            x = F.relu(self.l4_drop(self.l4(x)))
            out = self.l5(x)
        
        return out #F.log_softmax(out,dim=1)
def plot_df_summary(df_test_viz, plot_title=None, hover=["No.", "Usable area [m2]", "Predicted Usable Area [m2]", "Number of stories"]):
    scat = px.scatter(df_test_viz, x="Usable area [m2]", y="Predicted Usable Area [m2]", hover_data=hover, color="Error", color_continuous_scale="Bluered", width=800, height=800, title=plot_title)

    scat.add_scatter(x=np.linspace(60, 200, df_test_viz["Usable area [m2]"].to_numpy().shape[0]), y=np.linspace(60, 200, df_test_viz["Usable area [m2]"].to_numpy().shape[0]))

    scat.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="LightSteelBlue",
    )

    scat.add_annotation(x=70, y=200,
                text="R^2 = {:.2f} <br>Mean Absolute Error = {:.2f}<br>Median Absolute Error = {:.2f}<br>Max Error = {:.2f}<br>Min Error = {:.2f}<br>Median Absolute Percentage Error = {:.2f}".format(
                                                                            sklearn.metrics.r2_score(df_test_viz["Usable area [m2]"].to_numpy(), df_test_viz["Predicted Usable Area [m2]"].to_numpy()), 
                                                                            sklearn.metrics.mean_absolute_error(df_test_viz["Usable area [m2]"].to_numpy(), df_test_viz["Predicted Usable Area [m2]"].to_numpy()),
                                                                            sklearn.metrics.median_absolute_error(df_test_viz["Usable area [m2]"].to_numpy(), df_test_viz["Predicted Usable Area [m2]"].to_numpy()),
                                                                            sklearn.metrics.max_error(df_test_viz["Usable area [m2]"].to_numpy(), df_test_viz["Predicted Usable Area [m2]"].to_numpy()),
                                                                            np.abs(np.min(df_test_viz["Error"])),
                                                                            sklearn.metrics.mean_absolute_percentage_error(df_test_viz["Usable area [m2]"].to_numpy(), df_test_viz["Predicted Usable Area [m2]"].to_numpy())
                                                                            ),
                font=dict(
                family="Serif",
                size=16,
                color="#000000"
                ),
            align="center",
                bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#B3FCF5",
            opacity=0.8,
            showarrow=False
                )

    scat.show()

def regression(df, used_cols=["Covered area [m2]", "Height [m]"], dataset_name = "Without garages or boiler rooms", model_type = 'Regression', rs=42, out=False, which_set='test', show_plot=True):
    df_train_and_val, df_test = train_test_split(df, test_size=0.2, random_state=rs)

    df_test = df_test.reset_index()

    Y_test = df_test["Usable area [m2]"]
    X_test = df_test[used_cols].to_numpy()

    df_train, df_val = train_test_split(df_train_and_val, test_size=0.25, random_state=rs)

    Y_train = df_train["Usable area [m2]"]
    X_train = df_train[used_cols].to_numpy()

    Y_val = df_val["Usable area [m2]"]
    X_val = df_val[used_cols].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    model_lin = Ridge(random_state=rs)
    Y_train = Y_train.ravel()
    model_lin.fit(X_train, Y_train)
    if which_set == 'test':
        Y_pred = model_lin.predict(X_test)
    elif which_set == 'val':
        Y_pred = model_lin.predict(X_val)
    else:
        raise ValueError("Incorrect set value!")

    df_pred = pd.DataFrame(Y_pred, columns=["Predicted Usable Area [m2]"])

    df_test_viz = df_test.join(df_pred)
    df_test_viz["Error"] = np.abs(df_test_viz["Usable area [m2]"] - df_test_viz["Predicted Usable Area [m2]"])

    if show_plot:
        print(f"Dataset -> {dataset_name}\n"+f"Model -> {model_type}\n\n"+"Used data columns: \n> "+"\n> ".join(used_cols)+"\n\n"+f"R^2 = {r2_score(Y_test, Y_pred):.2f}")
        print("Mean Absolute Error = {:.2f}".format(sklearn.metrics.mean_absolute_error(Y_test, Y_pred)))
        print("Median Absolute Error = {:.2f}".format(sklearn.metrics.median_absolute_error(Y_test, Y_pred)))
        print("Max Error = {:.2f}".format(sklearn.metrics.max_error(Y_test, Y_pred)))
        print("Min Error = {:.2f}".format(np.min(np.abs(Y_test-Y_pred))))
        print("Median Absolute Percentage Error = {:.2f}".format(sklearn.metrics.mean_absolute_percentage_error(Y_test, Y_pred)))
        plot_df_summary(df_test_viz)

    if out:
        return df_test_viz

def test_model(df, model, used_cols, scaler, model_type, dataset_name, which_set="test"):
    X_test = df[used_cols].to_numpy()
    Y_test = df["Usable area [m2]"]
    X_test = scaler.transform(X_test)
    test_features = Variable(torch.from_numpy(X_test))
    test_labels = Variable(torch.from_numpy(Y_test.to_numpy()).double())
    # Model evaluation
    with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            Y_pred = model(test_features).numpy().reshape(-1)
            df_pred = pd.DataFrame(Y_pred, columns=["Predicted Usable Area [m2]"])
            df_test_viz = df.join(df_pred)
        
    df_test_viz["Error"] = np.abs(df_test_viz["Usable area [m2]"] - df_test_viz["Predicted Usable Area [m2]"])


    cprint("\n[INFO]", "magenta", end=" ")
    print(f"Dataset -> {dataset_name}")
    cprint("[INFO]", "magenta", end=" ")
    print(f"Model -> {model_type}\n\n")
    cprint("[INFO]", "magenta", end=" ")
    print("Used data columns: \n-----> "+"\n-----> ".join(used_cols)+"\n\n")
    cprint("[INFO]", "magenta", end=" ")
    print(f"R^2 = {r2_score(Y_test, Y_pred):.2f}")
    cprint("[INFO]", "magenta", end=" ")
    print("Mean Absolute Error = {:.2f}".format(sklearn.metrics.mean_absolute_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Median Absolute Error = {:.2f}".format(sklearn.metrics.median_absolute_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Max Error = {:.2f}".format(sklearn.metrics.max_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Min Error = {:.2f}".format(np.min(np.abs(Y_test-Y_pred))))
    cprint("[INFO]", "magenta", end=" ")
    print("Median Absolute Percentage Error = {:.2f}".format(sklearn.metrics.mean_absolute_percentage_error(Y_test, Y_pred)))

    plot_df_summary(df_test_viz, plot_title=f"{model_type} | {which_set} set")
def nn_for_usable_area(df_in, used_cols=["Covered area [m2]", "Height [m]"], learning_rate=0.5, λ=0.35, num_epochs=2000, out=True, 
                        architecture=DeepNet, arch_params=(2, 1, 4), verb=False, dataset_name="", which_set='test', 
                        rs=42, milestones=[100, 200, 350, 450, 1700, 1900]):
    np.random.seed(rs)
    torch.manual_seed(rs);
    
    model = architecture(*arch_params).double()
    model_type = 'Neural Network'

    model_name = f'DNN_lr={str(learning_rate).replace(".", ",")}_|_λ ={str(λ).replace(".", ",")}'



    df_train_and_val, df_test = train_test_split(df_in, test_size=0.2, random_state=rs)

    df_test = df_test.reset_index()

    Y_test = df_test["Usable area [m2]"]
    X_test = df_test[used_cols].to_numpy()

    df_train, df_val = train_test_split(df_train_and_val, test_size=0.25, random_state=rs)

    df_val = df_val.reset_index()
    df_train = df_train.reset_index()

    Y_train = df_train["Usable area [m2]"]
    X_train = df_train[used_cols].to_numpy()

    Y_val = df_val["Usable area [m2]"]
    X_val = df_val[used_cols].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    training_set_size = X_train.shape[0]
    batch_size = X_train.shape[0]
    train_features = Variable(torch.from_numpy(X_train))
    train_labels = Variable(torch.from_numpy(Y_train.to_numpy()).double())
    val_features = Variable(torch.from_numpy(X_val))
    val_labels = Variable(torch.from_numpy(Y_val.to_numpy()).double())
    test_features = Variable(torch.from_numpy(X_test))
    test_labels = Variable(torch.from_numpy(Y_test.to_numpy()).double())



    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # For SGD stochasticity
    permutation = np.arange(train_features.size()[0])
    hold_loss=[]
    hold_val_loss = []
    total_step = train_features.size()[0]

    # Model training

    cprint("[INFO]", "magenta", end=" ")
    print("training the network...")
    startTime = time.time()
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                for i in range(0, train_features.size()[0], batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_features, batch_labels = train_features[indices], train_labels[indices]
        
                    # Forward pass
                    outputs = model(batch_features)
                    loss = criterion(torch.reshape(outputs, (-1,)).double(), batch_labels)
        
                    # We manually add L2 regularization
                    if λ != 0:
                        l2_reg = 0.0
                        for param in model.parameters():
                            l2_reg += torch.norm(param)**2
                        loss += 1/training_set_size * λ/2 * l2_reg

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True) # This retain graph is important for 2nd gradient
                    optimizer.step()
                    scheduler.step() # here for milestones scheduler
                    
                    if verb:
                        cprint("[INFO]", "magenta", end=" ")
                        print ('Epoch [{}/{}], Train loss: {:.4f}' 
                                .format(epoch+1, num_epochs, loss.item()))
                    else:
                        print_mult = num_epochs//5
                        if epoch%print_mult == 0:
                            cprint("[INFO]", "magenta", end=" ")
                            print ('Epoch [{}/{}], Train loss: {:.4f}' 
                                    .format(epoch+1, num_epochs, loss.item()))

                    hold_loss.append(loss.item())

            if phase == 'val':
                predicted_labels_storage = []
                predicted_val_storage = []
                with torch.no_grad():
                    for i in range(len(val_features)):
                        outputs = model(val_features[i:i+1]) # this [i:i+1] is just for one extra embedding, 'cause criterion expects batches
                        val_loss = criterion(torch.reshape(outputs, (-1,)).double(), val_labels[i:i+1]) # the same as above

                        # We manually add L2 regularization
                        if λ != 0:
                            l2_reg = 0.0
                            for param in model.parameters():
                                l2_reg += torch.norm(param)**2
                            val_loss += 1/training_set_size * λ/2 * l2_reg

                        predicted_labels_storage.append(outputs.item())
                        predicted_val_storage.append(val_loss.item())

                    hold_val_loss.append(np.mean(predicted_val_storage))

    plt.figure()
    plt.plot(np.arange(num_epochs), hold_loss, c = 'black', label = 'training loss')
    plt.plot(np.arange(num_epochs), hold_val_loss, c = 'orange', label = 'validation loss')
    plt.ylim(0,15)
    plt.legend()
    plt.show()
    plt.close()

    endTime = time.time()
    cprint("[INFO]", "magenta", end=" ")
    print("total time taken to train the model: {:.2f}s".format(
    endTime - startTime))


    # Model evaluation
    with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            if which_set == 'test':
                Y_pred = model(test_features).numpy().reshape(-1)
                df_pred = pd.DataFrame(Y_pred, columns=["Predicted Usable Area [m2]"])
                df_test_viz = df_test.join(df_pred)
            elif which_set == 'val':
                Y_pred = model(val_features).numpy().reshape(-1)
                df_pred = pd.DataFrame(Y_pred, columns=["Predicted Usable Area [m2]"])
                df_test_viz = df_val.join(df_pred)
                Y_test = Y_val.copy()
            else:
                raise ValueError("Incorrect set value!")
        
    df_test_viz["Error"] = np.abs(df_test_viz["Usable area [m2]"] - df_test_viz["Predicted Usable Area [m2]"])

    cprint("\n[INFO]", "magenta", end=" ")
    print(f"Dataset -> {dataset_name}")
    cprint("[INFO]", "magenta", end=" ")
    print(f"Model -> {model_type}\n\n")
    cprint("[INFO]", "magenta", end=" ")
    print("Used data columns: \n-----> "+"\n-----> ".join(used_cols)+"\n\n")
    cprint("[INFO]", "magenta", end=" ")
    print(f"R^2 = {r2_score(Y_test, Y_pred):.2f}")
    cprint("[INFO]", "magenta", end=" ")
    print("Mean Absolute Error = {:.2f}".format(sklearn.metrics.mean_absolute_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Median Absolute Error = {:.2f}".format(sklearn.metrics.median_absolute_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Max Error = {:.2f}".format(sklearn.metrics.max_error(Y_test, Y_pred)))
    cprint("[INFO]", "magenta", end=" ")
    print("Min Error = {:.2f}".format(np.min(np.abs(Y_test-Y_pred))))
    cprint("[INFO]", "magenta", end=" ")
    print("Median Absolute Percentage Error = {:.2f}".format(sklearn.metrics.mean_absolute_percentage_error(Y_test, Y_pred)))

    plot_df_summary(df_test_viz, plot_title=model_type)

    if out:
        return df_test_viz, model, {"train": df_train, "val": df_val, "test": df_test}, scaler



def models_comparison_figure(df_reg, df_nn, plot_name = "Reg_VS_NN.pdf"):
    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    plt.tight_layout(pad=10, h_pad=5, w_pad=0.2)
    fs = 30
    fs_inset = 20
    lw = 3
    colormap = "coolwarm"



    axs[0].set_title("(a)\nLinear regression", fontsize=fs)
    axs[0].plot(np.linspace(60, 200), np.linspace(60, 200))
    # scat1 = axs[0].scatter(df_reg["Usable area [m2]"], df_reg["Predicted Usable Area [m2]"], c=df_reg.Error, cmap="coolwarm")
    norm = plt.Normalize(np.min(df_reg.Error), np.max(df_reg.Error))
    df_1s = df_reg.loc[df_reg["Number of stories"] == 1.0]
    df_2s = df_reg.loc[df_reg["Number of stories"] == 2.0]
    axs[0].scatter(df_1s["Usable area [m2]"], df_1s["Predicted Usable Area [m2]"], c=df_1s.Error, cmap=colormap, marker="D", norm=norm, s=10*fs, edgecolor='b')
    axs[0].scatter(df_2s["Usable area [m2]"], df_2s["Predicted Usable Area [m2]"], c=df_2s.Error, cmap=colormap, marker="o", norm=norm, s=10*fs, edgecolor='b')
    """
    axs[0].text(0.03, 0.96, "R^2 = {:.2f} \nMean Absolute Error = {:.2f}\nMedian Absolute Error = {:.2f}\nMax Error = {:.2f}\nMin Error = {:.2f}\nMedian Absolute Percentage Error = {:.2f}".format(
                                                                                sklearn.metrics.r2_score(df_reg["Usable area [m2]"].to_numpy(), df_reg["Predicted Usable Area [m2]"].to_numpy()), 
                                                                                sklearn.metrics.mean_absolute_error(df_reg["Usable area [m2]"].to_numpy(), df_reg["Predicted Usable Area [m2]"].to_numpy()),
                                                                                sklearn.metrics.median_absolute_error(df_reg["Usable area [m2]"].to_numpy(), df_reg["Predicted Usable Area [m2]"].to_numpy()),
                                                                                sklearn.metrics.max_error(df_reg["Usable area [m2]"].to_numpy(), df_reg["Predicted Usable Area [m2]"].to_numpy()),
                                                                                np.abs(np.min(df_reg["Error"])),
                                                                                sklearn.metrics.mean_absolute_percentage_error(df_reg["Usable area [m2]"].to_numpy(), df_reg["Predicted Usable Area [m2]"].to_numpy())
                                                                                ), bbox=dict(boxstyle="round", fc="lightsalmon", ec="coral", lw=2), horizontalalignment='left', verticalalignment='top', transform=axs[0].transAxes, fontsize=fs_inset);
    """
    divider = make_axes_locatable(axs[0])
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # cbar = plt.colorbar(scat1, cax=cax)
    # cbar.set_label('Error value', fontsize=fs)

    axs[0].set_xlabel(r"Usable area [m$^2$]", fontsize=fs)
    axs[0].set_ylabel(r"Predicted usable area [m$^2$]", fontsize=fs)

    axs[1].set_title("(b)\nNeural network", fontsize=fs)
    axs[1].plot(np.linspace(60, 200), np.linspace(60, 200))
    scat2 = axs[1].scatter(df_nn["Usable area [m2]"], df_nn["Predicted Usable Area [m2]"], c=df_nn.Error, cmap=colormap, marker="")
    norm = plt.Normalize(np.min(df_nn.Error), np.max(df_nn.Error))
    df_1s = df_nn.loc[df_nn["Number of stories"] == 1.0]
    df_2s = df_nn.loc[df_nn["Number of stories"] == 2.0]
    axs[1].scatter(df_1s["Usable area [m2]"], df_1s["Predicted Usable Area [m2]"], c=df_1s.Error, cmap=colormap, marker="D", norm=norm, s=10*fs, edgecolor='b')
    axs[1].scatter(df_2s["Usable area [m2]"], df_2s["Predicted Usable Area [m2]"], c=df_2s.Error, cmap=colormap, marker="o", norm=norm, s=10*fs, edgecolor='b')
    """
    axs[1].text(0.03, 0.96, "R^2 = {:.2f} \nMean Absolute Error = {:.2f}\nMedian Absolute Error = {:.2f}\nMax Error = {:.2f}\nMin Error = {:.2f}\nMedian Absolute Percentage Error = {:.2f}".format(
                                                                                sklearn.metrics.r2_score(df_nn["Usable area [m2]"].to_numpy(), df_nn["Predicted Usable Area [m2]"].to_numpy()), 
                                                                                sklearn.metrics.mean_absolute_error(df_nn["Usable area [m2]"].to_numpy(), df_nn["Predicted Usable Area [m2]"].to_numpy()),
                                                                                sklearn.metrics.median_absolute_error(df_nn["Usable area [m2]"].to_numpy(), df_nn["Predicted Usable Area [m2]"].to_numpy()),
                                                                                sklearn.metrics.max_error(df_nn["Usable area [m2]"].to_numpy(), df_nn["Predicted Usable Area [m2]"].to_numpy()),
                                                                                np.abs(np.min(df_nn["Error"])),
                                                                                sklearn.metrics.mean_absolute_percentage_error(df_nn["Usable area [m2]"].to_numpy(), df_nn["Predicted Usable Area [m2]"].to_numpy())
                                                                                ), bbox=dict(boxstyle="round", fc="lightsalmon", ec="coral", lw=2), horizontalalignment='left', verticalalignment='top', transform=axs[1].transAxes, fontsize=fs_inset);
    """
    axs[1].set_xlabel(r"Usable area [m$^2$]", fontsize=fs)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scat2, cax=cax)
    cbar.set_label('Error value', fontsize=fs)
    cax.tick_params(axis='both', which='major', labelsize=fs, width=2)


    axs[1].set_yticklabels([])

    for ax in axs.flatten():
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_zorder(0)

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize=fs, width=2)

    for ax in axs.flatten():
        ax.grid(True)

    plt.rc('grid', linestyle=":", color='gray')

    plt.savefig(plot_name)
    plt.show()