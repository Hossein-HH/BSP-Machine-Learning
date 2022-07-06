from turtle import forward
import matplotlib.pylab as plt
# from numpy import dtype
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torchinfo import summary

class FullyConnected(nn.Module):
    def __init__(self, in_dim=1, hiddens = [8], out_dim=1) -> None:
        super(FullyConnected, self).__init__()
        self.layers = None
        if len(hiddens)>0:
            self.layers = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hiddens[0]),
                nn.ReLU()
            )
            for i in range(1,len(hiddens)):
                self.layers.add_module(f'H{i}',nn.Linear(hiddens[i-1], hiddens[i]))
                self.add_module('relu',nn.ReLU())
            self.layers.add_module('Out',nn.Linear(hiddens[-1], out_dim))
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                #nn.ReLU()
            )
        # self.l1 = nn.Linear(in_dim, hiddens[0])
        # self.l2 = nn.Linear(hiddens[0], hiddens[1])
        # #....
        # self.ln = nn.Linear(hiddens[-1], out_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # z_l1 = self.l1(x)
        # z_l1 = self.relu(z_l1)
        # z_l2 = self.l2(z_l1)
        # z_l2 = self.relu(z_l2)
        # #....
        # y_hat = self.ln(z_ln_1)
        # y_hat = self.relu(y_hat)
        y_hat = self.layers(x)
        return y_hat

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    # torch.cuda.is_available = lambda : True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import time
    points = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(points)
    y_noise = np.sin(points)+0.3*np.random.randn(1000)

    data = np.c_[points.ravel(),y_noise.ravel()]
    X_train, X_test, y_train, y_test = train_test_split(points, y_noise, test_size=0.2, random_state=1)

    X_train = X_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    epochs = 10000
    lr = 0.001
    batch_size = 100
    myNetword = FullyConnected(1,[20],1).to(device)
    summary(myNetword, (batch_size,1,1))
    loss = nn.MSELoss().to(device)
    optim = SGD(myNetword.parameters(),lr=lr)
    x = torch.from_numpy(X_train).float().to(device)
    y = torch.from_numpy(y_train).float().to(device)
    losses = []
    best_loss = 10e20
    print(device)
    start = time.time()
    for epoch in range(1, epochs+1):
        # print(f'\r Epoch [{epoch}/{epochs}] ', end='')
        y_hat = myNetword(x)
        error = loss(y_hat, y)
        losses.append(error.item())
        optim.zero_grad()
        error.backward()
        optim.step()
        #if error<best_loss:
        #    torch.save(myNetword.state_dict(),'best_model.pt')
        #    best_loss = error
        #    # print('\t Model saved.')
        if epoch % 100 == 0:
           print(f'\rEpoch [{epoch}/{epochs}]: current loss->{error.item():0.5f}, '
                 f'avg loss->{np.mean(losses):.5f}', end='')
    print(f'\nTotal time=> {time.time()-start} ')
    plt.plot(list(range(len(losses))), losses)
    plt.show()
    x_plot = np.linspace(points.min(), points.max(), 1000)
    y_plot = myNetword(torch.from_numpy(x_plot.reshape(-1,1)).float().to(device))
    plt.figure()
    plt.scatter(x.numpy(),y.numpy(),s=2.5)
    plt.plot(x_plot,y_plot.detach().numpy())
    plt.show()