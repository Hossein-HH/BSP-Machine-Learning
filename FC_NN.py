from turtle import forward
import matplotlib.pylab as plt
from numpy import dtype
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import SGD
from torchinfo import summary

class MyNet(nn.Module):
    def __init__(self) -> None:
        super(MyNet, self).__init__()
        self.l1 = nn.Linear(in_features=2, out_features=10, bias=True)
        self.l2 = nn.Linear(in_features=10, out_features=5, bias=True)
        self.l3 = nn.Linear(in_features=5, out_features=1, bias=True)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sig(self.l3(x))
        return x

if __name__=="__main__":
    net = MyNet().cuda()
    summary(net, (2,))
    # exit(0)    
    data, target = make_moons(1000, noise=0.1)
    X,X_t,y,y_t = train_test_split(data,target,test_size=0.2)
    print(X.shape, y.shape, X_t.shape, y_t.shape)
    criterion = nn.BCELoss()
    optim = SGD(net.parameters(), lr=0.01)
    # net.train()
    losses = []
    x = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().view(-1,1)
    epochs = 10000
    for epoch in range(epochs):
        x = x.cuda()
        y = y.cuda()
        y_hat = net(x)
        # print(y_hat)
        # break
        loss = criterion(y_hat, y)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        y_hat[y_hat>=0.5] = 1.0
        y_hat[y_hat<0.5] = 0.0
        acc = torch.sum(y_hat == y)/y.size(0)
        if epoch % (epochs//10) == 0:
            print(f'Epoch [{epoch}/{epochs}] Loss: {loss.mean().item():.4f}, Train Acc: {acc.item()*100}')
    plt.plot(list(range(epochs)), losses)
    plt.figure()
    import numpy as np
    with torch.no_grad():
        # net.eval()
        x1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
        x2 = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100)
        xx, yy = np.meshgrid(x1,x2)
        xy = np.c_[xx.ravel(), yy.ravel()]
        xy = torch.from_numpy(xy).float().cuda()
        xy_hat = net(xy)
        y_hat = xy_hat.cpu()
        y_hat[y_hat>=0.5] = 1.0
        y_hat[y_hat<0.5] = 0.0
        plt.scatter(xy[:,0].cpu(), xy[:,1].cpu(),c=y_hat,cmap=plt.cm.Paired)
        plt.scatter(X[:,0],X[:,1],c=y.cpu())
        plt.show()

