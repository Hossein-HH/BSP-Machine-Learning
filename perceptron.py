from sklearn.datasets import make_classification, make_blobs
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def step_func(x):
    tmp = 1*(x>=0)
    return tmp

if __name__=='__main__':
    # data, target = make_blobs(200, centers=2, n_features=2, center_box=(0, 10))
    data, target = make_classification(200,n_features=2,n_classes=2,n_informative=2,n_redundant=0)
    print(data.shape, target.shape)
    y = target[:]
    target = target.reshape(-1,1)
    # plt.scatter(data[:,0], data[:,1], c=target)
    # plt.show()
    # x = np.c_[data,np.ones(data.shape[0],1)]
    # step_func = lambda x: 0 if x<0 else 1
    epochs = 100
    lr = 0.1
    w = np.random.randn(2,1)
    w0 = np.random.randn()
    best_acc = 0.0
    best_w = w[:]
    best_w0 = w0
    best_epoch = 0
    for epoch in range(epochs):
        losses = []
        for i, sample in enumerate(data):
            sample = sample.reshape(-1,1)
            score = w.T@sample+w0
            y_hat = step_func(score[0][0])
            error = target[i,0] - y_hat
            losses.append(error**2)
            w0 += lr*error
            w += lr*error*sample
        # score = np.dot(data,w)+w0
        # y_hat = step_func(score)
        # error = y_hat - target
        # losses.append(np.mean(error**2))
        # w0 += lr*np.sum(error)
        # w += lr*np.dot(data.T,error)
        if epoch %10 == 0:
            loss = np.mean(losses)
            scores = np.dot(data,w)+w0
            y_hat = step_func(scores)
            acc = np.mean(y_hat==target)
            # print(acc.shape)
            print(f'Epoch [{epoch}/{epochs}]: loss={loss:0.6f}, acc={acc*100:0.2f}')
            if acc > best_acc:
                best_w = w[:]
                besr_w0 = w0
                best_acc = acc
                best_epoch = epoch
    # plt.scatter(data[:,0], data[:,1], c=target)
    
    # slop = -w[0]/w[1]
    # bias = -w0/w[1]
    x1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1,1000)
    y1 = np.linspace(data[:,1].min()-1, data[:,1].max()+1,1000)
    xx, yy = np.meshgrid(x1,y1)
    xy = np.c_[xx.ravel(), yy.ravel()]
    sc = np.dot(xy,w)+w0
    xy_y_hat = step_func(sc.reshape(xy.shape[0],))
    plt.plot(xy[:,0][xy_y_hat==0], xy[:,1][xy_y_hat==0], 'g', alpha=0.2)
    plt.plot(xy[:,0][xy_y_hat==1], xy[:,1][xy_y_hat==1], 'b', alpha=0.2)
    
    plt.plot(data[:, 0][y==0], data[:, 1][y==0], 'g^')
    plt.plot(data[:, 0][y == 1], data[:, 1][y == 1], 'bs')
    
    print(best_w)
    print(best_w0) 
    print(best_epoch) 
    print(w)
    print(w0)  
    plt.show()