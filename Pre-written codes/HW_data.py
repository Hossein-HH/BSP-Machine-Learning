import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
points = np.linspace(-np.pi, np.pi, 1000)
y = np.sin(points)
y_noise = np.sin(points)+0.3*np.random.randn(1000)

data = np.c_[points.ravel(),y_noise.ravel()]
# print(data.shape)
# np.savetxt('E:\\Courses\\Fundamental of IC 1400_2\\code\\HW\\HW_data.csv', data, delimiter=',')

# plt.scatter(points, y_noise,s=1)
# plt.scatter(points, y,s=0.5)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(points, y_noise, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print(X_train.shape, X_test.shape)
mlp = MLPRegressor(hidden_layer_sizes=(64,64), solver="adam", max_iter=1000)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f'MeanSquareErro => {np.mean((y_pred - y_test)**2)}')
plt.plot(points, y, 'g')
plt.scatter(X_train[:,0], y_train[:,0],s=1)
plt.scatter(X_test[:,0], y_pred,s=0.5)
plt.show()

