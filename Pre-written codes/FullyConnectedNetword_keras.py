import matplotlib.pylab as plt
# from numpy import dtype
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


if __name__ == '__main__':
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

    input_layer = keras.layers.Input(shape=(X_train.shape[1]))
    hidden1 = keras.layers.Dense(units=8, activation='relu')(input_layer)
    hidden2 = keras.layers.Dense(units=8, activation='relu')(hidden1)
    # hidden3 = keras.layers.Dense(units=128, activation='relu')(hidden2)
    output  = keras.layers.Dense(1)(hidden2)

    model = keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='sgd', 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=["mean_absolute_error"])
    print(model.summary())
    # exit(0)
    history = model.fit(X_train, y_train, epochs=100, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=1)

    print(score)
    print("Test MSE Score:", score[0])
    print("Test MAE Score:", score[1])

    x_plot = np.linspace(X_test.min(), X_test.max(), 1000)
    y_plot = model.predict(x_plot)
    plt.scatter(X_train[:,0], y_train[:,0], s=2.5)
    plt.plot(points, y, color='green')
    plt.plot(x_plot, y_plot, color='red')
    plt.show()
