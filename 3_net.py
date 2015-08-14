import theano
from theano import tensor as T
import numpy as np

from load import mnist, save_model


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def model(X, params):
    w_h, w_o = params
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx


def main_train():
    trX, teX, trY, teY = mnist(onehot=True)

    X = T.fmatrix()
    Y = T.fmatrix()

    w_h = init_weights((784, 625))
    w_o = init_weights((625, 10))

    params = [w_h, w_o]
    py_x = model(X, params)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    updates = sgd(cost, params)

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], trY[start:end])
        print np.mean(np.argmax(teY, axis=1) == predict(teX))
        if i % 10 == 0:
            name = 'media/model/net-{0}.model'.format(str(i))
            save_model(name, params)
    name = 'media/model/net-final.model'
    save_model(name, params)


main_train()
