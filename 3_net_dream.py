from PIL import Image
import theano
from theano import tensor as T
import numpy as np

from load import load_model


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def init_inputs(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01 + 0.5))


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
    drY = np.identity(10)
    X = init_inputs((10, 784))
    Y = T.fmatrix()

    params = load_model('media/model/net-final.model')
    py_x = model(X, params)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    updates = sgd(cost, [X])

    dream = theano.function(inputs=[Y], outputs=cost, updates=updates, allow_input_downcast=True)

    ss = None
    costs = []
    for i in range(100):
        for j in range(100):
            cost = dream(drY)
            costs.append(cost)
        print i, cost
        xv = X.get_value()
        s = None
        for xvi in xv:
            xvi.resize(28, 28)
            if s is None:
                s = np.array(xvi, copy=True)
            else:
                s = np.concatenate((s, xvi))

        if ss is None:
            ss = np.array(s, copy=True)
        else:
            ss = np.concatenate((ss, s), axis=1)
        im = Image.fromarray(s * 255)
        name = 'media/dream/net/{0}.png'.format(str(i))
        im = im.convert('RGB')
        im.save(name)
    im = Image.fromarray(ss * 255)
    name = 'media/dream/net/all.png'
    im = im.convert('RGB')
    im.save(name)


main_train()
