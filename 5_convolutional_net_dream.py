from PIL import Image
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from load import load_model

theano.config.floatX = 'float32'

srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def init_inputs(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01 + 0.5))


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def model(X, params, p_drop_conv, p_drop_hidden):
    w, w2, w3, w4, w_o = params
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


def main_rain():
    drY = np.identity(10)
    X = init_inputs((10,1,28,28))

    # trX, teX, trY, teY = mnist(onehot=True)
    #
    # trX = trX.reshape(-1, 1, 28, 28)
    # teX = teX.reshape(-1, 1, 28, 28)
    # X = T.ftensor4()
    Y = T.fmatrix()

    params = load_model('media/model/conv-final.model')

    l1, l2, l3, l4, py_x = model(X, params, 0., 0.)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))

    updates = RMSprop(cost, [X], lr=0.001)

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
        name = 'media/dream/conv/{0}.png'.format(str(i))
        im = im.convert('RGB')
        im.save(name)
    im = Image.fromarray(ss * 255)
    name = 'media/dream/conv/all.png'
    im = im.convert('RGB')
    im.save(name)


main_rain()
