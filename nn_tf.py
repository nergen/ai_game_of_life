import tensorflow as tf
import logging
import numpy as np

class tf_nn(object):
    """agent decision system by neural network"""

    def __init__(self, nn_arch, learning_rate=0.5, stop_learning_delta=0.001):
        self.learning_rate = learning_rate
        self.weigths = []
        self.bias = []
        self.activation_func = []
        for i in range(len(nn_arch['shape']) - 1):
            self.weigths.append(self.generate_weigth((nn_arch['shape'][i:i + 2])))
            self.bias.append(self.generate_bias(((1, nn_arch['shape'][i + 1]))))
            self.activation_func.append(nn_arch['active_func'][i + 1])

        self.init_forward_tensor()
        self.init_learn_tensor()

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def init_learn_tensor(self):
        self.out = tf.placeholder(tf.float32, name='output')
        loss = tf.reduce_mean(tf.square(self.forward_pass - self.out))
        self.learning = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def init_forward_tensor(self):
        self.in_data = tf.placeholder(tf.float32, name='input')
        layer_activation = [self.in_data]
        for weigth, bias, activ_func in zip(self.weigths, self.bias, self.activation_func):
            layer_activation.append(
                activ_func(tf.matmul(layer_activation[-1], weigth) + bias))
        self.forward_pass = layer_activation[-1]

    def generate_weigth(self, shape):
        logging.debug('weigth shape: {0}'.format(shape))
        init_vals = tf.random_normal(shape, stddev=0.5)
        return tf.Variable(init_vals)

    def generate_bias(self, shape):
        init_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_vals)

    def run_forward_pass(self, in_data_set):
        return self.sess.run(self.forward_pass, feed_dict={self.in_data: in_data_set})

    def run_learning(self, in_data_set, out_data, iter_count):
        for i in range(iter_count):
            self.sess.run(
                self.learning,
                feed_dict={self.in_data: in_data_set, self.out: out_data})

    def get_weigths(self):
        return list(map(self.sess.run, self.weigths))

    def get_answer_set(self, input_data):
        return self.run_forward_pass(input_data)

    def learn(self, iterations, data_set, answer_set, learning_rate=None):
        self.run_learning(data_set, answer_set, iterations)


def main():
    W = tf.Variable(tf.random_normal([2, 5], stddev=0.5))
    b = tf.Variable(tf.zeros(5))


    X = tf.constant(np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]))
    y = tf.constant(np.float32(np.array(
        [[0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]
        ]).T))

    predicted_val = tf.sigmoid((tf.matmul(X, W) + b))
    loss = tf.reduce_mean(tf.square(predicted_val - y))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    logging.debug('initialyze model var W:\n{0}'.format(sess.run(W)))
    logging.debug('initialyze model var b:\n{0}'.format(sess.run(b)))
    logging.debug('init predict:\n{0}'.format(sess.run(predicted_val)))
    logging.debug('correct answers:\n{0}'.format(sess.run(y)))
    for i in range(10000):
        sess.run(train)

    logging.debug('final answers:\n{0}'.format(sess.run(predicted_val)))

    logging.debug('new model var W:\n{0}'.format(sess.run(W)))
    logging.debug('new model var b:\n{0}'.format(sess.run(b)))
    logging.debug('new loss:\n{0}'.format(sess.run(predicted_val - y)))
    logging.debug('new loss:\n{0}'.format(sess.run(loss)))

if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.debug(tf.__version__)
    x = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.float32(np.array([
        [0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0]
    ]).T)
    # main()
    # a = [3,5,6]
    # print(tuple(a[1:3]))
    test_nn = tf_nn(
        {'shape': [2, 2, 5], 'active_func': [None, tf.nn.relu, tf.sigmoid]},
        None, None, None, None, 0.5, 0.1)
    # print(test_nn.get_weigthsigmoids())
    print(test_nn.run_forward_pass(x))
    test_nn.run_learning(x, y, 100000)
    print(test_nn.run_forward_pass(x))
    print(y - test_nn.run_forward_pass(x))
    # f, k = main2()
    # with tf.Session() as sess:
        # sess.run(init)
        # print(sess.run(k, feed_dict={f:10}))

