import numpy as np
import random as rand
from matplotlib import pyplot as plt



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class neural_network(object):
    """docstring for neural_network
        nn_arch - структура нейронной сети вида [n0, n1, n2]
            n - кол-во нейронов на слое
            n0 - кол-во характеристик (размер входного слоя)
    """

    def __init__(
        self, nn_arch, quality_func,
        quality_grad_func, activ_func,
        d_activ_func, learning_rate, stop_learning_delta):
        # if activ_func is None:
        #     activ_func = self.sigmoid
        #     d_activ_func = self.dsigmoid
        self.delta = stop_learning_delta
        self.alpha = learning_rate
        self.quality_func = quality_func
        self.quality_grad_func = quality_grad_func
        self.bias = np.asarray(
            [[rand.random() for x in range(n)] for n in nn_arch[1:]])
        self.activ_func_matrix = np.asarray(
            [[activ_func for x in range(n)] for n in nn_arch[1:]])
        self.d_activ_func_matrix = np.asarray(
            [[d_activ_func for x in range(n)] for n in nn_arch[1:]])
        self.weight = list(map(
            lambda x, y:
            np.asarray([[rand.random() for i in range(x)] for j in range(y)]),
            nn_arch[:-1],
            nn_arch[1:]))
        self.layer_cnt = len(nn_arch) - 1

    def print_weigths(self):
        print("w_matrix = {0}\nbias_matrix = {1}".format(
            self.weight, self.bias))

    def summ_func(self, layer, in_input):
        # print("in_input shape = {0}\nweight shape = {1}".format(
        #     in_input.shape,self.weight[layer].shape))
        # print(self.weight)
        # print("{0} *\n{1}\n- {2}".format(
        #     in_input, self.weight[layer].T, self.bias[layer]))
        # print(in_input.dot(self.weight[layer].T))
        return in_input.dot(self.weight[layer].T) + self.bias[layer]

    def set_activ_func(self, layer, neuron_id, activ_func, d_activ_func):
        self.activ_func_matrix[layer, neuron_id] = activ_func
        self.d_activ_func_matrix[layer, neuron_id] = d_activ_func

    def get_inside_error(self, prev_error, layer, in_sum):
        # print("calc error on level {2}:\n{0} *\n{1} *\n{3}".format(
        #     prev_error,
        #     self.weight[layer + 1],
        #     layer,
        #     self.calc_func_array(
        #         in_sum, layer, self.d_activ_func_matrix)))
        z = prev_error.dot(self.weight[layer + 1]) *\
            self.calc_func_array(in_sum, layer, self.d_activ_func_matrix)
        # print("test {0}".format(z))
        return z

    def calc_activation(self, in_sum, layer):
        return(np.asarray(list(map(
            lambda f, z: f(z), self.activ_func_matrix[layer], in_sum.T))).T)

    def calc_func_array(self, in_sum, layer, func_matrix):
        # print("calc_func_array\n{0}".format((in_sum, layer, func_matrix)))
        return(np.asarray(list(map(
            lambda f, z: f(z), func_matrix[layer], in_sum.T))).T)

    def learn(self, iterations, data_set, answer_set, learning_rate=None):
        self.alpha= self.alpha if learning_rate is None else learning_rate
        # print("data = {0}".format(data_set))
        # print('answer_set = {0}'.format(answer_set))
        quality_list = []
        for i in range(iterations):
            quality_list.append(self.learn_one_step(data_set, answer_set))
            if len(quality_list) >1 and np.mean(quality_list[-2] - quality_list[-1]) < self.delta:
                break
        # self.create_plot(np.asarray(quality_list).T)
        return quality_list

    def forward_prop(self, data_set):
        z = []
        activation = [data_set]
        for layer in range(self.layer_cnt):
            z.append(self.summ_func(layer, activation[layer]))
            activation.append(self.calc_activation(z[layer], layer))
        return(z, activation)

    def get_answer_set(self, input_data):
        (z, output) = self.forward_prop(input_data)
        return(output[-1])

    def learn_one_step(self, data_set, answer_set):
        (z, activation) = self.forward_prop(data_set)
        # print("sum = {0}".format(z))
        # print("activation matrix = {0}".format(activation))
        errors = [
            self.quality_grad_func(answer_set, activation[-1]) *
            self.calc_func_array(z[-1], -1, self.d_activ_func_matrix)]
        # print("last error = {0}".format(errors[0]))
        for layer in reversed(range(self.layer_cnt - 1)):
            # print("corrent layer = {0}".format(layer))
            errors.append(self.get_inside_error(
                errors[self.layer_cnt - layer - 2], layer, z[layer]))
        # print('errors = {0}'.format(errors))
        self.upd_weigth(activation, errors)
        quality = self.quality_func(answer_set, activation[-1])
        # print("quality func = {0}".format(quality))
        return quality

    def upd_weigth(self, activation_matrix, error_matrix):
        for layer in range(self.layer_cnt):
            # print("activation_matrix shape = {0}".format(activation_matrix[layer].T))
            # print("error matrix layer {0} = {1}".format(layer, error_matrix[self.layer_cnt - layer - 1]))
            # print("layer {0}\nmean error = {1}\n activation".format(layer, error_matrix[self.layer_cnt - layer - 1].mean(0)))
            self.bias[layer] -= self.alpha * error_matrix[self.layer_cnt - layer - 1].sum(0)
            # print("test {0}\n{1}".format(
            #     self.weight[layer],
            #     self.alpha * (
            #     activation_matrix[layer].T.dot(
            #         error_matrix[self.layer_cnt - layer - 1]))))
            self.weight[layer] -= self.alpha * (
                activation_matrix[layer].T.dot(
                    error_matrix[self.layer_cnt - layer - 1])).T

    def create_plot(self, in_data):
        data = in_data.sum(axis=0)
        plot = plt.plot(list(range(len(data))), data)
        plt.show(plot)


def quality(y, y_pred):
    z = y - y_pred
    return (z * z).sum(axis=0)


def quality_grad(y, y_pred):
    return (y_pred - y)  # .mean(axis=0)


if __name__ == '__main__':
    plt.matshow(np.random.rand(64, 64), fignum=100, cmap=plt.cm.gray)
    plt.show()
    exit()
    rand.seed(50)
    tst_nn = neural_network(
        [2, 2], quality, quality_grad, sigmoid, dsigmoid, 0.1, 0.000001)
    tst_nn.print_weigths()
    # tst_nn.print_weigths()
    data = np.asarray(
        [[1, 1, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0]])
    print("answer = {0}".format(tst_nn.get_answer_set(
        np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]]))))
    quality_list = []
    tst_nn.learn(10000, data[:, :-3], data[:, 2:4])
    # for i in range(100):
    #     quality_list += (tst_nn.learn(1, data[0:1, :-3], data[0:1, 3:4]))
    #     quality_list += (tst_nn.learn(1, data[1:2, :-3], data[1:2, 3:4]))
    #     quality_list += (tst_nn.learn(1, data[2:3, :-3], data[2:3, 3:4]))
    #     quality_list += (tst_nn.learn(1, data[3:4, :-3], data[3:4, 3:4]))
    # print(quality_list)
    # plot = plt.plot(list(range(len(quality_list))), quality_list)
    # plt.show(plot)
    print("answer = {0}".format(tst_nn.get_answer_set(
        np.asarray([[1, 1], [1, 0], [0, 1], [0, 0]]))))
    tst_nn.print_weigths()
