import nn_tf
import tensorflow as tf
import nn
import numpy as np


class ai_brain(object):
    """docstring for ai_brain"""
    learn_iterations = 10
    active_func_list = [tf.nn.relu, tf.sigmoid, tf.sin, lambda x:x]
    brain_id = 1

    def __init__(self, input_size, brain_type):
        self.input_size = input_size
        self.brain_type = brain_type
        # print("input_size{0}".format(input_size))
        self.learn_data = []
        self.generate_braint_type()

        # self.brain_conf = {
        #     'shape': [self.input_size, 30, 30, 5],
        #     'active_func': [None, tf.nn.relu, tf.nn.relu, tf.nn.softmax]}

        self.agent_nn = nn_tf.tf_nn(self.brain_conf, 0.1, 0.0001)

    def generate_braint_type(self):
        hidden_layers = np.random.randint(2, 10, size=(np.random.randint(1, 5)))
        shape = [self.input_size] + list(hidden_layers) + [5]
        active_func = (
            [None] +
            list(map(
                lambda x: self.active_func_list[x],
                np.random.randint(0, 4, size=len(shape) - 1))) +
            [tf.nn.softmax])
        self.brain_id += 1
        # self.brain_type = self.brain_id
        self.brain_conf = {'shape': shape, 'active_func': active_func}

    def get_direction(self, input_data):
        # print("input_data {0}".format(input_data))
        if self.brain_type == 0:
            return np.random.random(5).reshape(1, 5)
        else:
            return self.agent_nn.get_answer_set(np.asarray([input_data]))

    def dedublicate(self, input_data, answer_data):
        a = np.append(input_data, answer_data, axis=1)
        b = np.ascontiguousarray(a).view(np.dtype(
            (np.void, a.dtype.itemsize * a.shape[1])))
        _, idx = np.unique(b, return_index=True)
        unique_a = a[idx]
        # print("unique_a {0}".format(unique_a))
        return unique_a[:, : -5], unique_a[:, -5:]

    def learn(self, input_data, answer_data, learning_rate):
        # input_data = input_data / np.sum(input_data).reshape(-1, 1)

        # print("input_data shape{2}\n{0}\nanswer_data shape {3}\n{1}".format(input_data, answer_data, input_data.shape, answer_data.shape))
        self.agent_nn.learn(self.learn_iterations, input_data, answer_data, learning_rate)

