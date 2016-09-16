from brain import *


class ai_agent(object):
    learn_batch = 20
    dir_list = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    """docstring for ai_agent"""

    def __init__(self, in_pos, in_id, brain_type, view_range=5):
        self.view_range = view_range
        self.agent_id = in_id
        self.health = [1]
        self.position = in_pos
        self.brain = ai_brain(pow(view_range * 2 + 1, 2) + 5, brain_type)
        self.learning_data = []
        self.predicted_directions = []
        self.prepare_data = self.prepare_data_2  # self.prepare_data_1 if brain_type % 2 == 0 else
        self.learn = self.learn_v2  #self.learn_v1 if brain_type % 3 == 0 else 

    def convert_map_cell(self, cell):
        # if len(cell) == 0:
        #     return -2
        # else:
        return -1 if -1 in cell else len(cell)

    def get_position_area(self, in_map):
        nd_map = np.asarray(in_map)
        (x_max, y_max) = nd_map.shape
        # print("max{0}".format((x_max, y_max)))
        (x, y) = self.position
        # print("current{0}".format((x, y)))
        r = self.view_range
        x1 = x - r
        x2 = x + 1 + r
        y1 = y - r
        y2 = y + 1 + r
        x1_main = 0 if x1 < 0 else x1
        x2_main = x_max if x2 > x_max else x2
        y1_main = 0 if y1 < 0 else y1
        y2_main = y_max if y2 > y_max else y2
        center_matrix = nd_map[x1_main:x2_main, y1_main:y2_main]
        # print("center_matrix {0}".format(center_matrix))
        if x1 < 0:  # left part
            # print("left part")
            center_matrix = np.append(
                nd_map[x1:x_max, y1_main:y2_main], center_matrix, axis=0)
        elif x2 > x_max:  # right part`
            # print("right part\n")
            center_matrix = np.append(
                center_matrix, nd_map[0:x2 - x_max, y1_main:y2_main], axis=0)
        if y1 < 0:  # top part
            top_part = nd_map[x1_main:x2_main, y1:y_max]
            if x1 < 0:
                top_part = np.append(
                    nd_map[x1:x_max, y1:y_max],
                    top_part, axis=0)
            if x2 > x_max:
                top_part = np.append(
                    top_part,
                    nd_map[0:x2 - x_max, y1:y_max], axis=0)
            # print("top part\ncenter{0}\ntop{1}\ncoorinate{2}".format(center_matrix, top_part,(y1,y_max)))
            center_matrix = np.append(top_part, center_matrix, axis=1)
        elif y2 > y_max:  # bottom part
            # print("bottom part")
            bottom_part = nd_map[x1_main:x2_main, 0:y2 - y_max]
            if x1 < 0:
                bottom_part = np.append(
                    nd_map[x1:x_max, 0:y2 - y_max],
                    bottom_part, axis=0)
            if x2 > x_max:
                bottom_part = np.append(
                    bottom_part,
                    nd_map[0:x2 - x_max, 0:y2 - y_max], axis=0)
            center_matrix = np.append(
                center_matrix, bottom_part, axis=1)
        # print(map(self.convert_map_cell, center_matrix.flatten()))
        return list(map(self.convert_map_cell, center_matrix.flatten()))

    def prepare_data_1(self):
        if len(self.health) > 1 and self.health[-1] - self.health[-2] > 0:
            answer_data = [row >= np.amax(row) for row in self.predicted_directions]
            self.health.append(self.health[-1])
        else:
            answer_data = [(row < np.amax(row)) for row in self.predicted_directions]
        return answer_data

    def prepare_data_2(self):
        if len(self.health) > 1 and self.health[-1] - self.health[-2] > 0:
            answer_data = [row >= np.amax(row) for row in self.predicted_directions]
            self.health.append(self.health[-1])
        else:
            answer_data = [(row < np.amax(row)) if np.random.random() > 0.5 else row >= np.amax(row) for row in self.predicted_directions]
        # print("size of data = {0}".format(len(answer_data)))
        return answer_data

    def learn_v1(self):
        if len(self.learning_data) > self.learn_batch:
            self.learning_data, self.predicted_directions = \
                self.brain.dedublicate(
                    self.learning_data, self.predicted_directions)
            # print("predicted_directions\n{0}".format(self.predicted_directions))
            answer_data = self.prepare_data()
            # print("answer_data{0}".format(np.asarray(answer_data)))
            self.brain.learn(np.asarray(self.learning_data), np.asarray(answer_data), 0.01)#2/(1+np.exp(0.5*self.health[-1]-5)) + 0.1)#1/(self.health[-1]/200-0.0001))
            # self.health.append(max([self.health[-1] - 1, 1]))
            self.learning_data = []
            self.predicted_directions = []
            # print("health = {0}".format(self.health[-1]))

    def learn_v2(self):
        if len(self.learning_data) > self.learn_batch or len(self.health) > 1 and self.health[-1] != self.health[-2]:
            self.learning_data, self.predicted_directions = \
                self.brain.dedublicate(
                    self.learning_data, self.predicted_directions)
            # print("predicted_directions\n{0}".format(self.predicted_directions))
            answer_data = self.prepare_data()
            # print("answer_data{0}".format(np.asarray(answer_data)))
            self.brain.learn(np.asarray(self.learning_data), np.asarray(answer_data), 0.01)#2/(1+np.exp(0.5*self.health[-1]-5)) + 0.1)#1/(self.health[-1]/200-0.0001))
            # self.health.append(max([self.health[-1] - 1, 1]))
            self.learning_data = []
            self.predicted_directions = []
            # print("health = {0}".format(self.health[-1]))

    def move(self, in_map):
        # обучение агента
        self.learn()
        position_vector = self.get_position_area(in_map)
        # position_vector.append(self.health[-1])
        # насыщение входного вектора информацией о предыдущем направлении
        if len(self.predicted_directions) == 0:
            position_vector += [0, 0, 0, 0, 0]
        else:
            position_vector += list(
                self.predicted_directions[-1] >=
                np.amax(self.predicted_directions[-1]))
            # print("prev_y = {0}".format(predicted_directions[-1]))
        self.learning_data.append(position_vector)
        # print("agent {0} position = {1}".format(self.agent_id, position_vector))
        dir_prob = self.brain.get_direction(position_vector)
        dir_id = np.argmax(dir_prob)
        # print("dir_prob = {0}\ndir_id = {1}".format(dir_prob, dir_id))
        self.predicted_directions.append(dir_prob[0] >= np.amax(dir_prob[0]))
        # print("old_position {0}".format(self.position))
        return self.dir_list[dir_id]