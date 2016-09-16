import cProfile
from multiprocessing.dummy import Pool

import random as rnd
import operator
import numpy as np
import matplotlib.pyplot as plt
from game_viz import *
from agent import *


WIDTH = 80
HEIGH = 50
GAME_SIZE = (WIDTH, HEIGH)


class ai_game(object):
    """docstring for ai_game"""
    food_rate = 300

    def __init__(self, in_size, agent_types, game_view=True, scale=3):
        self.game_pool = Pool()
        self.hp_hist = []
        self.game_view = game_view
        self.game_width = in_size[0]
        self.game_heigh = in_size[1]
        self.game_size = in_size
        self.game_map = [
            [[] for x in range(0, self.game_heigh)]
            for y in range(self.game_width)]
        if game_view:
            self.game_viz = game_visualization(
                self.game_width,
                self.game_heigh,
                scale)
        self.agents_list = []
        for i, brain_type in enumerate(agent_types):
            t_pos = self.get_rnd_free_pos()
            self.agents_list.append(ai_agent(t_pos, i, brain_type))
            self.set_map_cell(t_pos, i)
        self.main_loop()

    def get_map_cell(self, pos, check_map=None):
        check_map = self.game_map if check_map is None else check_map
        # print(pos)
        return check_map[pos[0]][pos[1]]

    def set_map_cell(self, pos, value):
        self.game_map[pos[0]][pos[1]].append(value)

    def get_rnd_free_pos(self):
        t_pos = (
            rnd.randint(0, self.game_width - 1),
            rnd.randint(0, self.game_heigh - 1))
        i = 0
        while len(self.get_map_cell(t_pos)):
            if i >= 1000:
                raise "Couldn't get free place"
                return (0, 0)
            i += 1
            t_pos = (
                rnd.randint(0, self.game_width - 1),
                rnd.randint(0, self.game_heigh - 1))
        return t_pos

    def append_rules_old(self):
        for row in self.game_map:
            for cell in row:
                if len(cell) >= 2:
                    self.append_cell_rules_2(cell)

    def append_rules(self):
        for agent in self.agents_list:
            cell = self.get_map_cell(agent.position)
            if len(cell) >= 2:
                self.append_cell_rules_2(cell)

    def append_cell_rules_2(self, cell):
        if len(cell) == 2 and -1 in cell:
            cell.remove(-1)
            self.agents_list[cell[0]].health.append(
                self.agents_list[cell[0]].health[-1] + 1)
        elif -1 in cell:
            cell.remove(-1)
        return cell

    def append_cell_rules_1(self, cell):
        if len(cell) == 2 and -1 in cell:
            cell.remove(-1)
            self.agents_list[cell[0]].health.append(
                self.agents_list[cell[0]].health[-1] + 1)
        else:
            id = cell.pop()
            if id > 0:
                self.agents_list[id].health.append(0)
        return cell

    def append_cell_rules_3(self, cell):
        if -1 in cell:
            cell.remove(-1)
            if len(cell) == 1:
                self.agents_list[cell[0]].health.append(
                    self.agents_list[cell[0]].health[-1] + 1)
            else:
                 max_hp = max(self.agents_list[i].health[-1] for i in cell)



    def create_food(self):
        self.set_map_cell(self.get_rnd_free_pos(), -1)

    def agent_iteration(self, agent):
        return agent.move(self.game_map)

    def correct_position(self, value, max_value):
        if value < 0:
            return max_value + value
        elif value >= max_value:
            return value - max_value
        else:
            return value

    def move_agents(self, directions):
        # print(directions)
        for i, agent in enumerate(
            [agent for agent in self.agents_list if agent.health[-1] > 0]):
            # удаление агента с карты
            self.game_map[agent.position[0]][agent.position[1]].remove(agent.agent_id)
            # изменение позиции агента
            agent.position = tuple(map(
                operator.add, agent.position, directions[i]))
            # print("new_position {0}".format(agent.position))
            # корректировка позиции агента в случае выхода за карту
            map_size = (len(self.game_map), len(self.game_map[0]))
            agent.position = tuple(map(
                self.correct_position, agent.position, map_size))
            # print("correct_position {0}".format(agent.position))
            # добавление агента на карту
            self.game_map[agent.position[0]][agent.position[1]].append(agent.agent_id)

    def game_iteration(self, iter_count):
        directions = list(self.game_pool.map(
            self.agent_iteration,
            [agent for agent in self.agents_list if agent.health[-1] > 0]))
        # self.game_pool.join()
        self.move_agents(directions)

        self.append_rules()
        if iter_count % self.food_rate == 0:
            for i in range(5):
                self.create_food()

    def game_iteration_old(self, iter_count):
        for agent in self.agents_list:
            if agent.health[-1] > 0:
                agent.move(self.game_map)
        self.append_rules()
        if iter_count % self.food_rate == 0:
            for i in range(5):
                self.create_food()

    def save_hp_hist(self):
        self.hp_hist += [[agent.health[-1] for agent in self.agents_list]]


    def main_loop(self):
        iter_count = 0
        while iter_count < 3000:
            if iter_count % 100 == 0:
                self.save_hp_hist()
            if iter_count % 1000 == 0:
                print(iter_count)
            if self.game_view:
                if not self.game_viz.check_event(pygame.event.get()):
                    break
                self.game_viz.drow_map(self.game_map)
            self.game_iteration(iter_count)
            iter_count += 1
        # for agent in self.agents_list:
        #     plt.plot(range(len(agent.health)), agent.health)
        #     print("agent brain = {1}\nsum hp = {0}".format(agent.health, agent.brain.brain_type))
        legend = []
        for agent in self.agents_list:
            print('agent id:{0}\nagent brain shape:\n{1}\nbrain_cative:\n{2}'.format(
                agent.brain.brain_type,
                agent.brain.brain_conf['shape'],
                agent.brain.brain_conf['active_func']))
        for id, agent_hp in enumerate(np.array(self.hp_hist).T):
            legend.append('brain type = {0}'.format(self.agents_list[id].brain.brain_type))
            print("agent_hp = {0}".format(agent_hp))
            plt.plot(range(len(agent_hp)), agent_hp)
        plt.legend(legend, loc='upper left')
        plt.show()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['game_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


def main(z):
    my_game = ai_game(GAME_SIZE, range(6), scale=4, game_view=True)

if __name__ == '__main__':
    main(1)
    # list(range(2, 17, 2))
    # t = ai_brain(5, 1)
    # t.generate_braint_type()
    # rnd.seed(52)
    # cProfile.run('main(1)')
    # pool = Pool()
    # pool.map(main, [1, 2])
