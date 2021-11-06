import copy
import math
import numpy as np

from common import MCTS_Node, MCTS_State
from config_vertiport import Config

#derived class from MCTS_State
class MultiAircraftState(MCTS_State):
    #initialising
    def __init__(self,
                 state,
                 index,
                 init_action,
                 hit_wall=False,
                 conflict=False,
                 reach_goal=False,
                 prev_action=None,
                 depth=0):
        #initialising members of parent class
        MCTS_State.__init__(self, state)
        #initialising other members
        self.index = index
        self.init_action = init_action
        self.hit_wall = hit_wall
        self.conflict = conflict
        self.reach_goal = reach_goal
        self.prev_action = prev_action
        self.depth = depth

        self.G = Config.G
        self.scale = Config.scale

        self.nearest_x = -1
        self.nearest_y = -1

    # reward function
    def reward(self):
        if self.hit_wall or self.conflict:
            r = 0
        elif self.reach_goal:
            r = 1
        else:
            r = 1 - self.dist_goal() / 1600.0
            r /= 4
        return r

    # check if terminal state
    def is_terminal_state(self, search_depth):
        if self.reach_goal or self.hit_wall or self.conflict or self.depth == search_depth:
            return True
        return False

    def move(self, a):
        if self.depth < 1:
            next_state = self._move(a)
        else:
            rand_action = np.random.randint(0, 3, size=self.state.shape[0])
            next_state = self._move(rand_action)
        return next_state

    def _move(self, a):
        # state: dimension: n by 8
        # [aircraft: x, y, vx, vy, v, heading, gx, gy]
        state = copy.deepcopy(self.state)
        hit_wall = False
        conflict = False
        reach_goal = False

        for _ in range(Config.simulate_frame):
            for index in range(state.shape[0]):
                heading = state[index, 5] + (a[index] - 1) * Config.d_heading  # degree
                speed = state[index, 4] + np.random.normal(0, Config.speed_sigma)
                speed = max(Config.min_speed, min(speed, Config.max_speed))  # project to range
                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)
                state[index, 0] += vx
                state[index, 1] += vy

                state[index, 2] = vx
                state[index, 3] = vy
                state[index, 4] = speed
                state[index, 5] = heading

            ownx = state[self.index][0]
            owny = state[self.index][1]
            goalx = state[self.index][6]
            goaly = state[self.index][7]

            #checks for variables denoting terminal states
            if not 0 < ownx < Config.window_width or not 0 < owny < Config.window_height:
                hit_wall = True
                break

            if self.dist_intruder(state, ownx, owny) < Config.minimum_separation:
                conflict = True
                break

            if self.metric(ownx, owny, goalx, goaly) < Config.goal_radius:
                reach_goal = True

        return MultiAircraftState(state, self.index, 'random', hit_wall, conflict, reach_goal, a, self.depth+1)

    def get_legal_actions(self):
        return [0, 1, 2]
    
    def metric(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        f_dist = math.sqrt(dx**2 + dy**2)
        return f_dist

    def dist_goal(self):
        return self.metric(self.ownx, self.owny, self.goalx, self.goaly)

    def dist_intruder(self, state, ownx, owny):
        distance = 5000
        for i in [x for x in range(state.shape[0]) if x != self.index]:
            otherx = state[i][0]
            othery = state[i][1]
            dist = self.metric(ownx, owny, otherx, othery)
            if dist < distance:
                distance = dist
                self.nearest_x = otherx
                self.nearest_y = othery
        return distance

    # state: (x, y, vx, vy, heading angle, gx, gy)
    @property
    def ownx(self):
        return self.state[self.index][0]

    @property
    def owny(self):
        return self.state[self.index][1]

    @property
    def goalx(self):
        return self.state[self.index][6]

    @property
    def goaly(self):
        return self.state[self.index][7]


class MultiAircraftNode(MCTS_Node):
    def __init__(self, state: MultiAircraftState, parent=None):
        MCTS_Node.__init__(self, parent)
        self.state = state

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def reward(self):
        return self.q / self.n if self.n else 0

    def expand(self):
        a = self.untried_actions.pop()
        if self.state.init_action == 'random':
            all_action = np.random.randint(0, 3, size=self.state.state.shape[0])
            # print('rand1')
        else:
            all_action = self.state.init_action.copy()
        all_action[self.state.index] = a
        next_state = self.state.move(all_action)
        child_node = MultiAircraftNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self, search_depth):
        return self.state.is_terminal_state(search_depth)

    def rollout(self, search_depth):
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal_state(search_depth):
            # possible_moves = current_rollout_state.get_legal_actions()
            # action = self.rollout_policy(possible_moves)
            action = np.random.randint(0, 3, size=self.state.state.shape[0])
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.reward()

    def backpropagate(self, result):
        self.n += 1
        self.q += result
        if self.parent:
            self.parent.backpropagate(result)
