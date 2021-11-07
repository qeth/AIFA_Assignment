import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding

from config_vertiport import Config
from AircraftClasses import Aircraft,AircraftDictionary,Goal,VerticalPort
class MultiAircraftEnv(gym.Env):

    def __init__(self, sd=2):
        self.load_config()  # load parameters for the simulator
        self.load_vertiport()  # load config for the vertiports
        self.state = None
        self.viewer = None

        # build observation space and action space
        self.observation_space = self.build_observation_space()  # observation space deprecated, not in use for MCTS
        self.position_range = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height]),
            dtype=np.float32)  # position range is the length and width of airspace
        self.action_space = spaces.Tuple((spaces.Discrete(3),) * self.n_evtol)
        # action space deprecated, since number of aircraft is changing from time to time

        self.conflicts = 0
        self.seed(sd)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_config(self):
        # input dim
        self.window_width = Config.window_width
        self.window_height = Config.window_height  # dimension of the airspace
        self.n_evtol = Config.n_evtol
        self.epochs = Config.epochs
        # self.pixel_meter = Config.pixel_meter
        self.scale = Config.scale  # 1 meter = ? pixels, set to 60 here
        self.minSep = Config.minSep
        self.nmacDist = Config.nmacDist
        # self.horDist = Config.horDist
        self.initMinDistance = Config.initMinDistance  # when aircraft generated, is shouldn't be too close to others
        self.goalRadius = Config.goalRadius
        self.initialVelocity = Config.initialVelocity
        self.minVelocity = Config.minVelocity
        self.maxVelocity = Config.maxVelocity

    def load_vertiport(self):
        self.vertiport_list = []
        # read the vertiport location from config file
        for i in range(Config.VerticalPortLocation.shape[0]):
            self.vertiport_list.append(VerticalPort(id=i, position=Config.VerticalPortLocation[i]))

    def reset(self):
        # aircraft is stored in this dict
        self.aircraft_dict = AircraftDictionary()
        self.id_tracker = 0  # assign id to newly generated aircraft, increase by one after generating aircraft.

        # keep track of number of conflicts, goals, and NMACs.
        self.conflicts = 0
        self.goals = 0
        self.NMACs = 0

        return self._get_ob()


    def _get_ob(self):
        s = []
        id = []
        # loop all the aircraft
        # return the information of each aircraft and their respective id
        # s is in shape [number_aircraft, 8], id is list of length number_aircraft
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # (x, y, vx, vy, speed, heading, gx, gy)
            s.append(aircraft.position[0])
            s.append(aircraft.position[1])
            s.append(aircraft.velocity[0])
            s.append(aircraft.velocity[1])
            s.append(aircraft.speed)
            s.append(aircraft.heading)
            s.append(aircraft.goal.position[0])
            s.append(aircraft.goal.position[1])

            id.append(key)

        return np.reshape(s, (-1, 8)), id

    def _get_normalized_ob(self):
        # state contains pos, vel for all intruder aircraft
        # pos, vel, speed, heading for ownship
        # goal pos
        def normalize_velocity(velocity):
            translation = velocity + self.maxVelocity
            return translation / (self.maxVelocity * 2)

        s = []
        id = []
        # loop all the aircraft
        # return the information of each aircraft and their respective id
        # s is in shape [number_aircraft, 8], id is list of length number_aircraft
        for key, aircraft in self.aircraft_dict.ac_dict.items():
            # (x, y, vx, vy, speed, heading, gx, gy)
            s.append(aircraft.position[0] / Config.window_width)
            s.append(aircraft.position[1] / Config.window_height)
            s.append(normalize_velocity(aircraft.velocity[0]))
            s.append(normalize_velocity(aircraft.velocity[1]))
            s.append((aircraft.speed - Config.minVelocity) / (Config.maxVelocity - Config.minVelocity))
            s.append(aircraft.heading / (2 * math.pi))
            s.append(aircraft.goal.position[0] / Config.window_width)
            s.append(aircraft.goal.position[1] / Config.window_height)

            id.append(key)

        return np.reshape(s, (-1, 8)), id

    def step(self, a, near_end=False):
        # a is a dictionary: {id: action, id: action, ...}
        # since MCTS is used every 5 seconds, there may be new aircraft generated during the 5 time step interval, which
        # MCTS algorithm doesn't generate an action for it. In this case we let it fly straight.
        for id, aircraft in self.aircraft_dict.ac_dict.items():
            try:
                aircraft.step(a[id])
            except KeyError:
                aircraft.step()

        for vertiport in self.vertiport_list:
            vertiport.step()  # increase the clock of vertiport by 1
            # generate new aircraft if the clock pass the interval
            if vertiport.clock_counter >= vertiport.time_next_aircraft and not near_end:
                goal_vertiport_id = random.choice([e for e in range(len(self.vertiport_list)) if not e == vertiport.id])
                # generate new aircraft and prepare to add it the dict
                aircraft = Aircraft(
                    id=self.id_tracker,
                    position=vertiport.position,
                    speed=self.initialVelocity,
                    heading=self.random_heading(),
                    goal_pos=self.vertiport_list[goal_vertiport_id].position
                )
                # calc its dist to all the other aircraft
                dist_array, id_array = self.dist_to_all_aircraft(aircraft)
                min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
                # add it to dict only if it's far from others
                if min_dist > 3 * self.minSep:  # and self.aircraft_dict.n_evtol < 10:
                    self.aircraft_dict.add(aircraft)
                    self.id_tracker += 1  # increase id_tracker

                    vertiport.generate_interval()  # reset clock for this vertiport and generate a new time interval

        # return the reward, done, and info
        reward, terminal, info = self._terminal_reward()

        return self._get_ob(), reward, terminal, info

    def _terminal_reward(self):
        """
        determine the reward and terminal for the current transition, and use info. Main idea:
        1. for each aircraft:
          a. if there a conflict, return a penalty for it
          b. if there is NMAC, assign a penalty to it and prepare to remove this aircraft from dict
          b. elif it is out of map, assign its reward as Config.wallPenalty, prepare to remove it
          c. elif if it reaches goal, assign its reward to Config.goalReward, prepare to remove it
          d. else assign its reward as Config.stepPenalty.
        3. remove out-of-map aircraft and goal-aircraft

        """
        reward = 0
        # info = {'n': [], 'c': [], 'w': [], 'g': []}
        info_dist_list = []
        aircraft_to_remove = []  # add goal-aircraft and out-of-map aircraft to this list

        for id, aircraft in self.aircraft_dict.ac_dict.items():
            # calculate min_dist and dist_goal for checking terminal
            dist_array, id_array = self.dist_to_all_aircraft(aircraft)
            min_dist = min(dist_array) if dist_array.shape[0] > 0 else 9999
            info_dist_list.append(min_dist)
            dist_goal = self.dist_goal(aircraft)

            conflict = False
            for id, dist in zip(id_array, dist_array):
                if dist >= self.minSep:  # safe
                    aircraft.conflict_id_set.discard(id)  # discarding element not in the set won't raise error

                else:  # conflict!!
                    conflict = True
                    if id not in aircraft.conflict_id_set:
                        self.conflicts += 1
                        aircraft.conflict_id_set.add(id)
                        # info['c'].append('%d and %d' % (aircraft.id, id))
                    aircraft.reward = Config.conflictPenalty

            if min_dist < self.nmacDist:
                aircraft.reward = Config.nmac_penalty
                aircraft_to_remove.append(aircraft)
                self.NMACs += 1

            elif not self.position_range.contains(np.array(aircraft.position)):
                aircraft.reward = Config.wallPenalty
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)

            elif dist_goal < self.goalRadius:
                aircraft.reward = Config.goalReward
                self.goals += 1
                if aircraft not in aircraft_to_remove:
                    aircraft_to_remove.append(aircraft)
                    
            elif not conflict:
                aircraft.reward = Config.stepPenalty
                
            reward += aircraft.reward
        for aircraft in aircraft_to_remove:
            self.aircraft_dict.remove(aircraft)
    
        return reward, False, info_dist_list


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # dist to all the aircraft
    def dist_to_all_aircraft(self, aircraft):
        id_list = []
        dist_list = []
        for id, intruder in self.aircraft_dict.ac_dict.items():
            if id != aircraft.id:
                id_list.append(id)
                dist_list.append(self.metric(aircraft.position, intruder.position))

        return np.array(dist_list), np.array(id_list)

    def dist_goal(self, aircraft):
        return self.metric(aircraft.position, aircraft.goal.position)

    def metric(self, pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def random_pos(self):
        return np.random.uniform(
            low=np.array([0, 0]),
            high=np.array([self.window_width, self.window_height])
        )

    def random_speed(self):
        return np.random.uniform(low=self.minVelocity, high=self.maxVelocity)

    def random_heading(self):
        return np.random.uniform(low=0, high=2 * math.pi)

    def build_observation_space(self):
        s = spaces.Dict({
            'pos_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'pos_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
            'vel_x': spaces.Box(low=-self.maxVelocity, high=self.maxVelocity, shape=(1,), dtype=np.float32),
            'vel_y': spaces.Box(low=-self.maxVelocity, high=self.maxVelocity, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=self.minVelocity, high=self.maxVelocity, shape=(1,), dtype=np.float32),
            'heading': spaces.Box(low=0, high=2 * math.pi, shape=(1,), dtype=np.float32),
            'goal_x': spaces.Box(low=0, high=self.window_width, shape=(1,), dtype=np.float32),
            'goal_y': spaces.Box(low=0, high=self.window_height, shape=(1,), dtype=np.float32),
        })

        return spaces.Tuple((s,) * self.n_evtol)