import math
import numpy as np
import random
from collections import OrderedDict

from config_vertiport import Config


class AircraftDictionary:
    def __init__(self):
        self.ac_dict = OrderedDict()

    def n_evtol(self):
        return len(self.ac_dict)

    def add(self, aircraft):
        assert aircraft.id not in self.ac_dict.keys(), 'aircraft id %d already in dict' % aircraft.id
        self.ac_dict[aircraft.id] = aircraft

    def remove(self, aircraft):
        try:
            del self.ac_dict[aircraft.id]
        except KeyError:
            pass

    def get_aircraft_by_id(self, aircraft_id):
        return self.ac_dict[aircraft_id]

class Goal:
    def __init__(self, position):
        self.position = position

class Aircraft:
    def __init__(self, id, position, speed, heading, goal_pos):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.heading = heading  # rad
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy], dtype=np.float32)

        self.reward = 0
        self.goal = Goal(goal_pos)
        dx, dy = self.goal.position - self.position
        self.heading = math.atan2(dy, dx)  

        self.load_config()

        self.conflict_id_set = set()  

    def load_config(self):
        self.g = Config.g
        self.scale = Config.scale
        self.minVelocity = Config.minVelocity
        self.maxVelocity = Config.maxVelocity
        self.vel_sigma = Config.vel_sigma
        self.pos_sigma = Config.pos_sigma
        self.d_heading = Config.d_heading

    def step(self, a=1):
        self.speed = max(self.minVelocity, min(self.speed, self.maxVelocity)) 
        self.speed += np.random.normal(0, self.vel_sigma)  
        self.heading += (a - 1) * self.d_heading  
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.velocity = np.array([vx, vy])

        self.position += self.velocity


class VerticalPort:
    def __init__(self, id, position):
        self.id = id
        self.position = np.array(position)  
        self.clock_counter = 0
        self.time_next_aircraft = np.random.uniform(0, 60)

    def generate_interval(self):
        self.time_next_aircraft = np.random.uniform(Config.time_interval_lower, Config.time_interval_upper)
        self.clock_counter = 0

    def step(self):
        self.clock_counter += 1
