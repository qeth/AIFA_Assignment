import math
import numpy as np

##
print('loading vertiport configuration')

## The 
def pointy_hex_corner(center, size, i):
    angle_deg = 90 * i
    angle_rad = math.radians(angle_deg)
    return (center[0] + size * math.cos(angle_rad),
            center[1] + size * math.sin(angle_rad))


class Config:
    # experiment setting
    #n_episodes 
    n_epochs= 5

    # airspace setting
    window_width = 800
    window_height = 800
    #num_aircraft
    n_evtol = 2
    #EPISODES 
    epochs = 1000
    g = 9.8
    #tick
    pixel_meter = 30
    scale = 60  # 1 pixel = 30 meters

    # distance param
    #minimum_separation 
    min_sep = 555/scale
    NMAC_dist = 150/scale
    horizon_dist = 4000/scale
    initial_min_dist = 3000/scale
    goal_radius = 600/scale

    # speed
    init_speed = 60/scale
    min_speed = 50/scale
    max_speed = 80/scale
    d_speed = 5/scale
    speed_sigma = 2/scale
    position_sigma = 0

    # heading in rad TBD
    d_heading = math.radians(5)
    heading_sigma = math.radians(2)

    # MCTS algorithm
    no_simulations = 100
    search_depth = 3
    no_simulations_lite = 10
    search_depth_lite = 2
    simulate_frame = 10

    # reward setting
    NMAC_penalty = -10
    conflict_penalty = -5
    wall_penalty = -5
    step_penalty = -0.01
    goal_reward = 20
    sparse_reward = True

    # vertiport parameter
    #time_interval_lower
    minTimeInterwal = 60
    #time_interval_upper
    maxTimeInterval = 180
    #vertiport_loc
    VerticalPortLocation= np.zeros([7, 2])
    #vertiport_center 
    VerticalPortCenter = np.array([window_width/2, window_height/2])
    VerticalPortLocation[0, :] = VerticalPortCenter
    for i in range(1, 7):
        VerticalPortLocation[i, :] = pointy_hex_corner(VerticalPortCenter, size=300, i=i)
