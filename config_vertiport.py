import math
import numpy as np

##
print('Loading the VTOL Port......')

## The
# pointy_hex_corner 
def vtol_station_arrangement(center, size, i):
    theta = 90 * i
    theta_rad = math.radians(theta)
    return (center[0] + size * math.cos(theta_rad),
            center[1] + size * math.sin(theta_rad))


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
    # represents meters per pixel
    scale = 60 

    # distance param
    #minimum_separation 
    minSep = 555/scale
    #NMAC_dist
    nmacDist = 150/scale
    #horizon_dist
    horDist= 4000/scale
    #initial_min_dist
    initMinDistance= 3000/scale
    #goal_radius
    goalRadius= 600/scale

    # speed
    #init_speed
    initialVelocity= 60/scale
    #min_speed
    minVelocity= 50/scale
    #max_speed
    maxVelocity= 80/scale
    #d_speed
    d_vel= 5/scale
    #speed_sigma
    vel_sigma= 2/scale
    #position_sigma
    pos_sigma = 0

    # heading in rad TBD
    d_heading = math.radians(5)
    heading_sigma = math.radians(2)

    # MCTS algorithm
    ## need to check with Soumick first for changes
    no_simulations = 100
    search_depth = 3
    no_simulations_lite = 10
    search_depth_lite = 2
    simulate_frame = 10

    # reward setting
    #NMAC_penalty
    nmacPenalty= -10
    #conflict_penalty
    conflictPenalty = -5
    #wall_penalty
    wallPenalty = -5
    #step_penalty
    stepPenalty= -0.01
    #goal_reward
    goalReward = 20
    #sparse_reward
    sparseReward= True

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
        VerticalPortLocation[i, :] = vtol_station_arrangement(VerticalPortCenter, size=300, i=i)
