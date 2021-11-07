import math
import numpy as np
 
def vtol_station_arrangement(center, size, i):
    theta = 90 * i
    theta_rad = math.radians(theta)
    return (center[0] + size * math.cos(theta_rad),
            center[1] + size * math.sin(theta_rad))
class Config:

    n_epochs= 5

    window_width = 800
    window_height = 800

    n_evtol = 2

    epochs = 1000
    g = 9.8

    pixel_meter = 30
    scale = 60 

    minSep = 555/scale
    nmacDist = 150/scale
    horDist= 4000/scale
    initMinDistance= 3000/scale
    goalRadius= 600/scale


    initialVelocity= 60/scale
    minVelocity= 50/scale
    maxVelocity= 80/scale
    d_vel= 5/scale
    vel_sigma= 2/scale
    pos_sigma = 0

    d_heading = math.radians(5)
    heading_sigma = math.radians(2)

    no_simulations = 100
    search_depth = 3
    no_simulations_lite = 10
    search_depth_lite = 2
    simulate_frame = 10

    nmacPenalty= -10
    conflictPenalty = -5
    wallPenalty = -5
    stepPenalty= -0.01
    goalReward = 20
    sparseReward= True

    minTimeInterwal = 60
    maxTimeInterval = 180
    VerticalPortLocation= np.zeros([7, 2])
    VerticalPortCenter = np.array([window_width/2, window_height/2])
    VerticalPortLocation[0, :] = VerticalPortCenter
    for i in range(1, 7):
        VerticalPortLocation[i, :] = vtol_station_arrangement(VerticalPortCenter, size=300, i=i)
