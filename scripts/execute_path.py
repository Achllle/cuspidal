#! /usr/bin/env python
"""
Script for executing a specified path
@author: Achille
"""
import logging

from cuspidal.arm_ik import *
from cuspidal.viz_manip import MatplotlibCuspidalVisualizer, MayaCuspidalVisualizer

logging.basicConfig(level=logging.DEBUG)

# choose a kinematics model, see arm_ik.py
kinematics = Manipulator2()

# visualize, choose which visualizer. Both are possible.
# viz = MatplotlibCuspidalVisualizer(kinematics)
viz = MayaCuspidalVisualizer(kinematics)
# viz.update()

print(kinematics.origins([0, 0, 0]))
print(kinematics.forward_kinematics_2D([0, 0, 0]))

# give user a moment to adjust windows
viz.pause(2)


# SHOWING SOME POSTURE
# viz.update(joint_angles=np.array([0, -pi/2, 0]))
####################

# SHOWING ALL IK SOLUTIONS
# specify an end effector (ee) position
# xee = 1.2
# yee = 1.
# zee = 0.5
# all_solns = kinematics.ik(xee, yee, zee)
# for soln in all_solns:
#     print(kinematics.origins(soln)[2, :])
#     print(np.sqrt(kinematics.origins(soln)[2, 0]**2 + kinematics.origins(soln)[2, 1]**2))
#     viz.update(soln)
#     viz.pause(0.5)
####################

# INTERPOLATE FROM ONE POSTURE TO ANOTHER
# xee = 1.2
# yee = 1.
# zee = 0.5
# all_solns = kinematics.ik(xee, yee, zee)
# interpolated_angles = np.linspace(all_solns[0], all_solns[2], num=40)
# print(interpolated_angles)

# viz.pause(2)

# for i in range(len(interpolated_angles)):
#     joints = kinematics.random_valid_config()
#     angles = interpolated_angles[i]
#     viz.update(angles)
####################

# INTERPOLATE TWO WORKSPACE POINTS TO CREATE PATH
# starting position
x_s = 2.45
y_s = 0.7
z_s = 0.2
# endpoint position
x_e = 0.3
y_e = 0.7
z_e = 0.2


interp_rhozee = np.linspace(np.array([x_s, y_s, z_s]), np.array([x_e, y_e, z_e]), num=40)

all_solns = kinematics.ik(x_s, y_s, z_s)
configs = [all_solns[1]]
norms = []

for x, y, z in interp_rhozee:
    postures = kinematics.ik(x, y, z)
    print("number of postures found: {}".format(len(postures)))
    # pick the closest config
    best_norm = np.inf
    best_config = None
    for posture in postures:
        norm = np.linalg.norm(np.array(posture)-np.array(configs[-1]))
        if norm < best_norm:
            best_norm = norm
            best_config = posture
    configs.append(best_config)
    norms.append(best_norm)

for config, norm in zip(configs, norms):
    viz.update(config)
    # tweak this to adjust speed of animation
    viz.pause(0.05)
######################################################

done = input("enter when done: ")
viz.shutdown()
