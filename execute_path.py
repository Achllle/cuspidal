#! /usr/bin/env python
"""
Script for executing a specified path
@author: Achille
"""
import time
import argparse
import logging

# from path_loader import * 
from arm_ik import *
from viz_manip import CuspidalVisualizer
from angles import normalize_angle_positive

from sympy import symbols, Eq
from sympy.plotting import plot_implicit
from numpy import pi

logging.basicConfig(level=logging.DEBUG)

theta2, theta3, rho, zee = symbols('theta2 theta3 rho zee')
kinematics = Manipulator2()

# visualize
viz = CuspidalVisualizer(kinematics)
# viz.update()
# viz.plot_waypoints(waypoints)

color = 'snow'
plot_detJ = plot_implicit(Eq(kinematics.determinant_jacobian(theta2, theta3), 0), 
                          x_var=(theta2, -pi, pi), y_var=(theta3, -pi, pi), 
                          line_color=color ,show=False)
viz.move_sympyplot_to_axes(plot_detJ, viz.ax3)

plot_discr = plot_implicit(Eq(kinematics.quartic_discriminant(rho, zee), 0), 
                           x_var=(rho, 0, 3.8), y_var=(zee, -2.8, 2.8), 
                           line_color=color, show=False, adaptive=False, points=800)
viz.move_sympyplot_to_axes(plot_discr, viz.ax2)

# print(kinematics.origins([0, 0, 0]))
# print(kinematics.forward_kinematics_2D([0, 0, 0]))

viz.pause(2)

xee = 1.2
yee = 1.
zee = 0.5


## SHOW SOME POSTURE
# viz.update(joint_angles=np.array([0, -pi/2, 0]))
#####################

## SHOW ALL SOLUTIONS
# all_solns = kinematics.ik(xee, yee, zee)
# for soln in all_solns:
#     print(kinematics.origins(soln)[2, :])
#     print(np.sqrt(kinematics.origins(soln)[2, 0]**2 + kinematics.origins(soln)[2, 1]**2))
#     viz.update(soln)
#     viz.pause(0.5)
#####################

## INTERPOLATE FROM ONE POSTURE TO ANOTHER
all_solns = kinematics.ik(xee, yee, zee)
interpolated_angles = np.linspace(all_solns[0], all_solns[2], num=40)
print(interpolated_angles)

viz.pause(2)

for i in range(len(interpolated_angles)):
    joints = kinematics.random_valid_config()
    angles = interpolated_angles[i]
    viz.update(angles)
#####################

## INTERPOLATE TWO WORKSPACE PATHS TO SHOW JOINT JUMPS
# x_s = 2.45
# y_s = 0.7
# z_s = 0.2
# x_e = 0.3
# y_e = 0.7
# z_e = 0.2


# interp_rhozee = np.linspace(np.array([x_s, y_s, z_s]), np.array([x_e, y_e, z_e]), num=40)

# all_solns = kinematics.ik(x_s, y_s, z_s)
# configs = [all_solns[0]]
# norms = []

# viz.pause(2)

# for x, y, z in interp_rhozee:
#     postures = kinematics.ik(x, y, z)
#     print("number of postures found: {}".format(len(postures)))
#     # pick the closest config
#     best_norm = np.inf
#     best_config = None
#     for posture in postures:
#         norm = sum(np.linalg.norm(np.vstack((configs[-1], posture)), axis=0))
#         if norm < best_norm:
#             best_norm = norm
#             best_config = posture
#     configs.append(best_config)
#     norms.append(best_norm)

# for config in configs:
#     viz.update(config)
#     viz.pause(0.15)

######################################################

done = input("enter when done: ")
viz.shutdown()
