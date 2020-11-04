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

# create the path
# waypoints = load_default()

# planner = Planner(logger=logging)
# configurations = planner.plan(waypoints)

theta2, theta3, rho, zee = symbols('theta2 theta3 rho zee')
kinematics = Manipulator1()

# visualize
viz = CuspidalVisualizer(kinematics)
viz.update()
# viz.plot_waypoints(waypoints)

plot_detJ = plot_implicit(Eq(kinematics.determinant_jacobian(theta2, theta3), 0), x_var=(theta2, -pi, pi), y_var=(theta3, -pi, pi), show=False)
viz.move_sympyplot_to_axes(plot_detJ, viz.ax3)

plot_discr = plot_implicit(Eq(kinematics.quartic_discriminant(rho, zee), 0), x_var=(rho, 0, 5), y_var=(zee, -4, 4), show=False, adaptive=False, points=400)
viz.move_sympyplot_to_axes(plot_discr, viz.ax2)

# print(kinematics.origins([0, 0, 0]))
# print(kinematics.forward_kinematics_2D([0, 0, 0]))

viz.pause(2)

xee = 1.79
yee = 1.75
print("rho: {}".format(np.sqrt(xee**2 + yee**2)))
zee = 0.5
all_solns = kinematics.ik(xee, yee, zee)
# for soln in all_solns:
#     print(kinematics.origins(soln)[2, :])
#     print(np.sqrt(kinematics.origins(soln)[2, 0]**2 + kinematics.origins(soln)[2, 1]**2))
#     viz.update(soln)
#     viz.pause(0.5)

interpolated_angles = np.linspace(all_solns[1], all_solns[2], num=50)
print(interpolated_angles)

for i in range(len(interpolated_angles)):
    # joints = kinematics.random_valid_config()
    angles = interpolated_angles[i]
    viz.update(angles)
    viz.pause(0.1)

done = input("enter when done: ")
viz.shutdown()
