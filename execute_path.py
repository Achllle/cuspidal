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

# visualize
viz = CuspidalVisualizer()
viz.update()
# viz.plot_waypoints(waypoints)

theta2, theta3, rho, zee = symbols('theta2 theta3 rho zee')
kinematics = OrthoManip1Kinematics()

plot_detJ = plot_implicit(Eq(kinematics.determinant_jacobian(theta2, theta3), 0), x_var=(theta2, -pi, pi), y_var=(theta3, -pi, pi), show=False)
viz.move_sympyplot_to_axes(plot_detJ, viz.ax3)

plot_discr = plot_implicit(Eq(kinematics.quartic_discriminant(rho, zee), 0), x_var=(rho, 0, 5), y_var=(zee, -4, 4), show=False, adaptive=False, points=400)
viz.move_sympyplot_to_axes(plot_discr, viz.ax2)

# print(kinematics.origins([0, 0, 0]))
# print(kinematics.forward_kinematics_2D([0, 0, 0]))

# for i in range(20):
#     joints = kinematics.random_valid_config()
#     viz.update(joints)
#     viz.pause(0.1)

done = input("enter when done: ")
viz.shutdown()
