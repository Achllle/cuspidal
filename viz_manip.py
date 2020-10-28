#! /usr/bin/env python
"""
Methods for visualizing the a robot arm and trajectories
@author: Achille
"""
import matplotlib.pyplot as plt
from sympy.plotting import plot_implicit
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from arm_ik import OrthoManip1Kinematics


class CuspidalVisualizer:

    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        plt.show()
        # self.ax.set_autoscale_on(False)
        # self.ax.set_xlim3d((-1, 2))
        # self.ax.set_ylim3d((-1, 2))
        # self.ax.set_zlim3d((0, 2))
        self.ax.set_aspect('auto')
        self.kinematics = OrthoManip1Kinematics()
        self.joint_angles = np.zeros(3)
        self.plotted_lines = []
        self.skip = 0
        # self.update()

    def update(self, joint_angles=np.zeros(3)):
        """Show the robot in the given state"""
        o1_3 = self.kinematics.origins(joint_angles)
        origins = np.vstack((np.array([0, 0, 0]), o1_3))
        try:
            # if self.skip > 4:
                self.ax.lines.pop(-1)
            # else:
            #     self.skip += 1
        except IndexError: pass
        self.plotted_lines.append(self.ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], color='r'))

        # show where the robot is on the other plots
        rho, zee = self.kinematics.forward_kinematics_2D(joint_angles)
        self.ax2.plot([rho], [zee], marker='o', markersize=3, color='red')
        self.ax3.plot([joint_angles[1]], [joint_angles[2]], marker='o', markersize=3, color='red')

        self.fig.canvas.draw()
        plt.pause(0.001)  # tiny delay to allow for visualizing

    def plot_waypoints(self, target_poses, style='arrow'):
        """
        Plot cartesian points in the robot's workspace
        :param style: arrow | point
                    When set to arrow, the arrow is along the z-axis of the waypoint
        :param target_poses: np.array of size N by 4 by 4
        """
        if style == 'arrow':
            self.ax.quiver(target_poses[:, 0, 3], target_poses[:, 1, 3], target_poses[:, 2, 3],
                           target_poses[:, 0, 2], target_poses[:, 1, 2], target_poses[:, 2, 2],
                           normalize=True, length=0.07, pivot='tip')
        elif style == 'point':
            self.ax.scatter(target_poses[:, 0, 3], target_poses[:, 1, 3], target_poses[:, 2, 3])
        else:
            raise AttributeError('style must be one off arrow|point')

    def pause(self, time):
        """pause the plot, time in s"""
        plt.pause(time)

    def shutdown(self):
        """Close the figure"""
        plt.close(self.fig)

    def move_sympyplot_to_axes(self, sympy_plot, ax):
        backend = sympy_plot.backend(sympy_plot)
        backend.ax = ax
        backend._process_series(backend.parent._series, ax, backend.parent)
        backend.ax.spines['right'].set_color('none')
        backend.ax.spines['bottom'].set_position('zero')
        backend.ax.spines['top'].set_color('none')
        plt.close(backend.fig)


if __name__ == '__main__':
    viz = CuspidalVisualizer()
