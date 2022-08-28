#! /usr/bin/env python
"""
Methods for visualizing the a robot arm and trajectories
@author: Achille
"""
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import gridspec
from sympy.plotting import plot_implicit
import numpy as np
from skimage import measure
from mayavi import mlab
from mayavi.api import Engine
from pyface.api import GUI
import logging
from sympy import symbols, Eq
from sympy.plotting import plot_implicit


class CuspidalVisualizer(object):
    link_colors = [(1, 0, 0), (1, 0, 0.5), (1, 0, 1), (0.5, 0, 1), (0, 0, 1)]
    color1 = 'black'
    color2 = 'deepskyblue'
    color3 = 'chartreuse'

    def __init__(self, kinematics_model):
        self.kinematics = kinematics_model

        # axes lenghts
        self.ax1_xr = (-0.6, 3)
        self.ax1_yr = (-2, 2)
        self.ax1_zr = (-0.05, 1)

        self.joint_angles = np.zeros(3)
        self.nb_updates = 0
        self.angle = -190

    def update(self, joint_angles=np.zeros(3)):
        """Show the robot in the given state"""
        self.nb_updates += 1

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

    def pause(self, duration=0.05):
        """pause the plot, duration in s"""
        time.sleep(duration)

    def shutdown(self):
        """Close the figure"""
        pass


class MayaCuspidalVisualizer(CuspidalVisualizer):
    resoln = 0.02

    def __init__(self, kinematics_model):
        super().__init__(kinematics_model)

        self.engine = Engine()
        self.engine.start()

        xl = np.linspace(self.ax1_xr[0], self.ax1_xr[1], int(np.diff(self.ax1_xr)[0]/self.resoln))
        yl = np.linspace(self.ax1_yr[0], self.ax1_yr[1], int(np.diff(self.ax1_yr)[0]/self.resoln))
        zl = np.linspace(self.ax1_zr[0], self.ax1_zr[1], int(np.diff(self.ax1_zr)[0]/self.resoln))
        X, Y, Z = np.meshgrid(xl, yl, zl)
        F = self.kinematics.quartic_discriminant(np.sqrt(X**2 + Y**2), Z)
        verts, faces, normals, values = measure.marching_cubes(F, level=0, spacing=[self.resoln, self.resoln, self.resoln])
        verts[:, 0] += self.ax1_yr[0]
        verts[:, 1] += self.ax1_xr[0]
        verts[:, 2] += self.ax1_zr[0]

        mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, opacity=0.7, transparent=True)
        module_manager = self.engine.scenes[0].children[0].children[0].children[0]
        module_manager.scalar_lut_manager.lut_mode = 'summer'
        self.mfig = mlab.gcf()
        # mlab.show()
        GUI().process_events()

    def update(self, joint_angles=np.zeros(3)):
        super().update(joint_angles)

        origins = self.kinematics.origins_viz(joint_angles)

        if self.nb_updates > 2:
            # set opacity of previously generated robot. The indexing depends on the number of generated scenes in Mayavi, see comments
            # each robot has 4 links (4 scenes total) and 5 joints (2 scenes total since the end effector is plotted separately)
            scene_items = self.engine.scenes[0].children
            last_ee_point = scene_items[-1].children[0].children[0]
            last_points = scene_items[-2].children[0].children[0]
            last_ee_point.actor.property.opacity = 0.3
            last_points.actor.property.opacity = 0.2
            for line in scene_items[-(len(origins)-1)-2:-2]:
                line.children[0].children[0].children[0].children[0].actor.property.opacity = 0.1

        for joint in range(1, len(origins)):
            # one mayavi scene for each call
            mlab.plot3d(origins[joint-1:joint+1, 0], origins[joint-1:joint+1, 1], origins[joint-1:joint+1, 2], figure=self.mfig, color=self.link_colors[joint-1])

        # generates one scene for each call
        mlab.points3d(origins[:-1, 0], origins[:-1, 1], origins[:-1, 2], figure=self.mfig, scale_factor=0.1, opacity=0.8)
        mlab.points3d(origins[-1, 0], origins[-1, 1], origins[-1, 2], figure=self.mfig, scale_factor=0.15, color=(1,0.3,0), opacity=0.7)

        # # small delay to allow visualizing
        # mlab.process_ui_events()
        GUI().process_events()

    # def pause(self, duration=0.05):
    #     time.sleep(duration)

    # def shutdown(self):
    #     super().shutdown()


class MatplotlibCuspidalVisualizer(CuspidalVisualizer):

    def __init__(self, kinematics_model):
        super().__init__(kinematics_model)

        # keep history of plotted postures so we can add some effects
        self.plotted_lines = []

        plt.style.use('seaborn-dark')
        logging.getLogger('matplotlib.font_manager').disabled = True

        params = {'legend.fontsize': 'x-large',
                'axes.labelsize': 'large',
                'axes.titlesize':'x-large',
                'xtick.labelsize':'x-large',
                'ytick.labelsize':'x-large'}
        pylab.rcParams.update(params)

        plt.ion()
        self.fig = plt.figure(figsize=(18,10))

        self.axis_color = 'white'
        self.fig.patch.set_facecolor(self.color1)
        gs = gridspec.GridSpec(1, 3, width_ratios = [3, 1, 1]) 
        gs.update(wspace=0.1, hspace=0.05) # set the spacing between axes. 

        self.ax = plt.subplot(gs[0], projection='3d')
        # self.ax = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.ax3 = plt.subplot(gs[2])

        self.ax.set_autoscale_on(False)
        self.ax.set_xlim3d(self.ax1_xr)
        self.ax.set_ylim3d(self.ax1_yr)
        self.ax.set_zlim3d(self.ax1_zr)
        self.ax.set_aspect('auto')
        self.ax.set_xlabel('$X$')
        self.ax.set_ylabel('$Y$')
        self.ax.set_zlabel('$Z$')
        self.ax.set_facecolor(self.color1)
        # self.ax.set_proj_type('ortho')
        self.ax2.set_facecolor(self.color1)
        self.ax3.set_facecolor(self.color1)
        self.ax2.yaxis.label.set_color(self.axis_color)
        self.ax3.yaxis.label.set_color(self.axis_color)
        self.ax2.xaxis.label.set_color(self.axis_color)
        self.ax3.xaxis.label.set_color(self.axis_color)

        self.ax2.spines['bottom'].set_color(self.axis_color)
        self.ax3.spines['bottom'].set_color(self.axis_color)
        self.ax2.spines['bottom'].set_color(self.axis_color)
        self.ax3.spines['left'].set_color(self.axis_color)
        self.ax2.tick_params(colors=self.axis_color)
        self.ax3.tick_params(colors=self.axis_color)

        # create symbolic angles for plotting Jacobian discriminant with sympy
        theta2, theta3, rho, zee = symbols('theta2 theta3 rho zee')
        color = 'snow'
        plot_detJ = plot_implicit(Eq(self.kinematics.determinant_jacobian(theta2, theta3), 0), 
                                x_var=(theta2, -np.pi, np.pi), y_var=(theta3, -np.pi, np.pi), 
                                line_color=color ,show=False)
        self.move_sympyplot_to_axes(plot_detJ, self.ax3)

        plot_discr = plot_implicit(Eq(self.kinematics.quartic_discriminant(rho, zee), 0), 
                                x_var=(rho, 0, 3.8), y_var=(zee, -2.8, 2.8), 
                                line_color=color, show=False, adaptive=False, points=800)
        self.move_sympyplot_to_axes(plot_discr, self.ax2)

        # set viewing angles on first (3D) figure
        self.ax.view_init(8, self.angle)
        plt.show()

    def update(self, joint_angles=np.zeros(3)):
        super().update(joint_angles)

        # remove old lines to avoid cluttering the view
        try:
            if self.nb_updates > 50:
                self.ax.lines.pop(0)
        except IndexError: pass

        origins = self.kinematics.origins_viz(joint_angles)

        if self.nb_updates > 2:
            # fade $skip_amount most recent lines, don't touch older lines as they've been faded enough
            for prev_lines in self.plotted_lines[-2:-1]:
                for link in prev_lines:
                    link.set_alpha(max(0, link.get_alpha() * 0.8))
                    link.set_linewidth(link.get_linewidth() * 0.9)  # fade by factor 10% each time

        for joint in range(1, len(origins)):
            lines = self.ax.plot3D(origins[joint-1:joint+1, 0], origins[joint-1:joint+1, 1], origins[joint-1:joint+1, 2], '-o', linewidth=2*(len(origins) - joint), markersize=10, alpha=1.0, c=self.link_colors[joint-1])
            self.plotted_lines.append(lines)

        # set the ee color
        self.ax.plot(origins[-1, 0], origins[-1, 1], origins[-1, 2], 'o', color=self.color3, markersize=5)

        # show where the robot is on the other plots
        rho, zee = self.kinematics.forward_kinematics_2D(joint_angles)
        self.ax2.plot([rho], [zee], marker='x', markersize=10, color=self.color3)
        self.ax3.plot([joint_angles[1]], [joint_angles[2]], marker='x', markersize=10, color=self.color3)

        # rotate robot view
        self.angle -= 0.2
        self.ax.view_init(8, self.angle)

        plt.pause(0.001)  # tiny delay to allow for visualizing

    def move_sympyplot_to_axes(self, sympy_plot, ax):
        backend = sympy_plot.backend(sympy_plot)
        backend.ax = ax
        backend._process_series(backend.parent._series, ax, backend.parent)
        backend.ax.spines['right'].set_color('none')
        backend.ax.spines['bottom'].set_position('zero')
        backend.ax.spines['top'].set_color('none')
        plt.close(backend.fig)

    def pause(self, duration=0.05):
        plt.pause(duration)

    def shutdown(self):
        plt.close(self.fig)


if __name__ == '__main__':
    from arm_ik import NonCuspidal2
    viz2D = MatplotlibCuspidalVisualizer(NonCuspidal2())
    viz3D = MayaCuspidalVisualizer(NonCuspidal2())
    input("hit any key to close.")
