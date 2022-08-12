#! /usr/bin/env python
"""
Methods for visualizing the a robot arm and trajectories
@author: Achille
"""
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import gridspec
from sympy.plotting import plot_implicit
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import measure
from mayavi import mlab
from pyface.api import GUI


plt.style.use('seaborn-dark')

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


class CuspidalVisualizer:

    def __init__(self, kinematics_model, enable_3d=True):
        self.kinematics = kinematics_model
        self.enable_3d = enable_3d

        if self.enable_3d:
            try:
                self.engine = mayavi.engine
            except NameError:
                from mayavi.api import Engine
                self.engine = Engine()
                self.engine.start()

        plt.ion()
        self.fig = plt.figure(figsize=(18,10))
        self.color1 = 'black'
        self.color2 = 'deepskyblue'
        self.color3 = 'chartreuse'
        self.axis_color = 'white'
        self.fig.patch.set_facecolor(self.color1)
        gs = gridspec.GridSpec(1, 3, width_ratios = [3, 1, 1]) 
        gs.update(wspace=0.1, hspace=0.05) # set the spacing between axes. 

        self.ax = plt.subplot(gs[0], projection='3d')
        # self.ax = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.ax3 = plt.subplot(gs[2])

        ax1_xr = (-0.6, 3)
        ax1_yr = (-2, 2)
        ax1_zr = (-0.05, 1)
        resoln = 0.02

        xl = np.linspace(ax1_xr[0], ax1_xr[1], int(np.diff(ax1_xr)[0]/resoln))
        yl = np.linspace(ax1_yr[0], ax1_yr[1], int(np.diff(ax1_yr)[0]/resoln))
        zl = np.linspace(ax1_zr[0], ax1_zr[1], int(np.diff(ax1_zr)[0]/resoln))
        X, Y, Z = np.meshgrid(xl, yl, zl)
        F = self.kinematics.quartic_discriminant(np.sqrt(X**2 + Y**2), Z)
        verts, faces, normals, values = measure.marching_cubes(F, level=0, spacing=[resoln, resoln, resoln])
        verts[:, 0] += ax1_yr[0]
        verts[:, 1] += ax1_xr[0]
        verts[:, 2] += ax1_zr[0]
        # should use custom cmap
        # faces[:,[0, 1]] = faces[:,[1, 0]]
        # plot_trisurf(X, Y, triangles)
        # self.ax.plot_trisurf(verts[:, 1], verts[:, 0], faces, verts[:, 2], cmap='inferno', lw=0, alpha=0.7)

        if self.enable_3d:
            mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, opacity=0.7, transparent=True)
            module_manager = self.engine.scenes[0].children[0].children[0].children[0]
            module_manager.scalar_lut_manager.lut_mode = 'summer'
            self.mfig = mlab.gcf()
            # mlab.show()


        self.ax.set_autoscale_on(False)
        self.ax.set_xlim3d(ax1_xr)
        self.ax.set_ylim3d(ax1_yr)
        self.ax.set_zlim3d(ax1_zr)
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

        plt.tight_layout()
        plt.show()

        self.joint_angles = np.zeros(3)
        self.plotted_lines = []
        self.skip = 0
        self.angle = -190
        self.ax.view_init(8, self.angle)
        # self.update()

    def update(self, joint_angles=np.zeros(3)):
        """Show the robot in the given state"""
        origins = self.kinematics.origins_viz(joint_angles)
        try:
            if self.skip > 50:
                self.ax.lines.pop(0)
            else:
                self.skip += 1
        except IndexError: pass

        if self.skip <= 2:
            self.plotted_lines.append(self.ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o', color=self.color2, linewidth=5, markersize=8, alpha=1.0))
            # mlab.plot3d(origins[:, 0], origins[:, 1], origins[:, 2], figure=self.mfig)
            # mlab.points3d(origins[:, 0], origins[:, 1], origins[:, 2], figure=self.mfig, scale_factor=0.1)
        # self.plotted_lines.append(mlab.plot3d(origins[:, 0], origins[:, 1], origins[:, 2], figure=self.mfig))
        elif self.skip > 2:
            # color the previous line in a different color
            # last_point = self.engine.scenes[0].children[-1].children[0].children[0]
            # last_line = self.engine.scenes[0].children[-2].children[0].children[0].children[0].children[0]
            # last_line.actor.property.opacity = 0.15
            # last_point.actor.property.opacity = 0.15
                # line.set_color('lightskyblue')
                # line.set_alpha(0.4)
            #     line.set_linewidth(3)
            #     line.set_markersize(6)
            # mlab.plot3d(origins[:, 0], origins[:, 1], origins[:, 2], figure=self.mfig)
            # mlab.points3d(origins[:, 0], origins[:, 1], origins[:, 2], figure=self.mfig, scale_factor=0.1)
            for prev_lines in self.plotted_lines[-min(5, len(self.plotted_lines)):]:
                for link in prev_lines:
                    link.set_alpha(max(0, link.get_alpha() - 0.15))
                    link.set_linewidth(1)
            self.plotted_lines.append(self.ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o', color=self.color2, linewidth=2, markersize=8, alpha=1.0))

        # set the ee color
        self.ax.plot(origins[-1, 0], origins[-1, 1], origins[-1, 2], 'o', color=self.color3, markersize=5)

        # show where the robot is on the other plots
        rho, zee = self.kinematics.forward_kinematics_2D(joint_angles)
        self.ax2.plot([rho], [zee], marker='x', markersize=10, color=self.color3)
        self.ax3.plot([joint_angles[1]], [joint_angles[2]], marker='x', markersize=10, color=self.color3)

        # rotate robot view
        self.angle -= 0.2
        self.ax.view_init(8, self.angle)

        # GUI().process_events()
        # arr = mlab.screenshot()
        # self.ax.imshow(arr)
        self.fig.canvas.draw()
        # mlab.draw(figure=self.mfig)
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
