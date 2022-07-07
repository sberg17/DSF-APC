import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
# Define font and font size used for plotting
plt.rcParams["font.family"] = "Times New Roman"
s_font = 9
m_font = 10
l_font = 12
plt.rc('font', size=s_font)  # controls default text sizes
plt.rc('axes', titlesize=l_font)  # font size of the axes title
plt.rc('axes', labelsize=m_font)  # font size of the x and y labels
plt.rc('xtick', labelsize=s_font)  # font size of the tick labels
plt.rc('ytick', labelsize=s_font)  # font size of the tick labels
plt.rc('legend', fontsize=s_font)  # legend font size
plt.rc('figure', titlesize=l_font)  # font size of the figure title

class Plotter:
    def __init__(self, figsize=(6, 4), dpi=300, colormap="tab10"):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.get_cmap("tab10").colors

    def line_plot(self, y_data, x_data=None, title=None, xlabel="x", ylabel="y", save_name=None):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        if np.array(x_data).all() != None:
            plt.plot(x_data, y_data, color=self.colors[0])
        else:
            plt.plot(y_data, color=self.colors[0])
        if title is not None:
            plt.title(title)
        plt.xlim(0, np.max(x_data)*1.05)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()
        if save_name != None:
            plt.savefig(save_name)
        plt.show()

    def scatter_plot(self, x_data, y_data, title="Scatter plot", xlabel="x", ylabel="y", save_name=None):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for i in range(len(y_data)):
            plt.scatter(x_data[i], y_data[i])
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid()
            plt.tight_layout()
        if save_name != None:
            plt.savefig(save_name + ".png")
        plt.show()

    def array_of_plots(self, y_data, x_data=None, title=None, xlabel="x", ylabel="y"):
        if len(y_data) == 2:
            fig, (ax1, ax2) = plt.subplots(2, figsize=self.figsize, dpi=self.dpi)
            if x_data == None:
                ax1.plot(y_data[0], color=self.colors[0])
                ax2.plot(y_data[1], color=self.colors[1])
            else:
                ax1.plot(x_data[0], y_data[0], color=self.colors[0])
                ax2.plot(x_data[1], y_data[1], color=self.colors[1])
            fig.suptitle(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

        elif len(y_data) == 3:
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=self.figsize, dpi=self.dpi)
            if x_data == None:
                ax1.plot(y_data[0], color=self.colors[0])
                ax2.plot(y_data[1], color=self.colors[1])
                ax3.plot(y_data[2], color=self.colors[2])
            else:
                ax1.plot(x_data[0], y_data[0], color=self.colors[0])
                ax2.plot(x_data[1], y_data[1], color=self.colors[1])
                ax3.plot(x_data[2], y_data[2], color=self.colors[2])
            fig.suptitle(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

    def agent_test(self, n_episode, depths, torques, velocities=None, accelerations=None, target=None,
                   xlabel="x", ylabel="y", save_dir=None, show_plot=True, show_vel_acc=False):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.subplots_adjust(right=0.75)
        p1, = ax.plot(depths, torques, label="Applied torque", color=self.colors[0])
        if target != None:
            p2, = ax.plot(depths, target, label="Friction", color=self.colors[3])
        if show_vel_acc:
            twin1 = ax.twinx()
            twin2 = ax.twinx()
            twin2.spines.right.set_position(("axes", 1.14))
            p3, = twin1.plot(depths, velocities, label="Velocity", color=self.colors[1])
            p4, = twin2.plot(depths, accelerations, label="Acceleration", color=self.colors[2])
            twin1.set_ylabel("Velocity [m/s]")
            twin2.set_ylabel(r"Acceleration [$\mathregular{m/s^{2}}$]")
            twin1.yaxis.label.set_color(p3.get_color())
            twin2.yaxis.label.set_color(p4.get_color())
            twin1.tick_params(axis='y', colors=p3.get_color())
            twin2.tick_params(axis='y', colors=p4.get_color())
        else:
            plt.ylim(0, np.max(torques)+0.75)
        # Put legend to the right of the current axis
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        if show_vel_acc:
            ax.legend(handles=[p1, p2, p3, p4], loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)
        else:
            ax.legend(handles=[p1, p2], loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "plot_test_" + str(n_episode) + ".png"))
        if show_plot:
            plt.show()
