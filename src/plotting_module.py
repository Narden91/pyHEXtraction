import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.collections as mcollection
import matplotlib.colors as mcolors
import seaborn as sns


def bar_plot_times(df, subject_number, task_number, output_directory: str = "Plotting"):
    """
    Plot the bar plot of the task

    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param task_number: the number of the task
    :param output_directory: the output directory of the plot
    :return: None
    """
    df['Status'] = df['Pressure'].apply(lambda x: 'On-paper' if x != 0 else 'In-air')

    time_summary = df.groupby('Status')['Time'].count() * (df['Time'].iloc[1] - df['Time'].iloc[0])
    time_summary = time_summary.reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Status', y='Time', data=time_summary, palette="Set2")
    plt.title(f'Subject {subject_number} - Task {task_number} Total Time Spent In-air vs On-paper', fontsize=16)
    plt.xlabel('Status')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    # Save the plot
    filename_png = os.path.join(output_directory, f"Subject_{subject_number}_bar_plot.png")
    filename_svg = os.path.join(output_directory, f"Subject_{subject_number}_bar_plot.svg")

    plt.savefig(filename_png)
    plt.savefig(filename_svg, format='svg', bbox_inches='tight')

    plt.show()


def kde_plot(df, subject_number, output_directory: str = "Plotting"):
    """
    Plot the box plot of the task

    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param output_directory: the output directory of the plot
    :return: None
    """
    plt.figure(figsize=(12, 8))
    sns.kdeplot(df['Pressure'], fill=True, alpha=0.5, color="blue")
    plt.title('Density Plot of Pressure Levels')
    plt.xlabel('Pressure Level')
    plt.ylabel('Density')
    # plt.legend()

    # Save the plot
    filename_png = os.path.join(output_directory, f"Subject_{subject_number}_kde_plot.png")
    filename_svg = os.path.join(output_directory, f"Subject_{subject_number}_kde_plot.svg")

    plt.savefig(filename_png)
    plt.savefig(filename_svg, format='svg', bbox_inches='tight')

    plt.show()


def plot_task(df, subject_number, task_number, output_directory: str = "Plotting"):
    """
    Plot the 3D and the 2D data of the task
    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param task_number: the number of the task
    :param output_directory: the output directory of the plot
    :return: None
    """
    x = df['PointX']
    y = df['PointY']
    z = df['Pressure'].max() - df['Pressure']

    # Create a time parameter for interpolation
    t = np.arange(len(x))
    t_new = np.linspace(t.min(), t.max(), 5 * len(x))  # Increase points for smoother curve

    # Cubic spline interpolation for all dimensions
    x_interp = interp1d(t, x, kind='cubic')(t_new)
    y_interp = interp1d(t, y, kind='cubic')(t_new)
    z_interp = interp1d(t, z, kind='cubic')(t_new)  # Interpolate 'PointZ'

    # Normalize z to [0, 1] for colormap
    z_norm = (z_interp - z_interp.min()) / (z_interp.max() - z_interp.min())
    colors = plt.cm.RdBu(z_norm)  # Colormaps: viridis, plasma, inferno, magma, cividis

    # Plotting in 3D
    fig = plt.figure(figsize=(18, 8))

    ax = fig.add_subplot(131, projection='3d')
    # Create a continuous norm to map from data points to colors
    for i in range(len(x_interp) - 1):
        ax.plot(x_interp[i:i + 2], y_interp[i:i + 2], z_interp[i:i + 2], color=colors[i], linewidth=1)

    z_max = df['Pressure'].max()

    # get the label single values for the task
    label = df['Label'].iloc[0]

    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.set_zlim(0, z_max)
    ax.set_xlabel('PointX')
    ax.set_ylabel('PointY')
    ax.set_zlabel('Pressure')
    ax.zaxis.set_ticks([0, z_max/4, z_max/2, z_max*0.75, z_max])
    ax.zaxis.set_ticklabels(['1', '0.75', '0.5', '0.25', '0'])  # Set tick labels for z axis [] for empty labels
    ax.set_title(f'3D Task execution')

    ax1 = fig.add_subplot(122)
    for i in range(len(x_interp) - 1):
        if z_interp[i] > 0:
            color = 'blue'
        else:
            color = 'red'
        ax1.plot(x_interp[i:i + 2], y_interp[i:i + 2], color=color, linewidth=2)

    ax1.set_xlabel('PointX')
    ax1.set_ylabel('PointY')
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(0, 1080)
    ax1.set_title('2D Plot Task Execution')
    ax1.grid(True)
    fig.suptitle(f'Subject Class {label} - Task {task_number}', fontsize=20)

    plt.tight_layout()

    path_to_save = os.path.join(output_directory, f"Task_{task_number}")
    os.makedirs(path_to_save, exist_ok=True)

    # Save the plot
    filename_png = os.path.join(path_to_save, f"Subject_Class_{label}_task_plot.png")
    # filename_svg = os.path.join(path_to_save, f"Subject_Class_{label}_task_plot.svg")

    plt.savefig(filename_png)
    # plt.savefig(filename_svg, format='svg', bbox_inches='tight')
    plt.show()


def plot_2d(df, subject_number, task_number, output_directory_csv: str = "output"):
    """
    Plot the 2D data of the task
    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param task_number: the number of the task
    :param output_directory_csv: the output directory of the csv files
    :return: None
    """
    # Focus on 'PointX' and 'PointY'
    x = df['PointX'].values
    y = df['PointY'].values

    # Interpolating the points
    # Since we don't have a direct parameter like time that's strictly increasing and uniformly distributed,
    # we'll create one based on the index, assuming equal time steps between points
    t = np.arange(len(x))
    t_new = np.linspace(t.min(), t.max(), 5 * len(x))  # Increase the number of points for a smoother curve

    # Cubic spline interpolation
    x_interp = interp1d(t, x, kind='cubic')(t_new)
    y_interp = interp1d(t, y, kind='cubic')(t_new)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_interp, y_interp, 'b-', linewidth=0.5)  # Plot interpolated curve without markers
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    plt.xlabel('PointX')
    plt.ylabel('PointY')
    plt.title('Interpolated Curve of PointX and PointY')
    plt.show()


def plot_3dd(df, subject_number, task_number, output_directory: str = "Plotting"):
    # Create a figure
    fig = plt.figure(figsize=(20, 12))

    # Create a 2D XY plot
    ax2 = fig.add_subplot(122)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax2.plot(x, y)
    ax2.set_title('XY 2D Plot')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')

    # Create a 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x ** 2 + y ** 2))
    z_max = z.max()
    ax1.plot_surface(x, y, z, cmap='viridis')
    ax1.set_xlabel('PointX')
    ax1.set_ylabel('PointY')
    ax1.set_zlabel('Pressure')
    ax1.zaxis.set_ticks([0, z_max])
    ax1.zaxis.set_ticklabels(['1', '0'])
    # ax1.set_box_aspect([1, 4, 2])
    ax1.set_title('3D Plot')

    # Add title on top of the figure
    fig.suptitle('3D and 2D Plots', fontsize=16)

    plt.tight_layout()
    plt.show()
