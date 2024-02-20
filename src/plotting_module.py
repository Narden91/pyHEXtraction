import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.collections as mcollection
import matplotlib.colors as mcolors


def plot_3d(df, subject_number, task_number, output_directory: str = "Plotting"):
    """
    Plot the 3D data of the task
    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param task_number: the number of the task
    :param output_directory: the output directory of the plot
    :return: None
    """
    # Extract 'PointX', 'PointY', and 'PointZ'
    # df['z'] = df.apply(lambda row: 0 if row['Pressure'] != 0 else row['Pressure'], axis=1)

    # Extracting 'PointX', 'PointY', and the adjusted 'Pressure' (z) for plotting
    x = df['PointX']
    y = df['PointY']

    # Subtract z from the maximum value to invert the z-axis
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
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a continuous norm to map from data points to colors
    for i in range(len(x_interp) - 1):
        ax.plot(x_interp[i:i + 2], y_interp[i:i + 2], z_interp[i:i + 2], color=colors[i], linewidth=1)

    # ax.plot3D(x_interp, y_interp, z_interp, 'b-', linewidth=0.5)  # Plot interpolated 3D curve
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.set_zlim(0, df['Pressure'].max())  # Set limits for z axis, adjust as necessary
    ax.set_xlabel('PointX')
    ax.set_ylabel('PointY')
    ax.set_zlabel('Pressure')
    ax.zaxis.set_ticklabels([])  # Hide z-axis labels
    ax.set_title(f'Subject {subject_number} - Task {task_number}')
    plt.show()

    # Save the plot
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(os.path.join(output_directory, f"Subject_{subject_number}_Task_{task_number}.png"))


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
