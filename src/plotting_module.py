import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# def plot_3d(df, subject_number, task_number, output_directory_csv: str = "output"):
#     """
#     Plot the 3D data of the task
#     :param task_dataframe: the dataframe of the task
#     :param subject_number: the number of the subject
#     :param task_number: the number of the task
#     :param output_directory_csv: the output directory of the csv files
#     :return: None
#     """
#     if not os.path.exists(output_directory_csv):
#         os.makedirs(output_directory_csv)
#
#     df.to_csv(os.path.join(output_directory_csv, f"Subject_{subject_number}_Task_{task_number}.csv"), index=False)
#
#     df['z'] = df.apply(lambda row: 0 if row['Pressure'] != 0 else row['Pressure'], axis=1)
#
#     # Extracting 'PointX', 'PointY', and the adjusted 'Pressure' (z) for plotting
#     x = df['PointX']
#     y = df['PointY']
#     z = df['z']
#
#     # Plotting
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # Use plot3D or plot for a 3D line plot, adjusting the line width with 'linewidth'
#     ax.plot(x, y, z, color='r', marker='o', linewidth=0.5)  # Adjust linewidth to your preference
#     ax.set_xlabel('PointX')
#     ax.set_ylabel('PointY')
#     ax.set_zlabel('Pressure (Adjusted)')
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # ax.plot(task_dataframe['x'], task_dataframe['y'], task_dataframe['z'], marker='o')
#     # ax.set_xlabel('X Label')
#     # ax.set_ylabel('Y Label')
#     # ax.set_zlabel('Z Label')
#     # plt.title(f"Subject {subject_number} - Task {task_number}")
#     # plt.savefig(os.path.join(output_directory_csv, f"Subject_{subject_number}_Task_{task_number}.png"))
#     # plt.show()


def plot_3d(df, subject_number, task_number, output_directory_csv: str = "output"):
    """
    Plot the 3D data of the task
    :param df: the dataframe of the task
    :param subject_number: the number of the subject
    :param task_number: the number of the task
    :param output_directory_csv: the output directory of the csv files
    :return: None
    """
    # Extract 'PointX', 'PointY', and 'PointZ'
    df['z'] = df.apply(lambda row: 0 if row['Pressure'] != 0 else row['Pressure'], axis=1)

    # Extracting 'PointX', 'PointY', and the adjusted 'Pressure' (z) for plotting
    x = df['PointX']
    y = df['PointY']
    z = df['z']

    # Create a time parameter for interpolation
    t = np.arange(len(x))
    t_new = np.linspace(t.min(), t.max(), 5 * len(x))  # Increase points for smoother curve

    # Cubic spline interpolation for all dimensions
    x_interp = interp1d(t, x, kind='cubic')(t_new)
    y_interp = interp1d(t, y, kind='cubic')(t_new)
    z_interp = interp1d(t, z, kind='cubic')(t_new)  # Interpolate 'PointZ'

    # Plotting in 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')  # Set up for 3D plotting
    ax.plot3D(x_interp, y_interp, z_interp, 'b-', linewidth=0.5)  # Plot interpolated 3D curve
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.set_zlim(0, 36000)  # Set limits for z axis, adjust as necessary
    ax.set_xlabel('PointX')
    ax.set_ylabel('PointY')
    ax.set_zlabel('PointZ')
    ax.set_title('Interpolated 3D Curve of PointX, PointY, and PointZ')
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
