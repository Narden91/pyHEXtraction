import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_sample import HandwritingSample
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def tilt_azimuth_transformation(row: pd.Series):
    """ Correct the tilt and azimuth values from the csv file to be used with the HandwritingSample library

    Args:
        row (pd.Series): Row of the dataframe containing the tilt and azimuth values
    """
    # Tilt and Azimuth correction
    if row['TiltX'] == 0 and row['TiltY'] == 0:
        return np.pi / 2, 0
    elif row['TiltX'] == 0 and row['TiltY'] > 0:
        return np.pi / 2 - row['TiltY'], np.pi / 2
    elif row['TiltX'] == 0 and row['TiltY'] < 0:
        return np.pi / 2 + row['TiltY'], 3 * np.pi / 2
    elif row['TiltX'] > 0 and row['TiltY'] == 0:
        return np.pi / 2 - row['TiltX'], 0
    elif row['TiltX'] < 0 and row['TiltY'] == 0:
        return np.pi / 2 + row['TiltX'], np.pi
    else:
        azimuth = np.arctan(np.tan(row['TiltY']) / np.tan(row['TiltX']))
        tilt = np.arctan(np.sin(azimuth) / np.tan(row['TiltY']))
        return tilt, azimuth


def convert_to_HandwritingSample_library(data_source: pd.DataFrame) -> pd.DataFrame:
    """ Convert the data from the csv to a HandwritingSample object using the HandwritingFeatures library
    If the plot looks weird, go to preprocessing.py and check the coordinates_manipulation function

    Args:
        data_source (pd.DataFrame): Dataframe containing the data from the csv file
    Returns:
        pd.DataFrame: Dataframe containing the data to HandwritingSample object-ready
    """

    # Create a new column pen_status with 1 if pressure is not 0 and 0 if pressure is 0
    data_source['pen_status'] = np.where(data_source['Pressure'] != 0, 1, 0)

    # Create new column Tilt that is the first positive value of the TiltX and TiltY columns
    data_source['Tilt'] = np.where(data_source['TiltX'] > 0, data_source['TiltX'], data_source['TiltY'])

    # Extract, Reorder and Rename the columns of the dataframe
    data_source = data_source.iloc[:, [1, 2, 0, 10, 6, 11, 4]]

    # Rename the columns
    data_source.columns = ['x', 'y', 'time', 'pen_status', 'azimuth', 'tilt', 'pressure']

    # # Correct Tilt and Azimuth for Library HandwritingSample
    # data_source[["Tilt", "Azimuth_1"]] = data_source.apply(lambda row:
    #                                                        pd.Series(tilt_azimuth_transformation(row)), axis=1)
    #
    # # Drop the columns TiltX, TiltY, Azimuth
    # data_source = data_source.drop(columns=['TiltX', 'TiltY', 'Azimuth'])
    #
    # # Rename the column Azimuth_1 to Azimuth
    # data_source = data_source.rename(columns={'Azimuth_1': 'Azimuth'})
    #
    # # Extract, Reorder and Rename the columns of the dataframe
    # data_source = data_source.iloc[:, [1, 2, 0, 7, 9, 8, 4]]
    #
    # # Rename the columns
    # data_source.columns = ['x', 'y', 'time', 'pen_status', 'azimuth', 'tilt', 'pressure']
    #
    # # Absolute value of the tilt and azimuth
    # data_source['tilt'] = data_source['tilt'].abs()
    # data_source['azimuth'] = data_source['azimuth'].abs()

    return data_source


def get_strokes_from_data(data_source: pd.DataFrame):
    """ Get the strokes using the HandwritingSample library

    Args:
        data_source (pd.DataFrame): Dataframe containing the data
    Returns:
        None
    """

    # Meta data of the device Wacom One 13.3
    # meta_data = {"protocol_id": "dsa_2023",
    #              "device_type": "Wacom One 13.3",
    #              "device_driver": "2.1.0",
    #              "lpi": 2540,  # lines per inch
    #              "time_series_ranges": {
    #                  "x": [0, 1920],
    #                  "y": [0, 1080],
    #                  "azimuth": [0, 180],
    #                  "tilt": [0, 90],
    #                  "pressure": [0, 32767]}}

    # Create a HandwritingSample object from the dataframe
    handwriting_task = HandwritingSample.from_pandas_dataframe(data_source)

    # Add the metadata to the HandwritingSample object
    # handwriting_task.add_meta_data(meta_data=meta_data)

    # Transform all units
    # handwriting_task.transform_all_units()

    # get all strokes sequentially based on how the task was performed
    strokes = handwriting_task.get_strokes()

    # Print stroke's dataframes
    # print_strokes_dataframes(strokes)

    # get on surface strokes
    # stroke_on_surface = handwriting_task.get_on_surface_strokes()

    # get in air strokes
    # strokes_in_air = handwriting_task.get_in_air_strokes()

    # Plot all the strokes with different colors
    # handwriting_task.plot_strokes()

    # Plot the strokes (InAir and OnSurface) separately (plum for InAir and blue for OnSurface)
    # handwriting_task.plot_separate_movements()

    # Show in air data
    # handwriting_task.plot_in_air()

    # Plot all the data (x, y, azimuth, tilt, pressure)
    # handwriting_task.plot_all_data()

    return strokes


def print_strokes_dataframes(stroke_list: list):
    """ Print the dataframes.
    The strokes are stored in lists with this format: [stroke_number, HandwritingSample object].
    So we need to iterate over the list and get the HandwritingSample object to access the dataframe
    Then we print the dataframe using the data_pandas_dataframe attribute function from HandwritingSample class

    Args:
        stroke_list (list): List of the strokes
    Returns:
        None
    """
    print(f"For the current Task there are {len(stroke_list)} Strokes in total")

    # Iterate over the strokes on the surface
    for num, strokes in enumerate(stroke_list):
        stroke = strokes[1]  # Access the HandwritingSample object
        stroke_type = strokes[0]  # Access the stroke type (OnSurface or InAir)
        print(f"[+] Stroke {num + 1} -> {stroke_type}. \nDataframe:")
        print(stroke.data_pandas_dataframe)  # Print the dataframe
        print("---------------------------------------------------------------")

    return None


def get_stroke_features(strokes: list) -> pd.DataFrame:
    """ Get the features of the strokes

    Args:
        strokes (list): List of the strokes
    Returns:
        list: List of the features of the strokes
    """

    # Create an empty dataframe
    feature_dataframe = None

    # Iterate over the strokes
    for num, stroke in enumerate(strokes):
        # Get the Stroke type
        stroke_type = stroke[0]

        # Get the HandwritingSample object
        stroke_dataframe = stroke[1].data_pandas_dataframe

        if stroke_type == "in_air":
            # Set pen_status to 1 to trick the HandwritingFeatures class
            # and get the features of the in air stroke
            stroke_dataframe['pen_status'] = 1

        # Debug print
        # print(f"[+] Stroke {num + 1} -> {stroke_type}.")
        # print(f"[+] \nDataframe: \n{stroke_dataframe}")

        # Avoid printing the HandwritingFeatures class output
        with suppress_stdout_stderr():
            # Create a HandwritingFeatures object from the stroke dataframe
            feature_data = HandwritingFeatures.from_pandas_dataframe(stroke_dataframe)

        # Get the kinematic features
        kinematic_features_dict = get_kinematic_features(feature_data)

        # Get the dynamic features
        dynamic_features_dict = get_dynamic_features(feature_data)

        # Create a dictionary with num
        kinematic_features_dict = {"stroke_id": num + 1, "stroke_type": stroke_type,
                                   **kinematic_features_dict, **dynamic_features_dict}

        if feature_dataframe is None:
            # Create Pandas Dataframe from the kinematic features dictionary
            feature_dataframe = pd.DataFrame(kinematic_features_dict, index=[0])
        else:
            # Concatenate the dataframes
            feature_dataframe = pd.concat([feature_dataframe,
                                           pd.DataFrame([kinematic_features_dict])], ignore_index=True)

        # # Print the final  dataframe
        # print(feature_dataframe.to_string())

    return feature_dataframe


def get_kinematic_features(data: HandwritingFeatures) -> dict:
    """ Get the kinematic features of the strokes.
    Available_statistics = {
        "mean": mean,
        "std": std,
        "cv_parametric": cv_parametric,
        "median": median,
        "iqr": iqr,
        "cv_nonparametric": cv_nonparametric,
        "quartile_1": quartile_1,
        "quartile_3": quartile_3,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "slope_of_linear_regression": slope_of_linear_regression
    }

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary of the kinematic features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    kinematic_feature_dict = {'x_velocity_mean': round(data.velocity(axis="x", statistics=["mean"])[0], 3),
                              'x_velocity_std': round(data.velocity(axis="x", statistics=["std"])[0], 3),
                              'y_velocity_mean': round(data.velocity(axis="y", statistics=["mean"])[0], 3),
                              'y_velocity_std': round(data.velocity(axis="y", statistics=["std"])[0], 3),
                              'xy_velocity_mean': round(data.velocity(axis="xy", statistics=["mean"])[0], 3),
                              'xy_velocity_std': round(data.velocity(axis="xy", statistics=["std"])[0], 3),
                              'x_acceleration_mean': round(data.acceleration(axis="x", statistics=["mean"])[0], 3),
                              'x_acceleration_std': round(data.acceleration(axis="x", statistics=["std"])[0], 3),
                              'y_acceleration_mean': round(data.acceleration(axis="y", statistics=["mean"])[0], 3),
                              'y_acceleration_std': round(data.acceleration(axis="y", statistics=["std"])[0], 3),
                              'xy_acceleration_mean': round(data.acceleration(axis="xy", statistics=["mean"])[0], 3),
                              'xy_acceleration_std': round(data.acceleration(axis="xy", statistics=["std"])[0], 3),
                              'x_jerk_mean': round(data.jerk(axis="x", statistics=["mean"])[0], 3),
                              'x_jerk_std': round(data.jerk(axis="x", statistics=["std"])[0], 3),
                              'y_jerk_mean': round(data.jerk(axis="y", statistics=["mean"])[0], 3),
                              'y_jerk_std': round(data.jerk(axis="y", statistics=["std"])[0], 3),
                              'xy_jerk_mean': round(data.jerk(axis="xy", statistics=["mean"])[0], 3),
                              'xy_jerk_std': round(data.jerk(axis="xy", statistics=["std"])[0], 3)}

    return kinematic_feature_dict


def get_dynamic_features(data: HandwritingFeatures) -> dict:
    """ Get the dynamic features of the strokes.
    Available_statistics = {
        "mean": mean,
        "std": std,
        "cv_parametric": cv_parametric,
        "median": median,
        "iqr": iqr,
        "cv_nonparametric": cv_nonparametric,
        "quartile_1": quartile_1,
        "quartile_3": quartile_3,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "slope_of_linear_regression": slope_of_linear_regression
    }

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary of the dynamic features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    dynamic_feature_dict = {'azimuth_mean': round(data.azimuth(statistics=["mean"])[0], 3),
                            'azimuth_std': round(data.azimuth(statistics=["std"])[0], 3),
                            'tilt_mean': round(data.tilt(statistics=["mean"])[0], 3),
                            'tilt_std': round(data.tilt(statistics=["std"])[0], 3),
                            'pressure_mean': round(data.pressure(statistics=["mean"])[0], 3),
                            'pressure_std': round(data.pressure(statistics=["std"])[0], 3)}

    return dynamic_feature_dict
