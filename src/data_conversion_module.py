import numpy as np
import pandas as pd
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from handwriting_sample import HandwritingSample


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def tilt_azimuth_transformation(df: pd.DataFrame):
    """ Correct the tilt and azimuth values from the DataFrame to be used with the HandwritingSample library

    Args:
        df (pd.DataFrame): DataFrame containing the tilt and azimuth values
    """

    # Initialize Tilt and Azimuth with default values
    df['Tilt'] = np.pi / 2
    df['Azimuth'] = 0

    # Conditions
    condition_Y_positive = df['TiltY'] > 0
    condition_Y_negative = df['TiltY'] < 0
    condition_X_positive = df['TiltX'] > 0
    condition_X_negative = df['TiltX'] < 0

    # Apply transformations
    df.loc[condition_Y_positive, 'Tilt'] = np.pi / 2 - df['TiltY']
    df.loc[condition_Y_positive, 'Azimuth'] = np.pi / 2

    df.loc[condition_Y_negative, 'Tilt'] = np.pi / 2 + df['TiltY']
    df.loc[condition_Y_negative, 'Azimuth'] = 3 * np.pi / 2

    df.loc[condition_X_positive & ~condition_Y_positive & ~condition_Y_negative, 'Tilt'] = np.pi / 2 - df['TiltX']

    df.loc[condition_X_negative & ~condition_Y_positive & ~condition_Y_negative, 'Tilt'] = np.pi / 2 + df['TiltX']
    df.loc[condition_X_negative, 'Azimuth'] = np.pi

    # Handle non-zero TiltX and TiltY
    non_zero_conditions = (df['TiltX'] != 0) & (df['TiltY'] != 0)
    df.loc[non_zero_conditions, 'Azimuth'] = np.arctan(df['TiltY'] / df['TiltX'])
    df.loc[non_zero_conditions, 'Tilt'] = np.arctan(np.sin(df['Azimuth']) / df['TiltY'])

    # Ensure non-negative values
    df['Tilt'] = df['Tilt'].abs()
    df['Azimuth'] = df['Azimuth'].abs()

    return df


def convert_to_HandwritingSample_library_format(data_source: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """ Convert the data from the csv to a HandwritingSample object using the HandwritingFeatures library

    Args:
        data_source (pd.DataFrame): Dataframe containing the data from the csv file
        verbose (bool): If True, print the dataframe before and after the transformation
    Returns:
        pd.DataFrame: Dataframe containing the data to HandwritingSample object-ready
    """

    print(f"Dataframe before transformation: \n{data_source.head(5).to_string()}") if verbose else None

    # Create pen_status column
    data_source['pen_status'] = np.where(data_source['Pressure'] != 0, 1, 0)

    # Correct tilt and azimuth values
    data_source = tilt_azimuth_transformation(data_source)

    print(f"Dataframe after adding column: \n{data_source.head(5).to_string()}") if verbose else None

    # Extract, Reorder and Rename the columns of the dataframe
    data_source = data_source.iloc[:, [1, 2, 0, 10, 6, 11, 4]]

    # Rename the columns
    data_source.columns = ['x', 'y', 'time', 'pen_status', 'azimuth', 'tilt', 'pressure']

    return data_source


def get_handwriting_feature_dataframe(data_source: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the data from the dataframe and return a dataframe ready to be used with the HandwritingFeatures library

    Args:
        data_source (pd.DataFrame): Dataframe containing the data
    Returns:
        pd.DataFrame: Dataframe containing the data for HandwritingFeatures library
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
    #
    # # Add the metadata to the HandwritingSample object
    # handwriting_task.add_meta_data(meta_data=meta_data)
    #
    # # Transform all units
    # handwriting_task.transform_all_units()

    # Create a HandwritingSample object from the dataframe
    handwriting_task = HandwritingSample.from_pandas_dataframe(data_source)

    return handwriting_task


def stroke_segmentation(data_source: pd.DataFrame, meta_data_enabled: bool = False,
                        verbose: bool = False):
    """
    Segments the data into strokes based on HandwritingSample library
    Segmentation is done based on the pen_status column (pen-up and pen-down)

    Args:
        data_source (pd.DataFrame): Dataframe containing the data
        meta_data_enabled (bool): If True, add the metadata to the HandwritingSample object
        verbose (bool): If True, print the dataframes of the strokes
    Returns:
        pd.DataFrame: Dataframe containing the data segmented into strokes
    """

    # Avoid printing the HandwritingFeatures class output
    with suppress_stdout_stderr():
        # Create a HandwritingSample object from the dataframe
        handwriting_task = HandwritingSample.from_pandas_dataframe(data_source)

    if meta_data_enabled:
        print(f"Adding metadata to the HandwritingSample object")
        # Meta data of the device Wacom One 13.3
        meta_data = {"protocol_id": "dsa_2023",
                     "device_type": "Wacom One 13.3",
                     "device_driver": "2.1.0",
                     "lpi": 2540,  # lines per inch
                     "time_series_ranges": {
                         "x": [0, 1920],
                         "y": [0, 1080],
                         "azimuth": [0, 180],
                         "tilt": [0, 90],
                         "pressure": [0, 32767]}}

        # Add the metadata to the HandwritingSample object
        handwriting_task.add_meta_data(meta_data=meta_data)

        # Transform all units
        handwriting_task.transform_all_units()

    # Get all strokes sequentially based on how the task was performed
    strokes = handwriting_task.get_strokes()

    # Print stroke's dataframes
    print_strokes_dataframes(strokes) if verbose else None

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


def update_stroke_list_for_all_computation(stroke_list: list) -> list:
    """ Update the first element of the list to be used with the HandwritingSample library
    The first element of the list is the stroke type (OnSurface or InAir)
    The HandwritingSample library uses the following strings: "on_surface" and "in_air"

    Args:
        stroke_list (list): List of the strokes
    Returns:
        list: List of the strokes with the updated first element
    """
    # Create a new list of tuples with the updated first elements
    updated_stroke_list = [(get_new_first_element(old_first_element), second_element) for
                           old_first_element, second_element in stroke_list]

    return updated_stroke_list


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


def get_new_first_element(old_first_element):

    return "on_surface" if old_first_element == "in_air" else old_first_element
