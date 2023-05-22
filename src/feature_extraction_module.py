import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_sample import HandwritingSample


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

    # Correct Tilt and Azimuth for Library HandwritingSample
    data_source[["Tilt", "Azimuth_1"]] = data_source.apply(lambda row:
                                                           pd.Series(tilt_azimuth_transformation(row)), axis=1)

    # Drop the columns TiltX, TiltY, Azimuth
    data_source = data_source.drop(columns=['TiltX', 'TiltY', 'Azimuth'])

    # Rename the column Azimuth_1 to Azimuth
    data_source = data_source.rename(columns={'Azimuth_1': 'Azimuth'})

    # Extract, Reorder and Rename the columns of the dataframe
    data_source = data_source.iloc[:, [1, 2, 0, 7, 9, 8, 4]]

    # Rename the columns
    data_source.columns = ['x', 'y', 'time', 'pen_status', 'azimuth', 'tilt', 'pressure']

    # Absolute value of the tilt and azimuth
    data_source['tilt'] = data_source['tilt'].abs()
    data_source['azimuth'] = data_source['azimuth'].abs()

    return data_source


def plot_using_HandwritingSample_library(data_source: pd.DataFrame):
    """ Plot the data using the HandwritingSample library

    Args:
        data_source (pd.DataFrame): Dataframe containing the data
    Returns:
        None
    """
    print(f"[+] Data after Transformation for HandwritingSample Library : \n{data_source}")

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
    sample = HandwritingSample.from_pandas_dataframe(data_source)

    # get on surface strokes
    stroke_on_surface = sample.get_on_surface_strokes()

    # get in air strokes
    strokes_in_air = sample.get_in_air_strokes()

    # Add the metadata to the HandwritingSample object
    # sample.add_meta_data(meta_data=meta_data)

    # Transform all units
    sample.transform_all_units()

    # print(sample.x)

    # get all strokes
    strokes = sample.get_strokes()

    sample.plot_strokes()

    # print(strokes)

    # Show separate movements
    # sample.plot_separate_movements()

    # Show in air data
    # sample.plot_in_air()

    # Show all data
    # sample.plot_all_data()
    return None
