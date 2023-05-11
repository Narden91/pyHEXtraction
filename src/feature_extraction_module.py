import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_sample import HandwritingSample


def convert_to_HandwritingSample_library(data_source: pd.DataFrame) -> pd.DataFrame:
    """ Convert the data from the csv to a HandwritingSample object using the HandwritingFeatures library
    If the plot looks weird, go to preprocessing.py and check the coordinates_manipulation function"""

    # Create a new column pen_status with 1 if pressure is not 0 and 0 if pressure is 0
    data_source['pen_status'] = np.where(data_source['Pressure'] != 0, 1, 0)

    # # Create new column tilt where the value is taken from TiltX if its value is positive, otherwise from TiltY
    # data_source['tilt'] = np.where(data_source['TiltX'] > 0, data_source['TiltX'], data_source['TiltY'])

    # Extract and reorder and rename the columns of the dataframe
    data_source = data_source.iloc[:, [1, 2, 0, 10, 6, 8, 4]]

    # Rename the columns
    data_source.columns = ['x', 'y', 'time', 'pen_status', 'azimuth', 'tilt', 'pressure']

    # make positive the values of tilt
    data_source['tilt'] = data_source['tilt'].abs()

    return data_source


def plot_using_HandwritingSample_library(data_source: pd.DataFrame):
    """ Plot the data using the HandwritingSample library"""

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

    # Add the metadata to the HandwritingSample object
    # sample.add_meta_data(meta_data=meta_data)

    # Transform all units
    sample.transform_all_units()

    # print(sample.x)

    # get all strokes
    strokes = sample.get_strokes()

    #sample.plot_strokes()

    print(len(strokes))

    # Show separate movements
    # sample.plot_separate_movements()

    # Show in air data
    # sample.plot_in_air()

    # Show all data
    # sample.plot_all_data()
    return None
