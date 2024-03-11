import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

from handwriting_sample import HandwritingSample


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


# ----------------------------- Features Extraction ------------------------------------- #
def stroke_approach_feature_extraction(strokes: list, task_dataframe: pd.DataFrame,
                                       verbose: bool = False) -> pd.DataFrame:
    """ Compute the features of the strokes and the global features for
    the stroke approach feature extraction.

    Args:
        strokes (list): Strokes list of the task
        task_dataframe (pd.DataFrame): Dataframe containing the data for compute the global features
        verbose (bool): Boolean to print the features shape
    Returns:
        feature_dataframe: dataframe containing the features of the global features and stroke features
    """
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

    # Create an empty dataframe
    feature_dataframe = None

    # Iterate over the stroke list
    # TODO: Add features that provide the x,y of the start and end of the stroke
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
            handwriting_df = HandwritingSample.from_pandas_dataframe(task_dataframe)

            # Transform the data to mm
            handwriting_df.transform_axis_to_mm(conversion_type=HandwritingSample.transformer.LPI,
                                                lpi_value=meta_data["lpi"],
                                                shift_to_zero=False)

            # Transform the azimuth and tilt to degrees
            handwriting_df.transform_angle_to_degree(angle=HandwritingSample.TILT)
            handwriting_df.transform_angle_to_degree(angle=HandwritingSample.AZIMUTH)

            # Create a HandwritingFeatures object from the stroke dataframe
            feature_data = HandwritingFeatures.from_sample(handwriting_df)

        # Get the kinematic features
        kinematic_features_dict = get_kinematic_features(feature_data)

        # Get the dynamic features
        dynamic_features_dict = get_dynamic_features(feature_data)

        # Get the spatial features
        spatial_features_dict = get_spatial_features(feature_data)

        # Get the temporal features
        temporal_features_dict = get_temporal_features(feature_data)

        # Create a dictionary with num
        features_dict = {"stroke_id": num + 1, "stroke_type": stroke_type,
                         **kinematic_features_dict, **dynamic_features_dict,
                         **spatial_features_dict, **temporal_features_dict}

        if feature_dataframe is None:
            # Create Pandas Dataframe from the kinematic features dictionary
            feature_dataframe = pd.DataFrame(features_dict, index=[0])
        else:
            # Concatenate the dataframes
            feature_dataframe = pd.concat([feature_dataframe,
                                           pd.DataFrame([features_dict])], ignore_index=True)

    # Avoid printing the HandwritingFeatures class output
    with suppress_stdout_stderr():
        handwriting_df_glob = HandwritingSample.from_pandas_dataframe(task_dataframe)

        # Transform the data to mm
        handwriting_df_glob.transform_axis_to_mm(conversion_type=HandwritingSample.transformer.LPI,
                                                 lpi_value=meta_data["lpi"],
                                                 shift_to_zero=False)

        # Transform the azimuth and tilt to degrees
        handwriting_df_glob.transform_angle_to_degree(angle=HandwritingSample.TILT)
        handwriting_df_glob.transform_angle_to_degree(angle=HandwritingSample.AZIMUTH)

        # Create a HandwritingFeatures object from the task dataframe to get the global features
        feature_global_data = HandwritingFeatures.from_sample(handwriting_df_glob)

    # Get the composite features
    # composite_features_dict = get_composite_features(feature_global_data)

    # Get the global spatial features
    global_spatial_features_dict = get_global_spatial_features(feature_global_data)

    # Get the global temporal features
    global_temporal_features_dict = get_global_temporal_features(feature_global_data)

    global_features_dict = {**global_spatial_features_dict, **global_temporal_features_dict}

    # iterate over the global features dictionary
    for key, value in global_features_dict.items():
        # Add the global features to the dataframe
        feature_dataframe[key] = value

    return feature_dataframe


def get_kinematic_features(data: HandwritingFeatures, in_air=False) -> dict:
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
    -------------------------------------------------------------
    a) x velocity
    b) y velocity
    c) x acceleration
    d) y acceleration
    e) x jerk
    f) y jerk
    g) xy velocity
    h) xy acceleration
    i) xy jerk

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
        in_air (bool): Boolean to get the in air features
    Returns:
        dict: Dictionary of the kinematic features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    return {'x_velocity_mean': round(data.velocity(axis="x", in_air=in_air, statistics=["mean"])[0], 3),
            'x_velocity_std': round(data.velocity(axis="x", in_air=in_air, statistics=["std"])[0], 3),
            'y_velocity_mean': round(data.velocity(axis="y", in_air=in_air, statistics=["mean"])[0], 3),
            'y_velocity_std': round(data.velocity(axis="y", in_air=in_air, statistics=["std"])[0], 3),
            'xy_velocity_mean': round(data.velocity(axis="xy", in_air=in_air, statistics=["mean"])[0], 3),
            'xy_velocity_std': round(data.velocity(axis="xy", in_air=in_air, statistics=["std"])[0], 3),
            'x_acceleration_mean': round(data.acceleration(axis="x", in_air=in_air, statistics=["mean"])[0], 3),
            'x_acceleration_std': round(data.acceleration(axis="x", in_air=in_air, statistics=["std"])[0], 3),
            'y_acceleration_mean': round(data.acceleration(axis="y", in_air=in_air, statistics=["mean"])[0], 3),
            'y_acceleration_std': round(data.acceleration(axis="y", in_air=in_air, statistics=["std"])[0], 3),
            'xy_acceleration_mean': round(data.acceleration(axis="xy", in_air=in_air, statistics=["mean"])[0], 3),
            'xy_acceleration_std': round(data.acceleration(axis="xy", in_air=in_air, statistics=["std"])[0], 3),
            'x_jerk_mean': round(data.jerk(axis="x", in_air=in_air, statistics=["mean"])[0], 3),
            'x_jerk_std': round(data.jerk(axis="x", in_air=in_air, statistics=["std"])[0], 3),
            'y_jerk_mean': round(data.jerk(axis="y", in_air=in_air, statistics=["mean"])[0], 3),
            'y_jerk_std': round(data.jerk(axis="y", in_air=in_air, statistics=["std"])[0], 3),
            'xy_jerk_mean': round(data.jerk(axis="xy", in_air=in_air, statistics=["mean"])[0], 3),
            'xy_jerk_std': round(data.jerk(axis="xy", in_air=in_air, statistics=["std"])[0], 3)}


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
    -------------------------------------------------------------
    a) Azimuth
    b) Tilt
    c) Pressure

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


def get_spatial_features(data: HandwritingFeatures) -> dict:
    """ Get the spatial features of the strokes.
    a) stroke length
    b) stroke height
    c) stroke width
    d) writing length
    e) writing height
    f) writing width
    g) writing duration

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary of the spatial features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    return {'stroke_length': round(data.stroke_length()[0], 3),
            'stroke_height': round(data.stroke_height()[0], 3),
            'stroke_width': round(data.stroke_width()[0], 3),
            'writing_length': round(data.writing_length()[0], 3),
            'writing_height': round(data.writing_height()[0], 3),
            'writing_width': round(data.writing_width()[0], 3),
            }


def get_temporal_features(data: HandwritingFeatures) -> dict:
    """ Get the temporal features of the strokes.
    a) stroke duration

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary of the temporal features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    return {'stroke_duration': round(data.stroke_duration(statistics=["mean"])[0], 3),
            }


def get_composite_features(data: HandwritingFeatures) -> dict:
    """ Get the composite features of the strokes.
    a) writing tempo
    b) writing stops
    c) number of changes in x profile
    d) number of changes in y profile
    e) number of changes in azimuth
    f) number of changes in tilt
    g) number of changes in pressure
    h) number of changes in velocity profile
    i) relative number of changes in x profile
    j) relative number of changes in y profile
    k) relative number of changes in azimuth
    l) relative number of changes in tilt
    m) relative number of changes in pressure
    n) relative number of changes in velocity profile
    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary of the composite features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    fs = 200

    return {'writing_tempo': round(data.writing_tempo()[0], 3),
            'writing_stops': round(data.writing_stops()[0], 3),
            'number_of_changes_in_x_profile': round(data.number_of_changes_in_x_profile(fs=fs), 3),
            'number_of_changes_in_y_profile': round(data.number_of_changes_in_y_profile(fs=fs), 3),
            'number_of_changes_in_azimuth': round(data.number_of_changes_in_azimuth(fs=fs), 3),
            'number_of_changes_in_tilt': round(data.number_of_changes_in_tilt(fs=fs), 3),
            'number_of_changes_in_pressure': round(data.number_of_changes_in_pressure(fs=fs), 3),
            'number_of_changes_in_velocity_profile': round(
                data.number_of_changes_in_velocity_profile(fs=fs), 3),
            'relative_number_of_changes_in_x_profile': round(
                data.relative_number_of_changes_in_x_profile(fs=fs), 3),
            'relative_number_of_changes_in_y_profile': round(
                data.relative_number_of_changes_in_y_profile(fs=fs), 3),
            'relative_number_of_changes_in_azimuth': round(
                data.relative_number_of_changes_in_azimuth(fs=fs), 3),
            'relative_number_of_changes_in_tilt': round(
                data.relative_number_of_changes_in_tilt(fs=fs), 3),
            'relative_number_of_changes_in_pressure': round(
                data.relative_number_of_changes_in_pressure(fs=fs), 3),
            'relative_number_of_changes_in_velocity_profile': round(
                data.relative_number_of_changes_in_velocity_profile(fs=fs), 3)
            }


def get_global_spatial_features(data: HandwritingFeatures) -> dict:
    """
    Get the global spatial features from the HandwritingFeatures library
    -------------------------------------------------------------
    g) NISI = Number of Intra Stroke Intersections
    h) RNISI = Relative Number of Intra Stroke Intersections
    i) TNISI = Total Number of Intra Stroke Intersections
    j) RTNISI = Relative Total Number of Intra Stroke Intersections
    k) NOSI = Number of Inter Stroke Intersections
    l) RNOSI = Relative Number of Inter Stroke Intersections
    -------------------------------------------------------------
    m) vertical peaks indices
    n) vertical valleys indices
    o) vertical peaks values
    p) vertical valleys values
    q) vertical peaks velocity
    r) vertical valleys velocity
    s) vertical peaks distance
    t) vertical valleys distance
    u) vertical peaks duration
    v) vertical valleys duration

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
    Returns:
        dict: Dictionary containing the global spatial features of the handwriting
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    # Sampling frequency
    fs = 200

    # h_spatial_features = {'NISI':
    #                           round(data.number_of_intra_stroke_intersections(statistics=['mean'])[0], 3),
    #                       'RNISI':
    #                           round(data.relative_number_of_intra_stroke_intersections(statistics=['mean'])[0], 3),
    #                       'TNISI':
    #                           round(data.total_number_of_intra_stroke_intersections()[0], 3),
    #                       'RTNISI':
    #                           round(data.relative_total_number_of_intra_stroke_intersections()[0], 3),
    #                       'NOSI':
    #                           round(data.number_of_inter_stroke_intersections()[0], 3),
    #                       'RNOSI':
    #                           round(data.relative_number_of_inter_stroke_intersections()[0], 3)}

    spatial_features = {'vertical_peaks_indices_mean': round(
        data.vertical_peaks_indices(fs=fs, statistics=['mean'])[0], 3),
        'vertical_peaks_indices_std': round
        (data.vertical_peaks_indices(fs=fs, statistics=['std'])[0], 3),
        'vertical_valleys_indices_mean': round(
            data.vertical_valleys_indices(fs=fs, statistics=['mean'])[0], 3),
        'vertical_valleys_indices_std': round(
            data.vertical_valleys_indices(fs=fs, statistics=['std'])[0], 3),
        'vertical_peaks_values_mean': round(
            data.vertical_peaks_values(fs=fs, statistics=['mean'])[0], 3),
        'vertical_peaks_values_std': round(
            data.vertical_peaks_values(fs=fs, statistics=['std'])[0], 3),
        'vertical_valleys_values_mean': round(
            data.vertical_valleys_values(fs=fs, statistics=['mean'])[0], 3),
        'vertical_valleys_values_std': round(
            data.vertical_valleys_values(fs=fs, statistics=['std'])[0], 3),
        'vertical_peaks_velocity_mean': round(
            data.vertical_peaks_velocity(fs=fs, statistics=['mean'])[0], 3),
        'vertical_peaks_velocity_std': round(
            data.vertical_peaks_velocity(fs=fs, statistics=['std'])[0], 3),
        'vertical_valleys_velocity_mean': round(
            data.vertical_valleys_velocity(fs=fs, statistics=['mean'])[0], 3),
        'vertical_valleys_velocity_std': round(
            data.vertical_valleys_velocity(fs=fs, statistics=['std'])[0], 3),
        'vertical_peaks_distance_mean': round(
            data.vertical_peaks_distance(fs=fs, statistics=['mean'])[0], 3),
        'vertical_peaks_distance_std': round(
            data.vertical_peaks_distance(fs=fs, statistics=['std'])[0], 3),
        'vertical_valleys_distance_mean': round(
            data.vertical_valleys_distance(fs=fs, statistics=['mean'])[0], 3),
        'vertical_valleys_distance_std': round(
            data.vertical_valleys_distance(fs=fs, statistics=['std'])[0], 3),
        'vertical_peaks_duration_mean': round(
            data.vertical_peaks_duration(fs=fs, statistics=['mean'])[0], 3),
        'vertical_peaks_duration_std': round(
            data.vertical_peaks_duration(fs=fs, statistics=['std'])[0], 3),
        'vertical_valleys_duration_mean': round(
            data.vertical_valleys_duration(fs=fs, statistics=['mean'])[0], 3),
        'vertical_valleys_duration_std': round(
            data.vertical_valleys_duration(fs=fs, statistics=['std'])[0], 3)}

    # return {**h_spatial_features, **spatial_features}
    return spatial_features


def get_global_temporal_features(data: HandwritingFeatures) -> dict:
    """
    Get the global temporal features from the HandwritingFeatures library
    -------------------------------------------------------------
    a) writing duration
    b) ratio stroke duration
    c) ratio writing duration
    d) number of interruptions
    """
    return {'writing_duration': round(data.writing_duration_overall()[0], 3),
            'ratio_stroke_duration_mean': round(data.ratio_of_stroke_durations(statistics=["mean"])[0], 3),
            'ratio_stroke_duration_std': round(data.ratio_of_stroke_durations(statistics=["std"])[0], 3),
            'ratio_writing_duration': round(data.ratio_of_writing_durations()[0], 3),
            'number_of_interruptions': round(data.number_of_interruptions()[0], 3),
            'number_of_interruptions_relative': round(data.number_of_interruptions_relative()[0], 3)
            }
