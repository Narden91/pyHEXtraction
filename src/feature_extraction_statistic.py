from pprint import pprint

import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_features.interface.featurizer import FeatureExtractor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

from handwriting_sample import HandwritingSample


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def statistical_feature_extraction(task_dataframe: pd.DataFrame, fs: int = 200) -> dict:
    """ Compute the statistical feature extraction for the task.

    Args:
        task_dataframe (pd.DataFrame): Dataframe containing the data for compute the global features
        fs (int): Sampling frequency of the data in Hz
    Returns:
        feature_dataframe: dataframe containing the features of the global features
    """

    features_pipeline = []

    # Meta data Wacom One 13.3
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

    # Avoid printing
    with suppress_stdout_stderr():
        handwriting_df = HandwritingSample.from_pandas_dataframe(task_dataframe)

        # Transform the data to mm
        handwriting_df.transform_axis_to_mm(conversion_type=HandwritingSample.transformer.LPI,
                                            lpi_value=meta_data["lpi"],
                                            shift_to_zero=False)

        # Transform the azimuth and tilt to degrees
        handwriting_df.transform_angle_to_degree(angle=HandwritingSample.TILT)
        handwriting_df.transform_angle_to_degree(angle=HandwritingSample.AZIMUTH)

        sample_values = np.expand_dims(handwriting_df.data_numpy_array, 0)
        sample_labels = None

    # Get the kinematic features
    fs_kin = kinematic_features_pipeline()

    features_pipeline = features_pipeline + fs_kin

    extractor_configuration = {"fs": fs, "logging_settings": {"soft_validation": True}}

    with suppress_stdout_stderr():
        extractor = FeatureExtractor(sample_values, sample_labels, **extractor_configuration)
        extracted = extractor.extract(features_pipeline)
    print(f"Features shape: {extracted['features'].shape}\n")
    pprint(extracted)

    return


def kinematic_features_pipeline() -> list:
    """ Get the kinematic features pipeline."""
    features_pipeline = [
        {
            'name': 'velocity',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'axis': ['x', 'y', 'xy'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'acceleration',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'axis': ['x', 'y', 'xy'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'jerk',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'axis': ['x', 'y', 'xy'],
                'in_air': [True, False]
            }
        }
    ]

    return features_pipeline


def dynamic_features_pipeline() -> list:
    """ Get the dynamic features pipeline."""
    features_pipeline = [
        {
            'name': 'azimuth',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'tilt',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'pressure',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        }
    ]

    return features_pipeline


def spatial_features_pipeline() -> list:
    """ Get the spatial features pipeline."""
    features_pipeline = [
        {
            'name': 'stroke_length',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'stroke_height',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'stroke_width',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_length',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_height',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_width',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'vertical_peaks_indices',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_valleys_indices',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_peaks_values',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_valleys_values',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_peaks_velocity',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_valleys_velocity',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_peaks_distance',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_valleys_distance',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_peaks_duration',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'vertical_valleys_duration',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        }
    ]

    return features_pipeline


def temporal_features_pipeline() -> list:
    """ Get the temporal features pipeline."""
    features_pipeline = [
        {
            'name': 'stroke_duration',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'ratio_of_stroke_durations',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_duration',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'writing_duration_overall',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'ratio_of_writing_durations',
            'args': {
                'statistics': ['mean', 'std', 'iqr'],
                'in_air': [True, False]
            }
        },
        {
            'name': 'number_of_interruptions',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        },
        {
            'name': 'number_of_interruptions_relative',
            'args': {
                'statistics': ['mean', 'std', 'iqr']
            }
        }
    ]

    return features_pipeline


def composite_features_pipeline() -> list:
    """ Get the composite features pipeline. """

    # TODO: Add other features
    features_pipeline = [
        {
            'name': 'writing_tempo',
            'args': {}
        },
        {
            'name': 'writing_stops',
            'args': {}
        },
        {
            'name': 'number_of_changes_in_x_profile',
            'args': {}
        },
        {
            'name': 'number_of_changes_in_y_profile',
            'args': {}
        }

    ]

    return features_pipeline


# def statistical_feature_extraction(task_dataframe: pd.DataFrame, in_air=False) -> dict:
#     """ Compute the statistical feature extraction for the task.
#
#     Args:
#         task_dataframe (pd.DataFrame): Dataframe containing the data for compute the global features
#         in_air (bool): Boolean to get the in air features
#     Returns:
#         feature_dataframe: dataframe containing the features of the global features
#     """
#
#     # Avoid printing the HandwritingFeatures class output
#     with suppress_stdout_stderr():
#         # Create a HandwritingFeatures object from the task dataframe to get the global features
#         feature_global_data = HandwritingFeatures.from_pandas_dataframe(task_dataframe)
#
#     # Get the kinematic features
#     kinematic_features_dict = get_kinematic_features(feature_global_data, in_air=in_air)
#
#     # Get the dynamic features
#     dynamic_features_dict = get_dynamic_features(feature_global_data, in_air=in_air)
#
#     # Get the spatial features
#     spatial_features_dict = get_spatial_features(feature_global_data, in_air=in_air)
#
#     # Get the temporal features
#     temporal_features_dict = get_temporal_features(feature_global_data, in_air=in_air)
#
#     # Get the composite features
#     composite_features_dict = get_composite_features(feature_global_data, in_air=in_air)
#
#     # Create a dictionary with the features of the task
#     features_dict = {**kinematic_features_dict, **dynamic_features_dict,
#                      **spatial_features_dict, **temporal_features_dict, **composite_features_dict}
#
#     return features_dict


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

    axis = ["x", "y", "xy"]
    statistics = ["mean", "std", "median", "iqr"]
    kinematic_features = {}

    for ax in axis:
        for feature in ["velocity", "acceleration", "jerk"]:
            for stat in statistics:
                kinematic_features[f"{ax}_{feature}_{stat}"] = getattr(data, feature)(axis=ax, in_air=in_air,
                                                                                      statistics=[stat])[0]

    return kinematic_features


def get_dynamic_features(data: HandwritingFeatures, in_air=False) -> dict:
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
        in_air (bool): Boolean to get the in air features
    Returns:
        dict: Dictionary of the dynamic features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    feat = ["azimuth", "tilt", "pressure"]
    statistics = ["mean", "std", "median", "iqr"]
    dynamic_feature_dict = {}

    for feature in feat:
        for stat in statistics:
            if feature == "pressure":
                dynamic_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(statistics=[stat])[0]
            else:
                dynamic_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(in_air=in_air, statistics=[stat])[0]

    return dynamic_feature_dict


def get_spatial_features(data: HandwritingFeatures, in_air=False) -> dict:
    """ Get the spatial features of the strokes.
    a) stroke length
    b) stroke height
    c) stroke width
    d) writing length
    e) writing height
    f) writing width
    g) number of intra-stroke intersections
    h) relative number of intra-stroke intersections
    i) total number of intra-stroke intersections
    j) relative total number of intra-stroke intersections
    k) number of inter-stroke intersections
    l) relative number of inter-stroke intersections
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
        in_air (bool): Boolean to get the in air features
    Returns:
        dict: Dictionary of the spatial features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    feat = ["stroke_length", "stroke_height", "stroke_width", "writing_length", "writing_height", "writing_width",
            "vertical_peaks_indices", "vertical_valleys_indices", "vertical_peaks_values", "vertical_valleys_values",
            "vertical_peaks_velocity", "vertical_valleys_velocity", "vertical_peaks_distance",
            "vertical_valleys_distance", "vertical_peaks_duration", "vertical_valleys_duration"]
    statistics = ["mean", "std", "median", "iqr"]
    fs = 200
    spatial_feature_dict = {}

    for feature in feat:
        for stat in statistics:
            if feature in ["writing_duration", "writing_length", "writing_height", "writing_width"]:
                spatial_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(in_air=in_air)[0]
            elif feature in ["vertical_peaks_indices", "vertical_valleys_indices", "vertical_peaks_values",
                             "vertical_valleys_values", "vertical_peaks_velocity", "vertical_valleys_velocity",
                             "vertical_peaks_distance", "vertical_valleys_distance", "vertical_peaks_duration",
                             "vertical_valleys_duration"]:
                try:
                    spatial_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(fs=fs, n=1, statistics=[stat])
                except IndexError:
                    # print(f"Error while computing {feature}_{stat}")
                    spatial_feature_dict[f"{feature}_{stat}"] = np.nan
            else:
                spatial_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(in_air=in_air, statistics=[stat])[0]

    return spatial_feature_dict


def get_temporal_features(data: HandwritingFeatures, in_air=False) -> dict:
    """ Get the temporal features of the strokes.
    a) stroke duration
    b) ratio of stroke durations (on-surface / in-air strokes)
    c) writing duration
    d) writing duration overall
    e) ratio of writing durations (on-surface / in-air writing)
    f) number of interruptions
    g) number of interruptions_relative

    Args:
        data (HandwritingFeatures): HandwritingFeatures object
        in_air (bool): Boolean to get the in air features
    Returns:
        dict: Dictionary of the temporal features
    """
    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    feat = ["stroke_duration", "ratio_of_stroke_durations", "writing_duration", "writing_duration_overall",
            "ratio_of_writing_durations", "number_of_interruptions", "number_of_interruptions_relative"]
    statistics = ["mean", "std", "median", "iqr"]
    fs = 200
    temporal_feature_dict = {}

    for feature in feat:
        for stat in statistics:
            if feature in ["ratio_of_stroke_durations"]:
                temporal_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(statistics=[stat])[0]
            elif feature in [""]:
                temporal_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(in_air=in_air)[0]
            elif feature in ["writing_duration", "writing_duration_overall", "ratio_of_writing_durations",
                             "number_of_interruptions", "number_of_interruptions_relative"]:
                temporal_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)()[0]
            else:
                temporal_feature_dict[f"{feature}_{stat}"] = getattr(data, feature)(in_air=in_air, statistics=[stat])[0]

    return temporal_feature_dict


def get_composite_features(data: HandwritingFeatures, in_air=False) -> dict:
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
        in_air (bool): Boolean to get the in air features
    Returns:
        dict: Dictionary of the composite features
    """

    if not isinstance(data, HandwritingFeatures):
        raise TypeError("data must be a HandwritingFeatures object")

    feat = ["writing_tempo", "writing_stops"]
    # statistics = ["mean", "std", "median", "iqr"]
    # fs = 200
    composite_feature_dict = {}

    for feature in feat:
        composite_feature_dict = {f"{feature}": len(getattr(data, feature)())}
        # print(f"{feature}: {composite_feature_dict[f'{feature}']}")

    return composite_feature_dict
