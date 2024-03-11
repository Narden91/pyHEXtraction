import warnings
import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_features.interface.featurizer import FeatureExtractor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

warnings.filterwarnings("ignore")

from handwriting_sample import HandwritingSample


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out


def statistical_feature_extraction(task_dataframe: pd.DataFrame, fs: int = 200, verbose: bool = None) -> dict:
    """ Compute the statistical feature extraction for the task.

    Args:
        task_dataframe (pd.DataFrame): Dataframe containing the data for compute the global features
        fs (int): Sampling frequency of the data in Hz
        verbose (bool): Boolean to print the features shape
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

    # Get the features pipeline
    fs_kin = kinematic_features_pipeline()
    fs_din = dynamic_features_pipeline()
    fs_spa = spatial_features_pipeline()
    fs_tem = temporal_features_pipeline()
    # fs_com = composite_features_pipeline()

    features_pipeline = features_pipeline + fs_kin + fs_din + fs_spa + fs_tem # + fs_com
    extractor_configuration = {"fs": fs, "logging_settings": {"soft_validation": True}}

    with suppress_stdout_stderr():
        extractor = FeatureExtractor(sample_values, sample_labels, **extractor_configuration)
        extracted_features = extractor.extract(features_pipeline)

    print(f"Features shape: {extracted_features['features'].shape}\n") if verbose else None
    # print(f"Features: {extracted_features['features']}\n") if verbose else None

    return extracted_features


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
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_height',
            'args': {
                'in_air': [True, False]
            }
        },
        {
            'name': 'writing_width',
            'args': {
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
            }
        },
        {
            'name': 'writing_duration_overall',
            'args': {
            }
        },
        {
            'name': 'ratio_of_writing_durations',
            'args': {
                'in_air': [True, False]
            }
        },
        {
            'name': 'number_of_interruptions',
            'args': {
            }
        },
        {
            'name': 'number_of_interruptions_relative',
            'args': {
            }
        }
    ]

    return features_pipeline


def composite_features_pipeline() -> list:
    """ Get the composite features pipeline."""
    features_pipeline = [
        {
            'name': 'writing_tempo',
            'args': {
            }
        },
        {
            'name': 'writing_stops',
            'args': {
            }
        },
        {
            'name': 'number_of_changes_in_x_profile',
            'args': {
            }
        },
        {
            'name': 'number_of_changes_in_y_profile',
            'args': {
            }
        },
        {
            'name': 'number_of_changes_in_pressure',
            'args': {
            }
        }
    ]

    return features_pipeline
