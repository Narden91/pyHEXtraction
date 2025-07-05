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
    """
    Compute the statistical feature extraction for the task.

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
                     "pressure": [0, 4096]}}

    # Avoid printing
    with suppress_stdout_stderr():
        handwriting_df = HandwritingSample.from_pandas_dataframe(task_dataframe)

        # Transform the data to mm
        handwriting_df.transform_axis_to_mm(conversion_type=HandwritingSample.transformer.LPI,
                                            lpi_value=meta_data["lpi"],
                                            shift_to_zero=True)

        # Transform the azimuth and tilt to degrees
        handwriting_df.transform_angle_to_degree(angle=HandwritingSample.TILT)
        handwriting_df.transform_angle_to_degree(angle=HandwritingSample.AZIMUTH)

        sample_values = np.expand_dims(handwriting_df.data_numpy_array, 0)
        sample_labels = None

    # Get the features pipeline (removed composite_features_pipeline)
    fs_kin = kinematic_features_pipeline()
    fs_din = dynamic_features_pipeline()
    fs_spa = spatial_features_pipeline()
    fs_tem = temporal_features_pipeline()
    # fs_com = composite_features_pipeline()

    features_pipeline = features_pipeline + fs_kin + fs_din + fs_spa + fs_tem #+ fs_com

    extractor_configuration = {"fs": fs, "logging_settings": {"soft_validation": True}}

    with suppress_stdout_stderr():
        extractor = FeatureExtractor(sample_values, sample_labels, **extractor_configuration)
        extracted_features = extractor.extract(features_pipeline)

    # Extract custom handwriting features
    custom_features = extract_custom_handwriting_features(sample_values)

    # Add custom features to the extracted features
    # Convert custom features to numpy array format to match the existing structure
    custom_features_array = np.array([[
        custom_features['custom_x_profile_changes'],
        custom_features['custom_y_profile_changes'],
        custom_features['custom_pressure_profile_changes']
    ]])

    # Combine original features with custom features
    if 'features' in extracted_features:
        # Concatenate along the feature dimension (axis=1)
        combined_features = np.concatenate([
            extracted_features['features'],
            custom_features_array
        ], axis=1)

        # Update the extracted_features dictionary
        extracted_features['features'] = combined_features

        # Create or update the labels (feature names)
        custom_feature_names = [
            'custom_x_profile_changes',
            'custom_y_profile_changes',
            'custom_pressure_profile_changes'
        ]

        # Check if labels exist in the extracted_features
        if 'labels' in extracted_features:
            # If labels exist, concatenate with custom feature names
            if isinstance(extracted_features['labels'], np.ndarray):
                extracted_features['labels'] = np.concatenate([
                    extracted_features['labels'],
                    custom_feature_names
                ])
            else:
                # If labels is a list, convert to list and extend
                extracted_features['labels'] = list(extracted_features['labels']) + custom_feature_names
        else:
            num_original_features = extracted_features['features'].shape[1] - len(custom_feature_names)
            original_feature_names = [f'feature_{i}' for i in range(num_original_features)]
            extracted_features['labels'] = original_feature_names + custom_feature_names
    else:
        # If no features exist, create new structure
        extracted_features = {
            'features': custom_features_array,
            'labels': [
                'custom_x_profile_changes',
                'custom_y_profile_changes',
                'custom_pressure_profile_changes'
            ]
        }

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
        },
        {
            'name': 'number_of_changes_in_velocity_profile',
            'args': {
            }
        }
    ]

    return features_pipeline


import numpy as np
import pandas as pd
from typing import Union, Optional


def calculate_number_of_changes_in_x_profile(
        x_coordinates: np.ndarray,
        pen_status: Optional[np.ndarray] = None,
        threshold: float = 1.0
) -> int:
    """Calculate the number of changes in x-direction profile."""
    if len(x_coordinates) < 3:
        return 0

    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        x_coords = x_coordinates[valid_indices]
    else:
        x_coords = x_coordinates

    if len(x_coords) < 3:
        return 0

    dx = np.diff(x_coords)
    significant_movement = np.abs(dx) > threshold
    dx_filtered = dx[significant_movement]

    if len(dx_filtered) < 2:
        return 0

    signs = np.sign(dx_filtered)
    non_zero_signs = signs[signs != 0]
    if len(non_zero_signs) < 2:
        return 0

    sign_changes = np.sum(np.diff(non_zero_signs) != 0)
    return int(sign_changes)


def calculate_number_of_changes_in_y_profile(
        y_coordinates: np.ndarray,
        pen_status: Optional[np.ndarray] = None,
        threshold: float = 1.0
) -> int:
    """Calculate the number of changes in y-direction profile."""
    if len(y_coordinates) < 3:
        return 0

    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        y_coords = y_coordinates[valid_indices]
    else:
        y_coords = y_coordinates

    if len(y_coords) < 3:
        return 0

    dy = np.diff(y_coords)
    significant_movement = np.abs(dy) > threshold
    dy_filtered = dy[significant_movement]

    if len(dy_filtered) < 2:
        return 0

    signs = np.sign(dy_filtered)
    non_zero_signs = signs[signs != 0]
    if len(non_zero_signs) < 2:
        return 0

    sign_changes = np.sum(np.diff(non_zero_signs) != 0)
    return int(sign_changes)


def calculate_pressure_profile_changes(
        pressure_values: np.ndarray,
        pen_status: Optional[np.ndarray] = None,
        method: str = 'peaks_valleys',
        smoothing_window: int = 3,
        min_pressure_change: float = 0.1
) -> int:
    """Calculate the number of significant changes in pressure profile."""
    if len(pressure_values) < 3:
        return 0

    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        pressure = pressure_values[valid_indices]
    else:
        pressure = pressure_values

    if len(pressure) < 3:
        return 0

    pressure_min, pressure_max = np.min(pressure), np.max(pressure)
    if pressure_max == pressure_min:
        return 0

    pressure_normalized = (pressure - pressure_min) / (pressure_max - pressure_min)

    if smoothing_window > 1 and len(pressure_normalized) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        pressure_smooth = np.convolve(pressure_normalized, kernel, mode='valid')
    else:
        pressure_smooth = pressure_normalized

    if method == 'peaks_valleys':
        changes = 0
        for i in range(1, len(pressure_smooth) - 1):
            if ((pressure_smooth[i] > pressure_smooth[i - 1] and
                 pressure_smooth[i] > pressure_smooth[i + 1]) or
                    (pressure_smooth[i] < pressure_smooth[i - 1] and
                     pressure_smooth[i] < pressure_smooth[i + 1])):
                local_range = max(pressure_smooth[i - 1:i + 2]) - min(pressure_smooth[i - 1:i + 2])
                if local_range > min_pressure_change:
                    changes += 1
    else:
        # Default to peaks_valleys if other methods not implemented
        changes = 0

    return int(changes)


def extract_custom_handwriting_features(sample_values: np.ndarray) -> dict:
    """
    Extract custom handwriting features from sample values.

    Args:
        sample_values: 3D array with shape (1, n_points, n_features)
                      where features are [x, y, time, pen_status, azimuth, tilt, pressure]

    Returns:
        Dictionary with custom features
    """
    if sample_values.shape[0] == 0 or sample_values.shape[1] == 0:
        return {
            'custom_x_profile_changes': 0,
            'custom_y_profile_changes': 0,
            'custom_pressure_profile_changes': 0
        }

    # Extract the sample (remove batch dimension)
    sample = sample_values[0]  # Shape: (n_points, 7)

    # Extract individual arrays
    x_coords = sample[:, 0]
    y_coords = sample[:, 1]
    pen_status = sample[:, 3].astype(bool)
    pressure = sample[:, 6]

    # Calculate custom features
    custom_features = {
        'custom_x_profile_changes': calculate_number_of_changes_in_x_profile(
            x_coords, pen_status, threshold=1.0
        ),
        'custom_y_profile_changes': calculate_number_of_changes_in_y_profile(
            y_coords, pen_status, threshold=1.0
        ),
        'custom_pressure_profile_changes': calculate_pressure_profile_changes(
            pressure, pen_status, method='peaks_valleys', min_pressure_change=0.1
        )
    }

    return custom_features