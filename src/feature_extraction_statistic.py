import warnings
import numpy as np
import pandas as pd
from handwriting_features.features import HandwritingFeatures
from handwriting_features.interface.featurizer import FeatureExtractor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Union, Optional

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
    if task_dataframe.empty or 'pen_status' not in task_dataframe.columns or task_dataframe['pen_status'].sum() == 0:
        if verbose:
            print("[-] Input DataFrame is empty or contains no on-paper data. Skipping feature extraction.")
        # Return a dictionary with empty but valid structure
        return {'features': np.array([[]]), 'labels': []}

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

    # Pre-process the dataframe to handle negative values
    task_dataframe_processed = task_dataframe.copy()

    # Shift x and y coordinates to ensure they are non-negative
    if 'x' in task_dataframe_processed.columns:
        x_min = task_dataframe_processed['x'].min()
        if x_min < 0:
            task_dataframe_processed['x'] = task_dataframe_processed['x'] - x_min

    if 'y' in task_dataframe_processed.columns:
        y_min = task_dataframe_processed['y'].min()
        if y_min < 0:
            task_dataframe_processed['y'] = task_dataframe_processed['y'] - y_min

    # Avoid printing
    with suppress_stdout_stderr():
        # Create HandwritingSample with validation disabled to avoid negative value error
        handwriting_df = HandwritingSample.from_pandas_dataframe(task_dataframe_processed, validate=False)

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
        custom_features['number_of_changes_in_x_profile'],
        custom_features['number_of_changes_in_y_profile'],
        custom_features['number_of_changes_in_pressure_profile']
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
            'number_of_changes_in_x_profile',
            'number_of_changes_in_y_profile',
            'number_of_changes_in_pressure_profile'
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
                'number_of_changes_in_x_profile',
                'number_of_changes_in_y_profile',
                'number_of_changes_in_pressure_profile'
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


def calculate_number_of_changes_in_x_profile(
        x_coordinates: np.ndarray,
        pen_status: Optional[np.ndarray] = None,
        adaptive_threshold: bool = True,
        min_threshold: float = 0.1
) -> int:
    """
    Calculate the number of changes in x-direction profile using robust methods.

    Uses multiple approaches:
    1. Adaptive threshold based on data statistics
    2. Zero-crossing detection on smoothed velocity
    3. Peak detection on absolute velocity
    4. Fallback to simple direction changes
    """
    if len(x_coordinates) < 3:
        return 0

    # Filter by pen status if provided
    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        x_coords = x_coordinates[valid_indices]
    else:
        x_coords = x_coordinates

    if len(x_coords) < 3:
        return 0

    # Calculate velocity (first derivative)
    dx = np.diff(x_coords)

    if len(dx) < 2:
        return 0

    # Method 1: Adaptive threshold approach
    if adaptive_threshold:
        # Calculate adaptive threshold based on data statistics
        dx_std = np.std(dx)
        dx_mad = np.median(np.abs(dx - np.median(dx)))  # Median Absolute Deviation

        # Use the more robust measure
        if dx_mad > 0:
            threshold = max(min_threshold, dx_mad * 1.4826)  # 1.4826 makes MAD comparable to std
        else:
            threshold = max(min_threshold, dx_std * 0.5)
    else:
        threshold = min_threshold

    # Method 2: Zero-crossing detection on smoothed velocity
    if len(dx) >= 5:
        # Apply simple moving average smoothing
        window_size = min(5, len(dx) // 3)
        if window_size >= 3:
            smoothed_dx = np.convolve(dx, np.ones(window_size) / window_size, mode='valid')

            # Find zero crossings (sign changes)
            zero_crossings = 0
            for i in range(1, len(smoothed_dx)):
                if smoothed_dx[i - 1] * smoothed_dx[i] < 0:  # Sign change
                    if abs(smoothed_dx[i - 1]) > threshold or abs(smoothed_dx[i]) > threshold:
                        zero_crossings += 1

            if zero_crossings > 0:
                return zero_crossings

    # Method 3: Peak detection on absolute velocity
    abs_dx = np.abs(dx)
    if len(abs_dx) >= 3:
        # Find peaks in absolute velocity (indicating direction changes)
        peaks = []
        for i in range(1, len(abs_dx) - 1):
            if (abs_dx[i] > abs_dx[i - 1] and abs_dx[i] > abs_dx[i + 1] and
                    abs_dx[i] > threshold):
                peaks.append(i)

        if len(peaks) > 0:
            return len(peaks)

    # Method 4: Fallback - simple direction changes with adaptive threshold
    significant_movement = np.abs(dx) > threshold
    dx_filtered = dx[significant_movement]

    if len(dx_filtered) < 2:
        # Further fallback - use even smaller threshold
        threshold_fallback = max(min_threshold * 0.1, np.percentile(np.abs(dx), 25))
        significant_movement = np.abs(dx) > threshold_fallback
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
        adaptive_threshold: bool = True,
        min_threshold: float = 0.1
) -> int:
    """
    Calculate the number of changes in y-direction profile using robust methods.

    Uses multiple approaches:
    1. Adaptive threshold based on data statistics
    2. Zero-crossing detection on smoothed velocity
    3. Peak detection on absolute velocity
    4. Fallback to simple direction changes
    """
    if len(y_coordinates) < 3:
        return 0

    # Filter by pen status if provided
    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        y_coords = y_coordinates[valid_indices]
    else:
        y_coords = y_coordinates

    if len(y_coords) < 3:
        return 0

    # Calculate velocity (first derivative)
    dy = np.diff(y_coords)

    if len(dy) < 2:
        return 0

    # Method 1: Adaptive threshold approach
    if adaptive_threshold:
        # Calculate adaptive threshold based on data statistics
        dy_std = np.std(dy)
        dy_mad = np.median(np.abs(dy - np.median(dy)))  # Median Absolute Deviation

        # Use the more robust measure
        if dy_mad > 0:
            threshold = max(min_threshold, dy_mad * 1.4826)  # 1.4826 makes MAD comparable to std
        else:
            threshold = max(min_threshold, dy_std * 0.5)
    else:
        threshold = min_threshold

    # Method 2: Zero-crossing detection on smoothed velocity
    if len(dy) >= 5:
        # Apply simple moving average smoothing
        window_size = min(5, len(dy) // 3)
        if window_size >= 3:
            smoothed_dy = np.convolve(dy, np.ones(window_size) / window_size, mode='valid')

            # Find zero crossings (sign changes)
            zero_crossings = 0
            for i in range(1, len(smoothed_dy)):
                if smoothed_dy[i - 1] * smoothed_dy[i] < 0:  # Sign change
                    if abs(smoothed_dy[i - 1]) > threshold or abs(smoothed_dy[i]) > threshold:
                        zero_crossings += 1

            if zero_crossings > 0:
                return zero_crossings

    # Method 3: Peak detection on absolute velocity
    abs_dy = np.abs(dy)
    if len(abs_dy) >= 3:
        # Find peaks in absolute velocity (indicating direction changes)
        peaks = []
        for i in range(1, len(abs_dy) - 1):
            if (abs_dy[i] > abs_dy[i - 1] and abs_dy[i] > abs_dy[i + 1] and
                    abs_dy[i] > threshold):
                peaks.append(i)

        if len(peaks) > 0:
            return len(peaks)

    # Method 4: Fallback - simple direction changes with adaptive threshold
    significant_movement = np.abs(dy) > threshold
    dy_filtered = dy[significant_movement]

    if len(dy_filtered) < 2:
        # Further fallback - use even smaller threshold
        threshold_fallback = max(min_threshold * 0.1, np.percentile(np.abs(dy), 25))
        significant_movement = np.abs(dy) > threshold_fallback
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
        method: str = 'adaptive_peaks',
        smoothing_window: int = 3,
        min_pressure_change: float = 0.05
) -> int:
    """
    Calculate the number of significant changes in pressure profile using robust methods.

    Methods available:
    - 'adaptive_peaks': Adaptive peak detection with multiple fallbacks
    - 'derivative_analysis': Analysis of pressure derivative patterns
    - 'quantile_transitions': Transitions between pressure levels
    - 'combined': Uses all methods and returns the maximum
    """
    if len(pressure_values) < 3:
        return 0

    # Filter by pen status if provided
    if pen_status is not None:
        valid_indices = pen_status == True
        if not np.any(valid_indices):
            return 0
        pressure = pressure_values[valid_indices]
    else:
        pressure = pressure_values

    if len(pressure) < 3:
        return 0

    # Handle constant pressure case
    if np.all(pressure == pressure[0]):
        return 0

    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(pressure, [5, 95])
    if p_max == p_min:
        p_min, p_max = np.min(pressure), np.max(pressure)

    if p_max == p_min:
        return 0

    pressure_normalized = np.clip((pressure - p_min) / (p_max - p_min), 0, 1)

    # Adaptive smoothing based on data length
    if len(pressure_normalized) > 10:
        smoothing_window = min(smoothing_window, len(pressure_normalized) // 5)
    else:
        smoothing_window = 1

    if smoothing_window > 1:
        # Use Gaussian-like smoothing instead of simple moving average
        kernel = np.exp(-0.5 * np.linspace(-2, 2, smoothing_window) ** 2)
        kernel = kernel / np.sum(kernel)
        pressure_smooth = np.convolve(pressure_normalized, kernel, mode='valid')
    else:
        pressure_smooth = pressure_normalized

    changes = 0

    if method == 'adaptive_peaks' or method == 'combined':
        # Method 1: Adaptive peak detection
        if len(pressure_smooth) >= 3:
            # Calculate adaptive threshold for peak detection
            pressure_std = np.std(pressure_smooth)
            pressure_mad = np.median(np.abs(pressure_smooth - np.median(pressure_smooth)))

            if pressure_mad > 0:
                adaptive_threshold = max(min_pressure_change, pressure_mad * 1.4826 * 0.5)
            else:
                adaptive_threshold = max(min_pressure_change, pressure_std * 0.3)

            # Find peaks and valleys
            peaks_valleys = 0
            for i in range(1, len(pressure_smooth) - 1):
                is_peak = (pressure_smooth[i] > pressure_smooth[i - 1] and
                           pressure_smooth[i] > pressure_smooth[i + 1])
                is_valley = (pressure_smooth[i] < pressure_smooth[i - 1] and
                             pressure_smooth[i] < pressure_smooth[i + 1])

                if is_peak or is_valley:
                    # Check if the change is significant
                    local_range = max(pressure_smooth[max(0, i - 2):min(len(pressure_smooth), i + 3)]) - \
                                  min(pressure_smooth[max(0, i - 2):min(len(pressure_smooth), i + 3)])
                    if local_range > adaptive_threshold:
                        peaks_valleys += 1

            changes = max(changes, peaks_valleys)

    if method == 'derivative_analysis' or method == 'combined':
        # Method 2: Pressure derivative analysis
        if len(pressure_smooth) >= 4:
            dp = np.diff(pressure_smooth)

            # Find significant changes in derivative
            dp_threshold = max(min_pressure_change, np.std(dp) * 0.5)

            # Count direction changes in significant derivatives
            significant_dp = dp[np.abs(dp) > dp_threshold]
            if len(significant_dp) >= 2:
                signs = np.sign(significant_dp)
                non_zero_signs = signs[signs != 0]
                if len(non_zero_signs) >= 2:
                    derivative_changes = np.sum(np.diff(non_zero_signs) != 0)
                    changes = max(changes, derivative_changes)

    if method == 'quantile_transitions' or method == 'combined':
        # Method 3: Quantile-based transitions
        if len(pressure_smooth) >= 5:
            # Use more granular quantiles for better detection
            quantiles = np.quantile(pressure_smooth, [0.2, 0.4, 0.6, 0.8])

            # Classify each point into pressure levels
            pressure_levels = np.digitize(pressure_smooth, quantiles)

            # Count level transitions
            level_changes = np.diff(pressure_levels)
            quantile_changes = np.sum(level_changes != 0)
            changes = max(changes, quantile_changes)

    # Fallback method: Simple threshold-based detection
    if changes == 0:
        # Use a very small threshold as last resort
        fallback_threshold = max(min_pressure_change * 0.1, np.std(pressure_smooth) * 0.1)

        # Find any significant variations
        for i in range(1, len(pressure_smooth)):
            if abs(pressure_smooth[i] - pressure_smooth[i - 1]) > fallback_threshold:
                changes += 1

        # Cap the fallback result to be reasonable
        changes = min(changes, len(pressure_smooth) // 2)

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
            'number_of_changes_in_x_profile': 0,
            'number_of_changes_in_y_profile': 0,
            'number_of_changes_in_pressure_profile': 0
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
        'number_of_changes_in_x_profile': calculate_number_of_changes_in_x_profile(
            x_coords, pen_status, adaptive_threshold=True, min_threshold=0.1
        ),
        'number_of_changes_in_y_profile': calculate_number_of_changes_in_y_profile(
            y_coords, pen_status, adaptive_threshold=True, min_threshold=0.1
        ),
        'number_of_changes_in_pressure_profile': calculate_pressure_profile_changes(
            pressure, pen_status, method='combined', min_pressure_change=0.05
        )
    }

    return custom_features