import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tsai.all import SlidingWindow


def check_features(df):
    no_variance_features = []
    nan_features = []

    for column in df.columns:
        # Check for columns with no variance
        if df[column].nunique() == 1:
            no_variance_features.append(column)
        # Check for columns with NaN values
        if df[column].isnull().any():
            nan_features.append(column)

    # Print warnings if no variance features are found
    if no_variance_features:
        print("Warning: The following features have no variance:")
        print(no_variance_features)
    else:
        print("All features have variance.")

    # Print warnings if NaN features are found
    if nan_features:
        print("Warning: The following features contain NaN values:")
        print(nan_features)
    else:
        print("No features contain NaN values.")

    return no_variance_features, nan_features


def plot_random_windows(X, y, feature_names, num_windows=3):
    # Check if the data has at least the number of requested windows
    if X.shape[0] < num_windows:
        raise ValueError("The total number of windows in X is less than the requested number of random windows to plot.")

    # Randomly select indices for the windows to plot
    indices = np.random.choice(X.shape[0], size=num_windows, replace=False)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_windows, 1, figsize=(10, num_windows * 3))

    # If there's only one window, axes might not be an array
    if num_windows == 1:
        axes = [axes]  # Make it iterable

    # Plot each selected window
    for i, ax in enumerate(axes):
        window_index = indices[i]
        ax.plot(X[window_index].T)  # Transpose to plot features as separate lines
        ax.set_title(f'Window {window_index}: {y[window_index]}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Feature Value')
        ax.legend(feature_names)

    plt.tight_layout()
    plt.show()


def plot_window(window_data, window_class, feature_names):
    # Verify that window_data has the shape [n_timesteps, n_features]
    if window_data.ndim != 2 or window_data.shape[0] != len(feature_names):
        raise ValueError(f"Expected window_data to have shape [n_features, n_timesteps] with n_features={len(feature_names)}, but got shape {window_data.shape}")

    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(window_data):  # Transpose to iterate over features
        plt.plot(feature, label=feature_names[i])

    plt.title(f"Class: {window_class}")
    plt.xlabel("Time Steps")
    plt.ylabel("Feature Values")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_column_frequencies(df):
    # Ensure the 'timestamp' column is sorted
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamp column must be sorted in ascending order")

    # Initialize a dictionary to store frequencies of each column
    frequency_dict = {}
    overall_mean_frequencies = []

    for column in df.columns:
        # Filter out NaN values
        valid_df = df[[column]].dropna()

        # Skip empty columns
        if valid_df.empty:
            continue

        # Calculate time intervals between valid timestamps
        intervals = valid_df.index.to_series().diff().dt.total_seconds().dropna()

        # Calculate mean interval and frequency
        mean_interval = intervals.mean()
        if mean_interval == 0:
            frequency_dict[column] = float('inf')  # Infinite frequency if intervals are zero
        else:
            frequency_dict[column] = 1 / mean_interval

        # Collect mean frequencies for overall calculation
        overall_mean_frequencies.append(1 / mean_interval)

    # Print frequencies of each column
    for col, freq in frequency_dict.items():
        print(f"Frequency of {col}: {freq} Hz")

    # Identify the column with the highest and lowest frequency
    max_freq_col = max(frequency_dict, key=frequency_dict.get)
    min_freq_col = min(frequency_dict, key=frequency_dict.get)
    print(f"Highest frequency column: {max_freq_col} with {frequency_dict[max_freq_col]} Hz")
    print(f"Lowest frequency column: {min_freq_col} with {frequency_dict[min_freq_col]} Hz")

    # Calculate and print overall mean frequency
    overall_mean_frequency = sum(overall_mean_frequencies) / len(overall_mean_frequencies) if overall_mean_frequencies else 0
    print(f"Overall mean frequency of DataFrame: {overall_mean_frequency} Hz")

    return frequency_dict


def extract_timeseries_classification(df, frequency_hz, window_length_sec, columns_to_drop, transpose=False, plot_result=False):
    window_len = int(frequency_hz * window_length_sec)
    stride = frequency_hz  # Stride by one second, equivalent to the frequency in Hz
    sliding_win = SlidingWindow(
        window_len=window_len,
        stride=stride,
        horizon=0,
        pad_remainder=True,
        padding='post',
        padding_value=0,
        add_padding_feature=False,
        get_y=[],
        seq_first=True
    )

    all_segments = []
    all_labels = []

    for mission_name in df['mission_name'].unique():
        mission_data = df[df['mission_name'] == mission_name].sort_index()

        # Drop 'mission_name' and other specified columns
        mission_data = mission_data.drop(columns=['mission_name'] + columns_to_drop)

        # Apply sliding window using tsai
        X_mission, _ = sliding_win(mission_data.values)

        if transpose:
            X_mission = np.transpose(X_mission, axes=(0, 2, 1))  # Transposing each window if needed

        all_segments.append(X_mission)
        all_labels.extend([mission_name] * X_mission.shape[0])  # Extend labels for each window

    # Concatenate all segments and convert labels to numpy array
    X = np.concatenate(all_segments)
    y = np.array(all_labels, dtype=str)
    feature_names = mission_data.columns.tolist()

    if plot_result:
        if transpose:
            plot_random_windows(np.transpose(X, axes=(0, 2, 1)), y, feature_names, num_windows=3)
        else:
            plot_random_windows(X, y, feature_names, num_windows=3)

    return X, y, feature_names


def plot_mission_data(df_cleaned):
    # Determine the number of subplots needed
    num_variables = len(df_cleaned.columns)
    fig, axes = plt.subplots(nrows=num_variables, ncols=1, figsize=(12, 4 * num_variables))  # Adjust figure size as needed

    # Plot each column in a separate subplot
    for i, column in enumerate(df_cleaned.columns):
        axes[i].plot(df_cleaned.index, df_cleaned[column])
        axes[i].set_title(f'Time Series for {column}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)

    # plt.tight_layout()  # Adjusts plot spacings to avoid overlap
    plt.show()


def extract_timeseries_forecast(df, input_columns, output_columns, frequency_hz, history_length_sec, forecast_horizon_sec, columns_to_drop,
                                normalize=True, transpose=False, plot=False):
    # Compute mean and standard deviation for each input feature
    features = list(set(input_columns + output_columns))
    feature_means = df[features].mean()
    feature_stds = df[features].std()

    if normalize:
        df[features] = (df[features] - feature_means[features]) / feature_stds[features]

    input_window_len = int(frequency_hz * history_length_sec)
    forecast_horizon = int(frequency_hz * forecast_horizon_sec)

    # Configure the SlidingWindow with appropriate horizon for forecasting
    stride = 1
    sliding_win = SlidingWindow(
        window_len=input_window_len,
        horizon=forecast_horizon,
        stride=stride,
        pad_remainder=True,
        padding='pre',
        padding_value=np.nan,
        add_padding_feature=False,
        get_x=input_columns,  # Use column indices for inputs
        get_y=output_columns,  # Use column indices for outputs
        seq_first=True
    )
    X, y, entire_mission = [], [], []
    for mission_name in df['mission_name'].unique():
        print(mission_name)
        df_mission = df[df['mission_name'] == mission_name].sort_index()
        # Continue if mission len is smaller than forecast horizon len
        if len(df_mission) < forecast_horizon:
            continue

        # Drop 'mission_name' and other specified columns
        df_mission = df_mission.drop(columns=['mission_name'] + columns_to_drop)
        if plot:
            plot_mission_data(df_mission)

        # Save full mission data for later visualization
        entire_mission.append(df_mission)

        # Fill each column with values starting from 0 up to the number of rows - 1 (for debugging purpose
        # for column in df_mission.columns:
        #     df_mission[column] = np.arange(len(df_mission)) + 1

        # Manually pre-pad the data
        pad_length = input_window_len - 2  # Number of rows to pad
        padding_df = pd.DataFrame(data=np.nan, index=np.arange(pad_length), columns=df_mission.columns)
        df_mission = pd.concat([padding_df, df_mission], ignore_index=True)

        # Backfill NaN values with the first valid entry in each column
        df_mission.fillna(method='bfill', inplace=True)

        # Apply sliding window using tsai
        X_mission, y_mission = sliding_win(df_mission)

        if transpose:
            X_mission = np.transpose(X_mission, axes=(0, 2, 1))  # Transposing each window if needed
            y_mission = np.transpose(y_mission, axes=(0, 2, 1))  # Transposing each window if needed

        X.append(X_mission)
        y.append(y_mission)

    # Concatenate all segments and convert labels to numpy array
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    feature_names = input_columns

    return X, y, feature_names, feature_means, feature_stds, entire_mission
