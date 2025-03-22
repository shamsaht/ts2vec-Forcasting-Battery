import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from cycler import cycler
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction
from utils_examples import check_features, extract_timeseries_forecast

def mase(actual, forecast, seasonal_diff):
    n = actual.shape[0]
    scaled_errors = np.mean(np.abs(actual - forecast)) / np.mean(np.abs(seasonal_diff))
    return scaled_errors

def smape(actual, forecast):
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    diff = np.abs(actual - forecast) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def validate(model, X, y, device, criterion, batch_size=128):
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_test_loss = 0
    all_targets = []
    all_outputs = []
    seasonal_diff = y[1:] - y[:-1]  # Assuming y is not shuffled for MASE calculation

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.prediction_outputs, targets)
            total_test_loss += loss.item() * inputs.size(0)

            # Flatten tensors for metric calculations
            all_targets.extend(targets.cpu().numpy().reshape(-1))
            all_outputs.extend(outputs.prediction_outputs.cpu().numpy().reshape(-1))

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    mse = mean_squared_error(all_targets, all_outputs)
    rmse = mse ** 0.5
    mae = mean_absolute_error(all_targets, all_outputs)
    mase_val = mase(np.array(all_targets), np.array(all_outputs), seasonal_diff)
    smape_val = smape(np.array(all_targets), np.array(all_outputs))

    return {'loss': avg_test_loss, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MASE': mase_val, 'SMAPE': smape_val}

def train_one_epoch(device, train_loader, model, optimizer, criterion, scheduler):
    model.train()  # Set the model to training mode
    total_train_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(past_values=inputs, future_values=targets)  # feedforward
        loss = criterion(outputs.prediction_outputs, targets)  # loss computation
        loss.backward()  # loss backward to compute gradients
        optimizer.step()  # apply gradients
        scheduler.step()  # adjust learning rate
        total_train_loss += loss.item() * inputs.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    return avg_train_loss

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 15))  # Set the figure size for the plots

    # Plot training and test loss
    plt.subplot(6, 1, 1)  # 5 rows, 1 column, 1st subplot
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_loss'], label='Test Loss', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(6, 1, 2)  # 2nd subplot
    plt.plot(epochs, history['train_MSE'], label='Train MSE', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_MSE'], label='Test MSE', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Plot RMSE
    plt.subplot(6, 1, 3)  # 3rd subplot
    plt.plot(epochs, history['train_RMSE'], label='Train RMSE', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_RMSE'], label='Test RMSE', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    # Plot MAE
    plt.subplot(6, 1, 4)  # 4th subplot
    plt.plot(epochs, history['train_MAE'], label='Train MAE', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_MAE'], label='Test MAE', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot MASE
    plt.subplot(6, 1, 5)  # 5th subplot
    plt.plot(epochs, history['train_MASE'], label='Train MASE', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_MASE'], label='Test MASE', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test MASE')
    plt.xlabel('Epochs')
    plt.ylabel('MASE')
    plt.legend()

    # Plot SMAPE
    plt.subplot(6, 1, 6)  # 6th subplot
    plt.plot(epochs, history['train_SMAPE'], label='Train SMAPE', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, history['test_SMAPE'], label='Test SMAPE', marker='x', linestyle='--', color='red')
    plt.title('Train vs Test SMAPE')
    plt.xlabel('Epochs')
    plt.ylabel('SMAPE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def perform_extended_forecast(model, initial_input, input_window_len, forecast_horizon, total_forecast_steps, device, feature_names, threshold=0.0):
    extended_predictions = np.copy(initial_input)

    # Convert the initial input to tensor and transfer it to the appropriate device
    sample_x_tensor = torch.tensor(initial_input, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        step = 0
        while True:  # Change to a while loop to keep forecasting
            output = model(sample_x_tensor)
            prediction = output.prediction_outputs.squeeze(0).cpu().numpy()

            # Extend the predictions by appending new predictions at the end
            if step == 0:
                extended_predictions = np.concatenate((extended_predictions, prediction), axis=0)
            else:
                extended_predictions = np.concatenate((extended_predictions[:-forecast_horizon], prediction), axis=0)

            # Check the condition for the battery remaining
            battery_remaining_index = feature_names.index('BatteryStatus0_remaining')  # Adjust if necessary
            if extended_predictions[-1, battery_remaining_index] <= threshold:
                break  # Exit the loop if battery remaining is less than or equal to the threshold

            # Update the tensor for the next prediction
            new_input_window = extended_predictions[-input_window_len:]  # Update the input with the most recent data
            sample_x_tensor = torch.tensor(new_input_window, dtype=torch.float32).unsqueeze(0).to(device)

            step += 1  # Increment step

    return extended_predictions


def plot_forecasting(model, missions_to_forecast, device, frequency_hz, history_length_sec, forecast_horizon_sec, feature_names):
    input_window_len = int(frequency_hz * history_length_sec)
    forecast_horizon = int(frequency_hz * forecast_horizon_sec)

    for mission in missions_to_forecast:
        if len(mission) <= input_window_len:
            continue  # Skip missions that are too short to create even one window

        # Initialize with the first input_window_len data points
        initial_input = mission.iloc[:input_window_len].to_numpy()

        # Calculate total forecast steps to cover the remainder of the mission
        total_forecast_steps = (len(mission) - input_window_len) // forecast_horizon
        if (len(mission) - input_window_len) % forecast_horizon != 0:
            total_forecast_steps += 1  # Ensure complete coverage

        # Perform extended forecasting
        extended_predictions = perform_extended_forecast(model, initial_input, input_window_len, forecast_horizon, total_forecast_steps, device, feature_names)

        # Plot settings
        plt.figure(figsize=(12, 2 * initial_input.shape[1]))
        history_time_steps = np.linspace(-history_length_sec, 0, num=input_window_len, endpoint=False)
        actual_forecast_length = len(mission) - input_window_len
        future_time_steps = np.linspace(0, forecast_horizon * total_forecast_steps, num=len(extended_predictions[input_window_len:]), endpoint=False)[:actual_forecast_length]

        for i in range(initial_input.shape[1]):
            plt.subplot(initial_input.shape[1], 1, i + 1)
            plt.plot(history_time_steps, initial_input[:, i], 'gray', label='History')
            plt.plot(future_time_steps, extended_predictions[input_window_len:input_window_len + len(future_time_steps), i], 'b--', label='Predicted Future')
            plt.plot(future_time_steps, mission.iloc[input_window_len:len(future_time_steps) + input_window_len, i].to_numpy(), 'r--', label='True Mission Future', alpha=0.5)
            plt.title(f'Feature: {feature_names[i]}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Value')
            plt.legend()

        plt.tight_layout()
        plt.show()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    history = checkpoint.get('history', None)
    epoch = checkpoint.get('epoch', None)
    return history, epoch

epochs, batch_size, lr, weight_decay, checkpoint_metric = 245, 8, 5e-5, 1e-5, 'MSE'
load_model = False

def main():
    # Set the style to 'classic' and change pallete color
    # plt.style.use('classic')
    plt.rc('axes', prop_cycle=(cycler('color', px.colors.qualitative.G10)))
    rcParams['figure.autolayout'] = True
    rcParams['legend.loc'] = 'lower right'

    # Load SRTA data
    df = pd.read_csv('../forecasting/unified1Hz.csv', parse_dates=['timestamp'], index_col='timestamp')

    no_variance_features, nan_features = check_features(df)

    # Define input columns
    input_columns = [
        #'SensorBaro0_temperature',  # Ambient temperature
        #'SensorBaro1_temperature',  # Ambient temperature
        'BatteryStatus0_current_a',  # Current draw in amperes.
        'BatteryStatus0_voltage_v',  # Battery voltage.
        'BatteryStatus0_discharged_mah',  # Energy discharged in milliamp-hours.
        'BatteryStatus0_remaining',
        #'VehicleStatus_takeoff_time',
        #'BatteryStatus0_state_of_health',  # Overall health of the battery.
        #'BatteryStatus0_average_power',  # Average power usage.
        #'BatteryStatus0_remaining_capacity_wh',  # Compensated battery capacity remaining in watt-hours.
        #'BatteryStatus0_full_charge_capacity_wh',  # Capacity of the battery when fully charged.
        #'BatteryStatus0_voltage_filtered_v',  # Filtered battery voltage.
        #'BatteryStatus0_current_average_a',  # Averaged current draw.
        #'BatteryStatus0_design_capacity',  # Design capacity of the battery.
        #'BatteryStatus0_max_cell_voltage_delta',  # Difference in voltage between the highest and lowest voltage cells.
    ]

    # Define output columns
    output_columns = input_columns
    # output_columns = [
    #     'BatteryStatus0_current_a',
    #     'BatteryStatus0_voltage_v',
    #     'BatteryStatus0_discharged_mah'
    # ]

    # Calculate columns to drop: all columns not in columns_to_keep plus no_variance and nan features
    columns_to_drop = list(set(df.columns.tolist()) - set(input_columns)) + nan_features

    # Generate data through windowing
    frequency_hz = 1  # frequency of the unified.csv (the resampled data)
    history_length_sec = 128  # duration of windows
    forecast_horizon_sec = 30  # duration of windows
    X, y, feature_names, feature_means, feature_stds, entire_mission = extract_timeseries_forecast(df, input_columns, output_columns, frequency_hz, history_length_sec, forecast_horizon_sec,
                                                                                                   columns_to_drop, transpose=True, plot=False)
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, y_train.shape)

    config = PatchTSTConfig(
        num_input_channels=X_train.shape[2],  # number of features per timestep in the input
        context_length=X_train.shape[1],  # number of time steps in each input sequence
        patch_length=24,
        patch_stride=12,
        prediction_length=y_train.shape[1]  # number of time steps you are predicting
    )

    # Define training device
    device = torch.device('cuda')

    # Initialize the model for prediction
    model = PatchTSTForPrediction(config).to(device)

    # Prepare training and test data
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define lr scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos', final_div_factor=1e4)

    if load_model:
        history, epoch = load_checkpoint('model_checkpoint.pth', model, optimizer, scheduler)

    # Transfer model to the appropriate computing device
    model.to(device)

    # Initialize a dictionary to store training and test loss
    history = {'train_loss': [], 'test_loss': [], 'train_MSE': [], 'train_RMSE': [], 'train_MAE': [], 'test_MSE': [], 'test_RMSE': [], 'test_MAE': [], 'train_MASE': [],
               'test_MASE': [],
               'train_SMAPE': [], 'test_SMAPE': []}
    best_test_metric = float('inf')

    # Begin training
    if not load_model:
        for epoch in range(epochs):
            avg_train_loss = train_one_epoch(device, train_loader, model, optimizer, criterion, scheduler)

            history['train_loss'].append(avg_train_loss)

            # Compute metrics on the training data
            train_metrics = validate(model, X_train, y_train, device, criterion)
            history['train_MSE'].append(train_metrics['MSE'])
            history['train_RMSE'].append(train_metrics['RMSE'])
            history['train_MAE'].append(train_metrics['MAE'])
            history['train_MASE'].append(train_metrics['MASE'])
            history['train_SMAPE'].append(train_metrics['SMAPE'])

            # Validate on test data
            test_metrics = validate(model, X_test, y_test, device, criterion)
            history['test_loss'].append(test_metrics['loss'])
            history['test_MSE'].append(test_metrics['MSE'])
            history['test_RMSE'].append(test_metrics['RMSE'])
            history['test_MAE'].append(test_metrics['MAE'])
            history['test_MASE'].append(test_metrics['MASE'])
            history['test_SMAPE'].append(test_metrics['SMAPE'])

            # Save checkpoint in case checkpoint_metric was improved
            saved = False
            if test_metrics[checkpoint_metric] < best_test_metric:
                best_test_metric = test_metrics[checkpoint_metric]
                torch.save({
                    'epoch': epoch,
                    'history': history,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, 'model_checkpoint.pth')
                saved = True

            # Print epoch summary
            current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train/Test Loss: {avg_train_loss:.4f}/{test_metrics['loss']:.4f}, "
                f"Train/Test MSE: {train_metrics['MSE']:.4f}/{test_metrics['MSE']:.4f}, "
                f"Train/Test RMSE: {train_metrics['RMSE']:.4f}/{test_metrics['RMSE']:.4f}, "
                f"Train/Test MAE: {train_metrics['MAE']:.4f}/{test_metrics['MAE']:.4f}, "
                f"Train/Test MASE: {train_metrics['MASE']:.4f}/{test_metrics['MASE']:.4f}, "
                f"Train/Test SMAPE: {train_metrics['SMAPE']:.2f}%/{test_metrics['SMAPE']:.2f}%, "
                f"Current LR: {current_lr}, "
                f"Checkpoint: {saved} "
            )

    history, epoch = load_checkpoint('model_checkpoint.pth', model, optimizer, scheduler)

    # Call the train function to train the model
    plot_metrics(history)

    # Plot Forecasting
    plot_forecasting(model, entire_mission, device, frequency_hz, history_length_sec, forecast_horizon_sec, feature_names)

if __name__ == '__main__':
    main()
