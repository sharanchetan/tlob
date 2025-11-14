import numpy as np
import pandas as pd
from numpy import ndarray
import torch
from typing import Tuple
import os

def _labeling_custom_5_days(
    X: ndarray,
    len: int,
    h: int):
    """
    Generate labels for the orderbook data based on future mid-price movements.
    Args:
        X (np.ndarray): The orderbook data with shape (num_samples, num_features).
                        It is assumed that the first column is the best bid price
                        and the third column is the best ask price. 
                        THIS IS DIFFERENT FROM THE ORIGINAL IMPLEMENTATION. 
        len (int): The time window smoothing length.
        h (int): The prediction horizon. According to the original paper, h is greater than len.
    """
    # X is the orderbook, in numpy array format, and NOT a pandas DataFrame.
    # len is the time window smoothing length
    # h is the prediction horizon
    assert len > 0, "Length must be greater than 0"
    assert h > 0, "Horizon must be greater than 0"
    
    if h < len: # if h is smaller than len, we can't compute the labels
        len = h
    # Calculate previous and future mid-prices for all relevant indices
    
    """
    Here, previous_ask_price is a numpy array. each array element has number of constituent = len.
    Why are we using sliding_window_view? To get an array with number of elements = len, for each time step. 
    Why is [:h] used? To align the previous prices with future prices. NOO, it's there only because your sliding wingdow wont be able to slide at the bottom, daaa.
    example:
    if X has shape (8828536,40) and sliding window has length 5, then previous_ask_prices will have shape (8828532,5) because firt 4 rows cannot have previous 5 ask prices.
    
    """
    previous_ask_prices_twodim_array = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[:-h]
    previous_bid_prices_twodim_array = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[:-h]
    future_ask_prices_twodim_array = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[h:]
    future_bid_prices_twodim_array = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[h:]

    previous_mid_prices_twodim_array = (previous_ask_prices_twodim_array + previous_bid_prices_twodim_array) / 2
    future_mid_prices_twodim_array = (future_ask_prices_twodim_array + future_bid_prices_twodim_array) / 2

    previous_mid_prices_onedim_array = np.mean(previous_mid_prices_twodim_array, axis=1)
    future_mid_prices_onedim_array = np.mean(future_mid_prices_twodim_array, axis=1)

    # Compute percentage change
    percentage_change_onedim_array = (future_mid_prices_onedim_array - previous_mid_prices_onedim_array) / previous_mid_prices_onedim_array
    
    # alpha is the average percentage change of the stock
    alpha = np.abs(percentage_change_onedim_array).mean() #/ 2
    alpha = max(alpha, 1e-4)  # setting a minimum threshold for alpha
    
    # alpha is the average spread of the stock in percentage of the mid-price
    #alpha = (X[:, 0] - X[:, 2]).mean() / ((X[:, 0] + X[:, 2]) / 2).mean()
        
    print(f"Alpha: {alpha}")
    labels = np.where(percentage_change_onedim_array < -alpha, 2, np.where(percentage_change_onedim_array > alpha, 0, 1))
    print(f"Number of labels: {np.unique(labels, return_counts=True)}")
    print(f"Percentage of labels: {np.unique(labels, return_counts=True)[1] / labels.shape[0]}")
    return labels

def _z_score_normalization_custom(
    df: pd.DataFrame
    ) -> pd.DataFrame:
    """Apply z-score normalization to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Z-score normalized DataFrame.
    """
    z_score_normalised_lob = pd.DataFrame(index=df.index)
    col_list = df.columns

    for col in col_list:
        mean = df[col].mean()
        std =  df[col].std()
        z_score_normalised_lob[col] = (df[col] - mean) / std
    
    return z_score_normalised_lob

def _directory_to_dataframe(directory_path:str)->pd.DataFrame:
    """
    Reads all files from the specified directory, filters rows with non-negative 'Spread'.
    FUNCTION ASSUMS THERE ARE CSV FILES IN THE DIRECTORY ONLY.
    
    Input: Directory path containing CSV files.
    
    output: A single concatenated DataFrame with filtered data from all CSV files.
    """
    list_of_csv = os.listdir(directory_path)
    sorted_list_of_csv = sorted(list_of_csv)
    list_of_dfs=[]
    for csv in sorted_list_of_csv:
        df = pd.read_csv(f"/home/jovyan/tlob/data/custom_5_days/{csv}")
        df =  df[df['Spread'] >= 0].reset_index(drop=True)
        if df['Timestamp'].is_monotonic_increasing:
            df = df.drop(columns=["Timestamp","MidPrice","Spread"])
            list_of_dfs.append(df)
        else:
            print(f"DataFrame from {csv} is not sorted by Timestamp.")
    final_df = pd.concat(list_of_dfs, ignore_index=True)
    return final_df


def custom_5_days_load(
    path: str,
    window_length: int,
    horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the RMoney_NSE dataset from a given path.

    Args:
        path (str): Path to the dataset file.
        ONLY FOR CSV FILES

    Returns:
        np.ndarray: Loaded dataset as a NumPy array.
    """
    
    new_lob = _directory_to_dataframe(path)
    labels = _labeling_custom_5_days(new_lob.values, len=window_length, h=horizon)
    aligned_lob = new_lob.iloc[window_length - 1 : -horizon, :]
    z_score_normalized_lob = _z_score_normalization_custom(aligned_lob)
    final_df = z_score_normalized_lob.copy()
    final_df['Label'] = labels
    final_df.reset_index(drop=True, inplace=True)
    
    split = [0.7, 0.15, 0.15]

    number_of_rows = final_df.shape[0]
    train_input = final_df.iloc[:int(split[0] * number_of_rows),:-1]
    train_labels = final_df.iloc[:int(split[0] * number_of_rows),-1]
    
    val_input = final_df.iloc[int(split[0] * number_of_rows):int((split[0] + split[1]) * number_of_rows),:-1]
    val_labels = final_df.iloc[int(split[0] * number_of_rows):int((split[0] + split[1]) * number_of_rows),-1]
    
    test_input = final_df.iloc[int((split[0] + split[1]) * number_of_rows):,:-1]
    test_labels = final_df.iloc[int((split[0] + split[1]) * number_of_rows):,-1]
    
    train_input = torch.from_numpy(train_input.values).float()
    train_labels = torch.from_numpy(train_labels.values).long()
    
    val_input = torch.from_numpy(val_input.values).float()
    val_labels = torch.from_numpy(val_labels.values).long()
    
    test_input = torch.from_numpy(test_input.values).float()
    test_labels = torch.from_numpy(test_labels.values).long()    
    
    return train_input, train_labels, val_input, val_labels, test_input, test_labels
    