import numpy as np
import pandas as pd
from numpy import ndarray
import torch
from typing import Tuple
import os
def _labeling_custom(
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
        h (int): The prediction horizon.
    """
    # X is the orderbook
    # len is the time window smoothing length
    # h is the prediction horizon
    assert len > 0, "Length must be greater than 0"
    assert h > 0, "Horizon must be greater than 0"
    
    if h < len: # if h is smaller than len, we can't compute the labels
        len = h
    # Calculate previous and future mid-prices for all relevant indices
    previous_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[:-h]
    previous_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[:-h]
    future_ask_prices = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len)[h:]
    future_bid_prices = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len)[h:]

    previous_mid_prices = (previous_ask_prices + previous_bid_prices) / 2
    future_mid_prices = (future_ask_prices + future_bid_prices) / 2

    previous_mid_prices = np.mean(previous_mid_prices, axis=1)
    future_mid_prices = np.mean(future_mid_prices, axis=1)

    # Compute percentage change
    percentage_change = (future_mid_prices - previous_mid_prices) / previous_mid_prices
    
    # alpha is the average percentage change of the stock
    alpha = np.abs(percentage_change).mean() / 2
    
    # alpha is the average spread of the stock in percentage of the mid-price
    #alpha = (X[:, 0] - X[:, 2]).mean() / ((X[:, 0] + X[:, 2]) / 2).mean()
        
    #print(f"Alpha: {alpha}")
    labels = np.where(percentage_change < -alpha, 2, np.where(percentage_change > alpha, 0, 1))
    #print(f"Number of labels: {np.unique(labels, return_counts=True)}")
    #print(f"Percentage of labels: {np.unique(labels, return_counts=True)[1] / labels.shape[0]}")
    return labels

def _z_score_normalization(
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


def rmoney_nse_load(
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
    list_of_csv = os.listdir(path)
    sorted_list_of_csv = sorted(list_of_csv) 
    lob = pd.read_csv(path)
    new_lob = lob.drop(columns=['Timestamp', 'MidPrice', 'Spread'])
    labels = _labeling_custom(new_lob.values, len=window_length, h=horizon)
    aligned_lob = new_lob.iloc[window_length - 1 : -horizon, :]
    z_score_normalized_lob = _z_score_normalization(aligned_lob)
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
    