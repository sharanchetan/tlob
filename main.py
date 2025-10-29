import os
import random
import warnings

import urllib
import zipfile
warnings.filterwarnings("ignore")
import numpy as np
import wandb
import torch
import constants as cst
import hydra
from config.config import Config
from run import run_wandb, run, sweep_init
from preprocessing.lobster import LOBSTERDataBuilder
from preprocessing.btc import BTCDataBuilder
from constants import DatasetType
from preprocessing.rmoney_nse import rmoney_nse_load  # Added for custom use.




@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    print("Using device: ", cst.DEVICE)
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    if config.dataset.type == DatasetType.FI_2010:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    elif config.dataset.type == DatasetType.BTC:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    elif config.dataset.type == DatasetType.LOBSTER:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46
    elif config.dataset.type == DatasetType.RMoney_NSE: ## Added for custom use.
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    
    if config.dataset.type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        # prepare the datasets, this will save train.npy, val.npy and test.npy in the data directory
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    elif config.dataset.type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        try:
            #take the .zip files name in data/FI_2010
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(dir)  # Extracts to the current directory           
            print("Data extracted.")
        except Exception as e:
            raise RuntimeError(f"Error downloading or extracting data: {e}")
        
    elif config.dataset.type == cst.DatasetType.BTC and not config.experiment.is_data_preprocessed:
        data_builder = BTCDataBuilder(
        data_dir=cst.DATA_DIR,
        date_trading_days=config.dataset.dates,
        split_rates=cst.SPLIT_RATES,
        sampling_type=config.dataset.sampling_type,
        sampling_time=config.dataset.sampling_time,
        sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
    
    elif config.dataset.type == cst.DatasetType.RMoney_NSE and not config.experiment.is_data_preprocessed: # Added for custom use.
        try:
            print("Please ensure that the RMoney NSE data is preprocessed and placed in the correct directory.")
            #rmoney_nse_load('/home/chetan/python_tlob_revised/TLOB/data/NSE_Custom/29_august_2025_2885.csv',50,100)
            
            # Add any specific preprocessing steps if required.
        except Exception as e:
            raise RuntimeError(f"Error in preprocessing RMoney NSE data: {e}")

    if config.experiment.is_wandb:
        if config.experiment.is_sweep:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    else:
        run(config, accelerator)
    

def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')
    
if __name__ == "__main__":
    set_torch()
    from hydra import initialize, compose

    with initialize(config_path="config"):
        cfg = compose(
        config_name="config",
        overrides=[
            "+model=tlob",
            "+dataset=rmoney_nse",
            "experiment.is_debug=True",
            "hydra.job.chdir=False"
        ],
    )

    # Directly call your Hydra app with the built config
    hydra_app(cfg)
    # hydra_app()
    
