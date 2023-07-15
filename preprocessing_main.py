import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from typing import Tuple
from datetime import datetime, timedelta
import argparse
import ast

from preprocessing import BasicPreprocessing, NetflixData

parser = argparse.ArgumentParser()
parser.add_argument("--w", "--whole", help="Indicate if you want to do the whole preprocessing or just the last part", default="False")

args = parser.parse_args()
whole_preprocessing = ast.literal_eval(args.w)

if whole_preprocessing:
    basic_pre = BasicPreprocessing()

    # Execute the basic preprocessing:
    data = basic_pre.step_1(basic_pre.data)
    data = basic_pre.step_2(data)
    data = basic_pre.step_3(data)
    train, test = basic_pre.step_4(data)

else:
    print("Reading csv files...")
    train = pd.read_csv("data/Subset1M_traindata.csv", names=["user_id", 'item_id', 'label', 'timestamp'])
    test = pd.read_csv("data/Subset1M_testdata.csv", names=["user_id", 'item_id', 'label', 'timestamp'])
    print("Data read successfully")


netflix_data = NetflixData(train, test)

netflix_data.create_negative_samples()
netflix_data.split()

netflix_data.train.to_csv("data/train_new.csv")
netflix_data.validation.to_csv("data/validation_new.csv")
netflix_data.test.to_csv("data/test_new.csv")

print("Train, validation and test datasets have been successfully stored in the data folder")


