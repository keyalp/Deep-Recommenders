import torch
import pandas as pd
import argparse
from typing import List, Tuple
from models import RandomRecommender, PopularityModel, DeepRecommender
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument("--m", "--model", help="Input the name of the model that you want to train and validate. Possible models: ['random', 'popularity', 'deep', 'residual', 'compact']")

args = parser.parse_args()
model_name = args.m

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def generate_hparams(train: pd.DataFrame) -> dict:
    print("Generating hparams...")

    recommended_embeding_dim_movies = int(sqrt(sqrt(len(train["item_id"].unique()))))
    recommended_embeding_dim_users = int(sqrt(sqrt(len(train["user_id"].unique()))))
    
    hparams = {
        'batch_size': 200000,
        'num_epochs': 3,
        'lr': 0.02,
        'max_user_id': int(train["user_id"].max()),
        'max_item_id': int(train["item_id"].max()),
        'embeding_dim_users': recommended_embeding_dim_users + 20,
        'embeding_dim_items': recommended_embeding_dim_movies + 20
    }

    return hparams

def read_data() -> Tuple[pd.DataFrame]:

    train = pd.read_csv("train.csv")
    validation = pd.read_csv("validation.csv")
    test = pd.read_csv("test.csv")

    return (train, validation, test)

def train_model(model_name: str, hparams: dict) -> None:

    print("Reading data ...")
    train, validation, test = read_data()
    print("Data successfully read")

    if model_name == "random":
        model = RandomRecommender(test)
        model.predict()

    if model_name == "popularity":
        model = PopularityModel(train, test)
        model.predict()

    if model_name == "deep":
        model = DeepRecommender(hparams["max_user_id"]+1, hparams["max_item_id"]+1, hparams["embeding_dim_users"] + 1, hparams["embeding_dim_items"] + 1)
        model = model.to(device)



train, validation, test = read_data()
hparams = generate_hparams(train)
train_model(model_name, hparams)
