import torch
import pandas as pd
import argparse
from typing import List, Tuple
from models import RandomRecommender, PopularityModel, DeepRecommender, ResidualRecommender, CompactRecommender
from preprocessing import NetflixData
from train_functions import generate_hparams, train_function, plot_loss, set_seeds
from test_functions  import test_function, generate_test_df
import warnings
import numpy as np
import random

# Suppress the warning messages that appear when showing the loss functions during training:
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--m", "--model", help="Input the name of the model that you want to train and validate. Possible models: ['random', 'popularity', 'deep', 'residual', 'compact']")

args = parser.parse_args()
model_name = args.m

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def read_data() -> Tuple[pd.DataFrame]:

    train = pd.read_csv("data/train_new.csv", index_col=0)
    validation = pd.read_csv("data/validation_new.csv", index_col=0)
    test = pd.read_csv("data/test_new.csv", index_col=0)

    return (train, validation, test)

def train_model(model_name: str) -> None:

    print("Reading data...")
    train, validation, test = read_data()
    print("Data successfully read")

    hparams = generate_hparams(model_name, train)

    print("Generating data loaders...")
    netflix_data = NetflixData(for_preprocessing=False)
    netflix_data.train = train
    netflix_data.validation = validation
    netflix_data.test = test
    
    netflix_data.generate_loaders(hparams)
    print("Data loaders successfully generated")

    if model_name == "random":
        model = RandomRecommender(netflix_data.test)
        model.predict()

    if model_name == "popularity":
        model = PopularityModel(netflix_data.train, netflix_data.test)
        model.predict()

    if model_name == "deep":

        model = DeepRecommender(hparams["max_user_id"]+1, hparams["max_item_id"]+1, hparams["embeding_dim_users"], hparams["embeding_dim_items"])
        model = model.to(device)

        train_loss_list, validation_loss_list = train_function(
            model = model, 
            train_loader = netflix_data.train_loader,
            validation_loader = netflix_data.validation_loader,
            lr = hparams["lr"], 
            num_epochs = hparams["num_epochs"], 
            device = device
        )

        plot_loss(train_loss_list, validation_loss_list)

        print("Generating test metrics...")
        # Generate the predictions and store them in a DataFrame:
        test_df = generate_test_df(model, netflix_data.test_loader, device)

        hit_ratio, ndcg = test_function(test_df)
        print("Hit ratio:")
        print(hit_ratio)
        print("NDCG:")
        print(ndcg)
    
    if model_name == "residual":

        model = ResidualRecommender(hparams["max_user_id"]+1, hparams["max_item_id"]+1, hparams["embeding_dim_users"], hparams["embeding_dim_items"])
        model = model.to(device)

        train_loss_list, validation_loss_list = train_function(
            model = model, 
            train_loader = netflix_data.train_loader,
            validation_loader = netflix_data.validation_loader,
            lr = hparams["lr"],
            num_epochs = hparams["num_epochs"],
            device = device
        )

        plot_loss(train_loss_list, validation_loss_list)
        
        print("Generating test metrics...")
        # Generate the predictions and store them in a DataFrame:
        test_df = generate_test_df(model, netflix_data.test_loader, device)

        hit_ratio, ndcg = test_function(test_df)
        print("Hit ratio:")
        print(hit_ratio)
        print("NDCG:")
        print(ndcg)

    if model_name == "compact":

        model = CompactRecommender(hparams["max_user_id"]+1, hparams["max_item_id"]+1, hparams["embeding_dim_users"], hparams["embeding_dim_items"])
        model = model.to(device)

        train_loss_list, validation_loss_list = train_function(
            model = model, 
            train_loader = netflix_data.train_loader,
            validation_loader = netflix_data.validation_loader,
            lr = hparams["lr"],
            num_epochs = hparams["num_epochs"],
            device = device
        )

        plot_loss(train_loss_list, validation_loss_list)
        
        print("Generating test metrics...")
        # Generate the predictions and store them in a DataFrame:
        test_df = generate_test_df(model, netflix_data.test_loader, device)

        hit_ratio, ndcg = test_function(test_df)
        print("Hit ratio:")
        print(hit_ratio)
        print("NDCG:")
        print(ndcg)

if __name__ == "__main__":
    set_seeds()
    train_model(model_name)
