import torch
from typing import Tuple
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from math import sqrt
import random
import torch

def set_seeds() -> None:
    # Set a seed values
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    # Set seed for CPU
    seed = 123
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
     

def generate_hparams(model_name: str, train: pd.DataFrame) -> dict:
    print("Generating hparams...")

    if (model_name == "random")  | (model_name == "popularity"):
        return None
 
    # A good mesure for the emdbeding dim can be the 4rth sqrt of the number of items
    recommended_embeding_dim_movies = int(sqrt(sqrt(len(train["item_id"].unique()))))
    recommended_embeding_dim_users = int(sqrt(sqrt(len(train["user_id"].unique()))))
    
    if model_name == "deep":
        hparams = {
            'batch_size': 500000,
            'num_epochs': 5,
            'lr': 0.025,
            'max_user_id': int(train["user_id"].max()),
            'max_item_id': int(train["item_id"].max()),
            'embeding_dim_users': recommended_embeding_dim_users + 20,
            'embeding_dim_items': recommended_embeding_dim_movies + 20
        }

    if model_name == "residual":
        hparams = {
            'batch_size': 500000,
            'num_epochs': 2,
            'lr': 0.001,
            'max_user_id': int(train["user_id"].max()),
            'max_item_id': int(train["item_id"].max()),
            'embeding_dim_users': recommended_embeding_dim_users,
            'embeding_dim_items': recommended_embeding_dim_movies
        }
    
    if model_name == "compact":
        hparams = {
            'batch_size': 500000,
            'num_epochs': 3,
            'lr': 0.001,
            'max_user_id': int(train["user_id"].max()),
            'max_item_id': int(train["item_id"].max()),
            'embeding_dim_users': 10,
            'embeding_dim_items': 10
        }

    return hparams

def train_epoch(
        train_loader: DataLoader,
        validation_loader: DataLoader,
        network: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.functional,
        scheduler: torch.optim,
        device: torch.device
        ) -> Tuple[float, float]:

        network.train()

        train_loss = []
        validation_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                user_ids = data[:, 0]
                movie_ids = data[:, 1]

                output = network(user_ids, movie_ids)

                loss = criterion(output.squeeze(dim=1).float(), target.float())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.25)
                optimizer.step()
                if batch_idx % 10 == 0:
                  scheduler.step()

                train_loss.append(loss.item())

                print("Batch " + str(batch_idx) + " train loss")
                print(loss.item())

                # Calculate too the loss for the whole validation dataset in each batch:
                # We do a loop but in reality there is just one batch:
                for _, (data_val, target_val) in enumerate(validation_loader):
                        data_val, target_val = data_val.to(device), target_val.to(device)

                        user_ids_val = data_val[:, 0]
                        movie_ids_val = data_val[:, 1]

                        output_val = network(user_ids_val, movie_ids_val)

                        loss_val = criterion(output_val.squeeze(dim=1).float(), target_val.float())

                        validation_loss.append(loss_val.item())

                        print("Batch " + str(batch_idx) + " validation loss")
                        print(loss_val.item())


        return np.mean(train_loss), np.mean(validation_loss), train_loss, validation_loss

def train_function(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, lr, device, gamma=0.95, scheduler_step_size=1, num_epochs=3) -> Tuple[list]:
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

  criterion = nn.BCELoss()
  train_loss_list = []
  validation_loss_list = []

  for epoch in range(num_epochs):

      train_loss_mean, validation_loss_mean, train_loss, validation_loss = train_epoch(train_loader, validation_loader, model, optimizer, criterion, scheduler, device)
      print("Epoch mean train loss: " + str(train_loss_mean))
      print("Epoch mean validation loss: " + str(validation_loss_mean))

      train_loss_list = train_loss_list + train_loss
      validation_loss_list = validation_loss_list + validation_loss

  return (train_loss_list, validation_loss_list)

def plot_loss(train_loss_list: list, validation_loss_list: list) -> None:

    # Sample data
    x = list(range(len(train_loss_list)))

    # Create the line plot
    plt.plot(x, train_loss_list, label='Line 1')
    plt.plot(x, validation_loss_list, label='Line 2')

    # Add labels and title
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Loss Plot')

    # Add a legend
    plt.legend(['Train Loss', 'Validation Loss'])

    # Display the plot
    plt.show()