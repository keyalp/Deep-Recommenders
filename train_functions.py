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
from statistics import mean
from test_functions import TestFmModel, coverage
from models import FactorizationMachineModel, AbsolutePopularityModel
from tqdm import tqdm

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
        hparams = {
            'batch_size': 500000
        }
    
    if train is not None:
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
            'embeding_dim_users': recommended_embeding_dim_users + 30,
            'embeding_dim_items': recommended_embeding_dim_movies + 30
        }

    if model_name == "residual":
        hparams = {
            'batch_size': 500000,
            'num_epochs': 10,
            'lr': 0.005,
            'max_user_id': int(train["user_id"].max()),
            'max_item_id': int(train["item_id"].max()),
            'embeding_dim_users': recommended_embeding_dim_users + 30,
            'embeding_dim_items': recommended_embeding_dim_movies + 30
        }
    
    if model_name == "compact":
        hparams = {
            'batch_size': 500000,
            'num_epochs': 3,
            'lr': 0.01,
            'max_user_id': int(train["user_id"].max()),
            'max_item_id': int(train["item_id"].max()),
            'embeding_dim_users': 20,
            'embeding_dim_items': 20
        }
    
    if (model_name == "fm") | (model_name == "abs_popularity"):

        print(train)
        hparams = {
            'topk': 10,
            'lr': 0.001, 
            'num_items': len(train["item_id"].unique()),
            'num_users': len(train["user_id"].unique())
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

class TrainFmModel(): 
    def __init__(self, model, optimizer, data_loader, criterion, device, topk, full_dataset, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.topk = topk
        self.full_dataset = full_dataset

    def train_one_epoch(self):
        self.model.train()
        total_loss = []

        for i, (interactions) in enumerate(self.data_loader):
            interactions = interactions.to(self.device)

            targets = interactions[:,2]
            predictions = self.model(interactions[:,:2])

            loss = self.criterion(predictions, targets.float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)

    def do_epochs(self):
        #Start training the model
        # DO EPOCHS NOW
        tb = False
        topk = 10
        for epoch_i in range(20):
            train_loss = self.train_one_epoch()
            hr, ndcg = TestFmModel.testModel(self.model, self.full_dataset, self.device, topk=topk)
            print(f'epoch {epoch_i}:')
            print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')


def run_fm_model(full_dataset, data_loader, hparams, device):
    #Model, loss and optimizer definition
    print("Begining training...")
    model_fm = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model_fm.parameters(), lr=hparams['lr'])
    
    train_model_fm = TrainFmModel(
        model_fm, optimizer, data_loader, criterion, device, hparams['topk'], full_dataset
    )
    train_model_fm.do_epochs()

    ###TEST EVALUATION FM 
    # user_test = full_dataset.test_set[1]
    # out = model_fm.predict(user_test,device)

    # out[:10]
    # values, indices = torch.topk(out, 10)

    # RANKING LIST TO RECOMMEND
    # recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
    # print('Recommended List: ',recommend_list)
    # gt_item = 14966
    # print(gt_item in recommend_list)

    coverage_per_item = 100*coverage(full_dataset.test_set,hparams['num_items'],hparams['topk'],model_fm,device)
    print(f'Coverage: {coverage_per_item:.2f}')
    
    # Check Init performance
    hr, ndcg = TestFmModel.testModel(model_fm, full_dataset, device, topk=hparams['topk'])
    print("initial HR: ", hr)
    print("initial NDCG: ", ndcg)

########## ABOLUTE POPULARITY MODEL ##########
def run_pop_model(full_dataset, data_loader, hparams):
    topk = hparams['topk']
    pop_model = AbsolutePopularityModel(hparams['num_items'],topk)
    user_test_pop = full_dataset.test_set[5][0][0]

    ranked_sorted = pop_model.fit(data_loader.dataset.interactions)
    pop_recommend_list = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, user_test_pop,topk)
    print(pop_recommend_list[:10])

    #usersID = 7795
    usersID = hparams["num_users"]
    items_for_all_users = []

    for i in tqdm(range(usersID)):
        # extract the list of recomendations for each user:
        pop_recommend_list_user = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, i,topk)

        items_for_all_users.append(pop_recommend_list_user)

    flattened_items_for_all_users = np.array(items_for_all_users).flatten()
    num_items_recommended = np.unique(flattened_items_for_all_users)

    coverage_pop = num_items_recommended / hparams["num_items"]
    print(coverage_pop)

    print(len(num_items_recommended) / hparams["num_items"])