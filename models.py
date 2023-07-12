import pandas as pd
import random
from typing import List
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

class RandomRecommender:

    def __init__(self, test: pd.DataFrame, k: int=10):
        self.test = test
        self.k = k

    def predict(self):

        # For each one of the users we will generate a random recommendation:
        correctly_classified =  0

        for i, (us_id, user_df) in enumerate(self.test.groupby("user_id")):
            # Generate the random sample:
            random_sample = random.sample(list(user_df["item_id"]), self.k)

            # Get the id of the positive movie for user us_id:
            positive_movie_id = user_df.loc[user_df["label"] == 1, "item_id"].iloc[0]

            # Check if the positive item is on the random sample:
            if positive_movie_id in random_sample:
                correctly_classified += 1

            if i % 1000 == 0:
                print(f"User number {i} done!")

            # Now we want to know in how many cases the positive interaction of the user in the test set appeared in the k-sample:
            hit_ratio = correctly_classified / len(self.test.loc[self.test["label"] == 1])
        
        print(f"Hit ratio {hit_ratio} random recommender")

class PopularityModel:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, k: int=10):
        self.train = train
        self.test = test
        self.k = k

        movies_popularity = train.groupby("item_id").count()
        self.movies_popularity = movies_popularity.sort_values(by="user_id", ascending=False).drop("label", axis=1)

    def sort_movies_by_popularity(self, movies: List):

        """
        Gets a list of movie_ids and returns the top 10 most popular in the movies_popularity
        """
        # Get the movies with the amount of views and sort them by popularity:
        sorted_movies = self.movies_popularity.loc[movies].sort_values(by="user_id", ascending=False)

        # Get the top_10
        top_k = list(sorted_movies.iloc[:self.k].index)

        return top_k
    
    def predict(self):
    
        # Initiate the count for correctly classified at 0:
        correctly_classified = 0

        for i, (us_id, user_df) in enumerate(self.test.groupby("user_id")):
            # From all the movies in the test set for the user us_id we want to know how popular they are:
            # For this the first step will be to get a list of the movie_ids in the dataframe user_df:
            list_movie_ids = list(user_df["item_id"].unique())

            top_k_movies = self.sort_movies_by_popularity(list_movie_ids)

            # Get the id of the positive movie for user us_id:
            positive_movie_id = user_df.loc[user_df["label"] == 1, "item_id"].iloc[0]

            # Check if one of the elements of the list top_k_movies is the positive sample of the user:
            if positive_movie_id in top_k_movies:
                correctly_classified += 1

            if i % 1000 == 0:
                print(f"User number {i} done!")

        # Now we want to know in how many cases the positive interaction of the user in the test set appeared in the k-sample:
        hit_ratio = correctly_classified / len(self.test.loc[self.test["label"] == 1])

        print(f"Hit ratio {hit_ratio} popluarity model")

class DeepRecommender(nn.Module):
    def __init__(self, user_count, movie_count, embeding_size_user, embeding_size_movie):

        super().__init__()
        self.user_embeding = nn.Embedding(user_count, embeding_size_user)
        self.movie_embeding = nn.Embedding(movie_count, embeding_size_movie)

        self.fc_u = nn.Linear(embeding_size_user, embeding_size_user + 5)
        self.fc_m = nn.Linear(embeding_size_movie, embeding_size_movie + 5)

        self.fc1 = nn.Linear((embeding_size_user + 5) + (embeding_size_movie + 5), 30)
        self.fc2 = nn.Linear(30, 10)
        self.fc3 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.user_embeding.weight.data)
        torch.nn.init.xavier_uniform_(self.user_embeding.weight.data)

    def forward(self, user_id, movie_id):

        user_id_emb = self.user_embeding(user_id)
        movie_id_emb = self.movie_embeding(movie_id)

        user_id_emb = self.relu(self.fc_u(user_id_emb))
        movie_id_emb = self.relu(self.fc_m(movie_id_emb))

        combined_embedings = torch.cat((user_id_emb, movie_id_emb), dim=1)

        x = self.dropout(combined_embedings)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))

        return x

