import pandas as pd
import numpy as np
import random
from typing import List
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()

        self.linear = self.FeaturesLinear(field_dims,1)
        self.embedding = torch.nn.Embedding(field_dims, embedding_dim=embed_dim)
        self.fm = self.FM_operation(reduce_sum=True)

        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))

        return out.squeeze(1)

    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores

    # Linear part of the equation
    class FeaturesLinear(torch.nn.Module):

        def __init__(self, field_dims, output_dim=1):
            super().__init__()

            self.emb = torch.nn.Embedding(field_dims, output_dim)
            self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

        def forward(self, x):
            """
            :param x: Long tensor of size ``(batch_size, num_fields)``
            """
            return torch.sum(self.emb(x), dim=1) + self.bias

    # FM part of the equation
    class FM_operation(torch.nn.Module):

        def __init__(self, reduce_sum=True):
            super().__init__()
            self.reduce_sum = reduce_sum

        def forward(self, x):
            """
            :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
            """
            square_of_sum = torch.sum(x, dim=1) ** 2
            sum_of_square = torch.sum(x ** 2, dim=1)
            ix = square_of_sum - sum_of_square
            if self.reduce_sum:
                ix = torch.sum(ix, dim=1, keepdim=True)
            return 0.5 * ix

class AbsolutePopularityModel:
    def __init__(self, num_items, topk):
        self.num_items = num_items
        self.item_popularity = None
        self.topk = topk

    def forward(self):
        pass

    def fit(self, interactions):
        """
        Fit the popularity model using user interactions.
        """
        # Calculate item popularity based on interaction counts
        movieid_column = interactions[:, 1]
        rating_column = interactions[:, 2]
        # Create a boolean mask to identify items with rating equal to 1
        mask = rating_column == 1
        # Filter movieids corresponding to items with rating equal to 1
        rated_movieids = movieid_column[mask]
        # Get unique items with rating equal to 1
        unique_rated_movieids, counts = np.unique(rated_movieids, return_counts=True)
        # Normalize popularity to obtain a probability distribution
        n_interactions = len(interactions)
        self.item_popularity = counts / n_interactions
        ranked_list = []
        for i in range(len(unique_rated_movieids)):
            column1 = unique_rated_movieids[i]
            column2 = self.item_popularity[i]
            ranked_list.append([column1, column2])
        ranked_sorted = sorted(ranked_list, key=lambda x: x[1], reverse=True)
        return ranked_sorted

    def predict(self, ranked_sorted, interactions, userID, topk):
        # Exclude ranked items if user-item interaction = 1
        # Get movieIDs with interaction equal to 1
        movieID_interact_1 = interactions[((interactions[:, 2] == 1) & (interactions[:,0] == userID)), 1]
        # Remove rows from ranked_sorted where movieID is in movieID_interact_1
        ranked_sorted = [row for row in ranked_sorted if row[0] not in movieID_interact_1]
        return ranked_sorted[:topk]

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

        # Define a list to store all the recomendad movies across all the users:
        total_list_of_recomended_movies = []

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

            total_list_of_recomended_movies.append(top_k_movies)

        # Now we want to know in how many cases the positive interaction of the user in the test set appeared in the k-sample:
        hit_ratio = correctly_classified / len(self.test.loc[self.test["label"] == 1])

        total_recommended_items = np.array(total_list_of_recomended_movies).flatten()
        num_unique_items_recommended = len(np.unique(total_recommended_items))

        # Number of all the unique items in test:
        num_total_unique_items = len(self.test["item_id"].unique())

        coverage = num_unique_items_recommended / num_total_unique_items

        print(f"Hit ratio {hit_ratio} popularity model")
        print(f"Coverage {coverage}")

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

class ResidualRecommender(nn.Module):
    def __init__(self, user_count, movie_count, embedding_size_user, embedding_size_movie):
        super().__init__()
        self.user_embedding = nn.Embedding(user_count, embedding_size_user)
        self.movie_embedding = nn.Embedding(movie_count, embedding_size_movie)

        self.fc_u = nn.Linear(embedding_size_user, embedding_size_user)
        self.fc_m = nn.Linear(embedding_size_movie, embedding_size_movie)

        self.fc1 = nn.Linear(embedding_size_user + embedding_size_movie, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.user_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.movie_embedding.weight.data)

    def forward(self, user_id, movie_id):
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)

        user_emb = self.fc_u(user_emb)
        movie_emb = self.fc_m(movie_emb)
        user_emb = self.dropout(user_emb)
        movie_emb = self.dropout(movie_emb)

        x = torch.cat((user_emb, movie_emb), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        residual2 = x  # Residual connection
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual2  # Skip connection
        x = self.relu(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.sigmoid(x)

        return x

class CompactRecommender(nn.Module):
    def __init__(self, user_count, movie_count, embedding_size_user, embedding_size_movie):
        super().__init__()

        self.user_embedding = nn.Embedding(user_count, embedding_size_user)
        self.movie_embedding = nn.Embedding(movie_count, embedding_size_movie)

        self.fc = nn.Sequential(
            nn.Linear(embedding_size_user + embedding_size_movie, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight.data)
        nn.init.xavier_uniform_(self.movie_embedding.weight.data)

    def forward(self, user_id, movie_id):
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)

        combined_embeddings = torch.cat((user_emb, movie_emb), dim=1)

        output = self.fc(combined_embeddings)

        return output