import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from typing import Tuple
from datetime import datetime, timedelta

# Set a seed value
random.seed(123)
np.random.seed(123)

class BasicPreprocessing(): 
    
    def __init__(self):

        # Get the dataset movies_sampled which is a sub-sample fo the Netflix dataset
        print("Reading csv data...")
        self.data = pd.read_csv('data/movies_sampled.csv', sep=",", low_memory=False)
        print("Data succesfully read!")

        # Eliminate usless columns:
        self.data = self.data.iloc[:, 2:]

    def step_1(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Executing step 1...")
        # Convert the data to numpy
        data = data.to_numpy()

        # sort columns:
        # change the columns 1label i 3movieid
        column_temp = np.copy(data[:, 1]) #bk label
        data[:, 1] = data[:, 3] # write to col1 movieid
        data[:, 3] = column_temp # write to col3 bklabel
        # change columns label/rating for timestamp
        column_temp = np.copy(data[:, 2]) #bk date
        data[:, 2] = data[:, 3] # write label to col 2
        data[:, 3] = column_temp # write to col3 bk date
        # change the format of the date for timestamp
        # Obtain the column dates
        dates_str = data[:, 3]
        # Convert the dates from strings to integers
        dates_int = [int(datetime.strptime(fecha_str, '%Y-%m-%d').strftime('%Y%m%d')) for fecha_str in dates_str]
        # Replace the column of dates with the date objects
        data[:, 3] = dates_int
        data[:, 0] = data[:, 0].astype(int)
        data[:, 1] = data[:, 1].astype(int)
        data[:, 2] = data[:, 2].astype(int)

        # add a mark for the users that have seen multiple movies the last day
        last_userid = 0
        last_timestamp = 0
        d = 1

        i=0
        for row in data:
            userid = row[0]
            timestamp = row[3]
            if userid == last_userid and timestamp == last_timestamp:
                data[i][3] =  last_timestamp + d
                d = d + 1
            last_userid = userid
            last_timestamp = timestamp
            i +=1

        data_sorted = sorted(data, key=lambda x: (x[0], x[3]))

        df = pd.DataFrame(data_sorted)

        print("Step 1 executed!")

        return df

    def step_2(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Exectuting step 2...")
        data = data.to_numpy() # to numpy array

        # Obtain a random sample:
        np.random.shuffle(data)
        # Obtain 1.2 million interactions:
        data = data[:1200000]

        unique_counts = [len(np.unique(data[:, i])) for i in range(data.shape[1])]

        # Obtain teh users with more than 20 interactions
        user_id, count = np.unique(data[:, 0], return_counts=True)
        users_with_interacciones = user_id[count >= 20]

        data = data[np.isin(data[:, 0], users_with_interacciones)]

        # Filter data to only keep the movies with enough interactions:
        movie_id, count = np.unique(data[:, 1], return_counts=True)
        # Identify movies with at least 5 interactions:
        movies_with_interacciones = movie_id[count >= 5]

        data = data[np.isin(data[:, 1], movies_with_interacciones)]

        data_sorted = sorted(data, key=lambda x: (x[0], x[3]))
        #1M interactions, 2000 users, 2000 movies
        df = pd.DataFrame(data_sorted)

        print("Step 2 executed!")

        return df
    
    def step_3(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Exectuting step 3...")
        data = data.to_numpy() # to numpy array

        # Creamos a dictionary to do the mappinmg of id to consecutive numbers
        user_ids = np.unique(data[:, 0])
        user_id_map = {user_id: index for index, user_id in enumerate(user_ids)}
        # Create the new array of user_ids mapped
        mapped_user_ids = np.array([user_id_map[user_id] for user_id in data[:,0]])
        # Replace the first colums with the new mappings
        data[:,0] = mapped_user_ids
        # Obtain all the unique movie_ids
        unique_movie_ids = np.unique(data[:, 1])
        # Create a dictionary for the mappings of movies:
        movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
        # Map the movie_ids with the 
        mapped_movie_ids = np.array([movie_id_map[movie_id] for movie_id in data[:, 1]])
        # Replace the second column of train_data with the new indexes
        data[:, 1] = mapped_movie_ids

        df = pd.DataFrame(data)

        print("Step 3 executed!")

        return df
    
    def step_4(self, data: pd.DataFrame) -> Tuple[pd.DataFrame]:
        print("Exectuting step 4...")
        data = data.to_numpy()
 
        #### 3. Split the dataset
        # Get the unique user IDs and their corresponding indices
        user_ids, indices = np.unique(data[:, 0], return_index=True)
        #print(len(indices))

        # Find the index of the last occurrence of each user ID
        last_indices = np.concatenate((indices[1:], [len(data)]))

        # Get the row with the most recent timestamp for each user
        test_data = data[last_indices-1]

        train_data = np.delete(data, last_indices-1, axis=0)

        df_test = pd.DataFrame(test_data)
        df_train = pd.DataFrame(train_data)

        # Change the column rating to the label 1 to indicate a positive interaction:
        df_test[2] = 1
        df_train[2] = 1

        # Change the name of the columns:
        df_test.columns = ["user_id", "item_id", "label", "timestamp"]
        df_train.columns = ["user_id", "item_id", "label", "timestamp"]

        print("Step 4 executed!")
        return (df_train, df_test)

class NetflixData():

    def __init__(self, train: pd.DataFrame=None, test: pd.DataFrame=None, for_preprocessing=True):
        
        # The boolean for_preprocessing indicates if we are using the class in the prerprocessing_main.py or in 
        # the main.py. In the second case we will just use the function generate_loaders and we will read from
        # a csv the validation, test and train dataframes

        # if for_preprocessing:
        #     self.train_raw = pd.read_csv("Subset1M_traindata_new.csv", names=["user_id", 'item_id', 'label', 'timestamp'])
        #     self.test_raw = pd.read_csv("Subset1M_testdata_new.csv", names=["user_id", 'item_id', 'label', 'timestamp'])

        if for_preprocessing:
            self.train_raw = train
            self.test_raw = test

        self.negative_samples_train = None
        self.negative_samples_validation = None
        self.negative_samples_test = None

        self.train = None
        self.validation = None
        self.test = None

        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None

    def create_negative_samples(self):

        # Concat the two dataframes to create the negative sampling. We concatenate because the code needs to be aware of all the
        # positive samples that we have in order to generate the negative ones:
        whole_dataset = pd.concat([self.train_raw, self.test_raw])

        # Create artificial negative samples:
        # We will create three negative_samples dataframes. One for training, one for validation and one for test:
        # For the validation and test we will put 99 negatives samples for each one of the users, and the rest we will use it for train
        non_existing_combinations_train_list = []
        non_existing_combinations_validation_list = []
        non_existing_combinations_test_list = []

        unique_movie_ids = whole_dataset["item_id"].unique()
        # For each user id we will create negative samples:

        print("Generating negative samples...")
        for us_id, us_df in whole_dataset.groupby("user_id"):

            all_combinations = set([(u_id, m_id) for u_id in us_df["user_id"].unique() for m_id in unique_movie_ids])
            existing_combinations = set(zip(us_df["user_id"], us_df["item_id"]))
            non_existing_combinations = all_combinations - existing_combinations
            # Get a sample of the non existing combinations for each one of the users:

            non_existing_combinations = list(non_existing_combinations)

            # We have to put this if statement beacuse some of the users have seen a lot of movies and the array non_existing_combinations
            # ends up beeing smaller than 400, so we would get a error

            if 400 <= len(non_existing_combinations):
                non_existing_combinations = random.sample(non_existing_combinations, 400)
            else:
                print(f"User {us_id} has seen to many movies!")
                non_existing_combinations = random.sample(non_existing_combinations, len(us_df))

            # Once you get the non_existing_combiantions sampled convert it to a dataframe ...
            non_existing_combinations = pd.DataFrame(non_existing_combinations, columns=["user_id", "item_id"])
            # ... and store the first 9 to test, the following 9 to validation and the last ones to train
            non_existing_combinations_test_list.append(non_existing_combinations.iloc[:99])
            non_existing_combinations_validation_list.append(non_existing_combinations.iloc[99: 198])
            non_existing_combinations_train_list.append(non_existing_combinations.iloc[198:])

            if len(non_existing_combinations.iloc[:99]) != len(non_existing_combinations.iloc[99: 198]):
                print("Alert fo user" + str(us_id))

        # concat all the mini-dataframes with the negative interactions of each of the users to create the dataframes with all the negative samples:
        self.negative_samples_test = pd.concat(non_existing_combinations_test_list)
        self.negative_samples_validation = pd.concat(non_existing_combinations_validation_list)
        self.negative_samples_train = pd.concat(non_existing_combinations_train_list)

        # Add the column "label" and specify that the user has not seen the movie:
        self.negative_samples_test["label"] = 0
        self.negative_samples_validation["label"] = 0
        self.negative_samples_train["label"] = 0

        print("Number of negative samples generated for:")
        print(f"For train dataset: {len(self.negative_samples_train)}")
        print(f"For validation dataset: {len(self.negative_samples_validation)}")
        print(f"For test dataset: {len(self.negative_samples_test)}")
    
    def split(self):
        """
        Add the negative samples and create the train, validation and test datasets
        """

        # Create the validation and test datasets:
        # To do it, we will simply get the last seen movie for every user of the train_raw dataset:
        self.validation = self.train_raw.loc[self.train_raw.groupby("user_id")["timestamp"].idxmax()]

        # Eliminate the validation rows of the train:
        self.train = self.train_raw.drop(index=self.validation.index)

        # Add the negative samples:
        self.test = pd.concat([self.test_raw, self.negative_samples_test])
        self.validation = pd.concat([self.validation, self.negative_samples_validation])
        self.train = pd.concat([self.train, self.negative_samples_train])

        # Drop the timestamp column and we have our datasets ready:
        self.validation = self.validation.drop("timestamp", axis=1)
        self.train = self.train.drop("timestamp", axis=1)
        self.test = self.test.drop("timestamp", axis=1)

    def generate_loaders(self, hparams) -> Tuple[DataLoader]:
        """
        Turn dataframes into Torch tensors and create the dataloaders
        """
                                                                    
        # Create the tensor datasets:
        features_train = torch.tensor(self.train.drop("label", axis=1).values)
        target_train = torch.tensor(self.train["label"].values)
        train_dataset = TensorDataset(features_train, target_train)

        features_validation = torch.tensor(self.validation.drop("label", axis=1).values)
        target_validation = torch.tensor(self.validation["label"].values)
        validation_dataset = TensorDataset(features_validation, target_validation)

        features_test = torch.tensor(self.test.drop("label", axis=1).values)
        target_test = torch.tensor(self.test["label"].values)
        test_dataset = TensorDataset(features_test, target_test)

        self.train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, len(validation_dataset), shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)








        

