import pandas as pd
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from typing import Tuple

class NetflixData():

    def __init__(self):
        
        self.train_raw = pd.read_csv("Subset1M_traindata.csv", names=["user_id", 'item_id', 'label', 'timestamp'])
        self.test_raw = pd.read_csv("Subset1M_testdata.csv", names=["user_id", 'item_id', 'label', 'timestamp'])

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
        print(f"For train dataset: {self.negative_samples_train}")
        print(f"For validation dataset: {self.negative_samples_validation}")
        print(f"For test dataset: {self.negative_samples_test}")
    
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

    def generate_lodaders(self, hparams) -> Tuple[DataLoader]:
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
        self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)o0'0








        

