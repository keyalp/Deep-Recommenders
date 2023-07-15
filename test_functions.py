#Define Metrics
import math
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import DataLoader

def getHitRatio(recommend_list: List, gt_item: int):
    if gt_item in recommend_list:
        return 1
    else:
        return 0

def getNDCG(recommend_list: np.ndarray, gt_item: int):
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0

def generate_test_df(model: nn.Module, test_loader: DataLoader, device) -> pd.DataFrame:
  """
  Returns a dataframe with the columns: ["user_id", "item_id", "target", "output_prob"]
  """
  for data, target in test_loader:

    data = data.to(device)
    target = target.to(device)

    user_ids = data[:, 0]
    movie_ids = data[:, 1]

    output = model(user_ids, movie_ids).squeeze()

    # Generate a dataframe with the reults: ["user_id", "item_id", "output_prob"]
    results_dataframe = pd.DataFrame(
        {
            "user_id": user_ids.cpu().detach().numpy(),
            "item_id": movie_ids.cpu().detach().numpy(),
            "target": target.cpu().detach().numpy(),
            "output_prob": output.cpu().detach().numpy()
            }
        )


  return results_dataframe

def test_function(test_df: pd.DataFrame) -> Tuple[float]:
  """
  Get's as input a dataframe with the predictions and calculates the HitRatio and the NDCG
  """
  # Initialize lists to store hit ratios and NDCG
  hitratio_list = []
  ndcg_list = []

  # iterate trough the users:
  for us_id, us_df in test_df.groupby("user_id"):
    # Get the recommendations in a list:
    recomendations = us_df.sort_values(by="output_prob", ascending=False)["item_id"][:10]

    # Get the positive interation ofr user us_id:
    pos_interaction = us_df.loc[us_df["target"] == 1, "item_id"].iloc[0]

    # Calculate the metrics and add them to the lists:
    hit_ratio = getHitRatio(list(recomendations), pos_interaction)
    ndcg = getNDCG(recomendations.values, pos_interaction)

    hitratio_list.append(hit_ratio)
    ndcg_list.append(ndcg)

  hit_ratio_mean = np.mean(np.array(hitratio_list))
  ndcg_mean = np.mean(np.array(ndcg_list))

  return (hit_ratio_mean, ndcg_mean)