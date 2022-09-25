import torch
from torch.utils.data import Dataset
import numpy as np


class POIDataset(Dataset):
    def __init__(self, df, field_dims):
        self.df = df
        self.field_dims = field_dims

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        data = self.df.iloc[idx][['gender','friend_num','follow_num','hex7','week','hour','venueCategory','bus','subway','parking','crime','interaction']].values.astype(int)
        target = self.df.iloc[idx]['visit'].astype(int)
        return torch.from_numpy(np.asarray(data)), torch.from_numpy(np.asarray(target))

    
class PrivatePOIDataset(Dataset):
    def __init__(self, user_df, user_id, field_dims):
        self.df = user_df
        self.field_dims = field_dims
        self.user_id = user_id

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        data = self.df.iloc[idx][['gender','friend_num','follow_num','hex7','week','hour','venueCategory','bus','subway','parking','crime','interaction']].values.astype(int)
        target = self.df.iloc[idx]['visit'].astype(int)
        return torch.from_numpy(np.asarray(data)), torch.from_numpy(np.asarray(target))