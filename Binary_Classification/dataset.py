
import torch
from torch.utils.data import Dataset
import os


class UA_melspectrogram(Dataset):
    """
    """
    def __init__(self, meta_df, cfg):
        self.meta_df = meta_df
        self.cfg = cfg


    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        # load the MELSPECTROGRAM
        S = torch.load(os.path.join(self.cfg.data_root, self.meta_df.iloc[idx]['f_name']), weights_only=True)   
        real_lbl = self.meta_df.iloc[idx]['intell_level']
        if real_lbl <= self.cfg.lbl_threshold:
            lbl = 0  # low intelligibility = high dysarthria
        else:
            lbl = 1  # high intelligibility = low dysarthria
        lbl = torch.tensor(lbl, dtype=torch.long)
        

        if self.cfg.norm == 'minmax':
            # normalization in [0,1]
            S_min = self.meta_df.iloc[idx]['S_min']
            S_max = self.meta_df.iloc[idx]['S_max']
            S = (S - S_min) / (S_max - S_min)

        elif self.cfg.norm == 'z':
            # standardizzazione
            S = (S - self.cfg.global_mean) / (self.cfg.global_std + 1e-8)
            """
            means = torch.tensor(mean_list)
            stds  = torch.tensor(std_list)

            global_mean = means.mean()
            global_var  = (stds**2 + (means - global_mean)**2).mean()
            global_std  = torch.sqrt(global_var)
            """

        return (S.unsqueeze(0), lbl) # add channel axis



if __name__ == '__main__':
    pass







        
   
