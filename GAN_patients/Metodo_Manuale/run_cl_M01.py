import os, sys
import random
import torch
import numpy as np
import logging
import pandas as pd
from torch.utils.data import DataLoader, random_split
from utils_gan import PairedMelSpectrogramDataset
from DCGAN_dys_V2 import DysarthricGAN
from train_Audio_Q import train_dcgan
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # iperparametri
    exp_ID = "001"
    batch_size = 32
    in_channels = 1
    num_epochs = 501
    beta1 = 0.4556
    lr_g = 0.000127
    lr_d = 0.000011    
    dropout_p = 0.5
    update_d_every = 7
    lambda_l1 = 0
    lambda_sc = 1
    lambda_mr = 3.0
    result_path = "/home/deepfake/DysarthricGAN/M01/results_audio"
    os.makedirs(result_path, exist_ok=True)

    # Dataset 
    dataset = PairedMelSpectrogramDataset("/home/deepfake/DysarthricGAN/M01/M01_MEL_SPEC")
    
   # Mostra quante coppie ci sono
    print(f"Numero di coppie nel dataset: {len(dataset)}")

    
    train_size = len(dataset) 
    generator = torch.Generator().manual_seed(42)
    train_dataset = random_split(dataset, [train_size], generator=generator)[0]

    #per salvare 
    train_indices = train_dataset.indices
    train_folders = []
    for idx in train_indices:
        sano_path, dis_path, sano_in_path = dataset.pairs[idx]
        folder_name = sano_path.parent.name
        train_folders.append(folder_name)
    # Rimuovo eventuali duplicati (per sicurezza)
    train_folders = sorted(set(train_folders))
    train_file = os.path.join(result_path, "train_folders.txt")
    # Salvataggio in file
    with open(train_file, "w") as f:
        for name in train_folders:
            f.write(name + "\n")
    print(" Salvate le cartelle in train_folders.txt") 

    # definizione dei dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    

    logging.info("\nTraining with validation")

    dc_gan = DysarthricGAN(in_channels=in_channels, device=device, residual_mode='sum', dropout_p=dropout_p)
    netG, netD = dc_gan.get_models()

    result_path = os.path.join(result_path, exp_ID)
    os.makedirs(result_path, exist_ok=True)

    # Configura logging su file e console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(result_path, "train_log.txt")),
            logging.StreamHandler()
        ]
    )


    # iniziaizza il tensorboard writer solo per testo locali
    writer = SummaryWriter(log_dir=result_path)

    best_gen_path, epoch_d_all, best_d_all, diff_best_d_all = train_dcgan(
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        netD=netD,
        netG=netG,
        train_loader=train_loader,
        val_loader=None,
        lambda_l1=lambda_l1,
        lambda_sc=lambda_sc,
        lambda_mr=lambda_mr,
        result_path=result_path,
        num_epochs=num_epochs,
        update_d_every=update_d_every,
        writer=writer
    )

    log_line = (
        f"[Best_epoch: {epoch_d_all} | distance: {best_d_all:.4f} | diff: {diff_best_d_all:.4f}"
    )

     # Log locale nel trial
    with open(os.path.join(result_path, "result_log.txt"), "w") as f:
        f.write(log_line)   


    print(f"last generator saved: ")
    
    logging.info(f"\nTRAINING COMPLETE — Results saved in:\n{result_path}")

if __name__ == "__main__":

    main()
