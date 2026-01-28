import random
import torch
import numpy as np
import os
import logging
import pandas as pd
from torch.utils.data import DataLoader, random_split
from utils_gan import PairedMelSpectrogramDataset
from DCGAN_dys import DysarthricGAN
from train_op import train_dcgan
from torchsummary import summary
from DCGAN_dys import Generator
import sys
import optuna
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


set_seed(42)
# Parametri fissi per l'esperimento
batch_size = 32
# Adattato l'input shape (assumo che sia l'input del Generator)
input_shape = (1, 80, 257) 
in_channels = 1


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

#Cartella principale per i risultati Optuna
result_path = "/home/deepfake/DysarthricGAN/M11/OptunaResultsM11"
os.makedirs(result_path, exist_ok=True) 

dataset_path = "/home/deepfake/DysarthricGAN/M11/M11_MEL_SPEC"
dataset = PairedMelSpectrogramDataset(dataset_path)

total_size = len(dataset)
# Calcola le dimensioni (90/5/5)
test_size = int(0.05 * total_size)
val_size = int(0.05 * total_size)
train_size = total_size - val_size - test_size

generator = torch.Generator().manual_seed(42)
# Esegue la divisione in tre parti una sola volta
train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size], 
    generator=generator
)

# per salvare train e val e sapere quali parole stanno in uno o nell'altro 
train_indices = train_dataset.indices
val_indices = val_dataset.indices
test_indices = test_dataset.indices

train_folders = []
val_folders = []
test_folders = []

for idx in train_indices:
    sano_path, dis_path, sano_in_path = dataset.pairs[idx]
    folder_name = sano_path.parent.name
    train_folders.append(folder_name)

for idx in val_indices:
    sano_path, dis_path, sano_in_path = dataset.pairs[idx]
    folder_name = sano_path.parent.name
    val_folders.append(folder_name)

for idx in test_indices:
    sano_path, dis_path, sano_in_path = dataset.pairs[idx]
    folder_name = sano_path.parent.name
    test_folders.append(folder_name)    

# Rimozione eventuali duplicati (per sicurezza, anche se non ce ne saranno)
train_folders = sorted(set(train_folders))
val_folders = sorted(set(val_folders))
test_folders = sorted(set(test_folders))

train_file = os.path.join(result_path, "train_folders.txt")
val_file = os.path.join(result_path, "val_folders.txt")
test_file = os.path.join(result_path, "test_folders.txt")

# Salvataggio in file
with open(train_file, "w") as f:
    for name in train_folders:
        f.write(name + "\n")

with open(val_file, "w") as f:
    for name in val_folders:
        f.write(name + "\n")

with open(test_file, "w") as f: 
    for name in test_folders:
        f.write(name + "\n")        

print("✔ Salvate le cartelle in train_folders.txt e val_folders.txt")

def objective(trial):
    set_seed(42)
    # Dataset e Split Globali (90/5/5)
    # Vengono definiti qui globalmente per essere usati nell'objective


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    lambda_l1 = 1
     # Hyperparametri da ottimizzare
    num_epochs = trial.suggest_categorical("epochs", [201, 301, 401])
    beta1 = trial.suggest_float("beta1", 0.3, 0.7)
    lr_g = trial.suggest_float("lr_g", 1e-5, 5e-2, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-2, log=True)
    lambda_sc = trial.suggest_categorical("lambda_sc", [0.5, 1, 1.5, 2, 2.5, 3])
    update_d_every = trial.suggest_categorical("update_d_every", [5, 7, 9, 10, 11, 13, 15, 17, 19])
    dropout_p = trial.suggest_float("dropout", 0.3, 0.7, log=True)
    # Crea una cartella dedicata per questo trial 
    trial_dir = os.path.join(result_path, f"{trial.number + 1}")  # numerazione da 1
    os.makedirs(trial_dir, exist_ok=True)
    tb_log_dir = os.path.join(trial_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_log_dir)

    result_dir = trial_dir

    dc_gan = DysarthricGAN(in_channels=in_channels, device=device, residual_mode='sum', dropout_p=dropout_p)
    netG, netD = dc_gan.get_models()
    

    best_gen_path, best_epoch, best_distance, best_diff, epoch_d_all, best_d_all, diff_best_d_all = train_dcgan(
        device=device,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        netD=netD,
        netG=netG,
        train_loader=train_loader,
        val_loader=val_loader,
        lambda_l1=lambda_l1,
        lambda_sc=lambda_sc,
        num_epochs=num_epochs,
        update_d_every=update_d_every,
        result_path=result_dir,
        writer=writer
    )

    writer.close()
    
    best_param = (best_distance + best_diff) / 2

    # Log del trial
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = (
        f"[{timestamp}] Trial {trial.number + 1} | Best_epoch: {best_epoch} | num_epoch: {num_epochs} | distance: {best_distance:.4f} | diff: {best_diff:.4f}"
        f" | beta1={beta1:.4f} | lr_g={lr_g:.6f} | lr_d={lr_d:.6f}, "
        f" | lambda_sc={lambda_sc:.2f} | dropout={dropout_p:.2f}"
        f" | update_d_every={update_d_every}\n"
    ) 

    log_line_2 = (
        f"[{timestamp}] Trial {trial.number + 1} | epoch_d_all: {epoch_d_all} | num_epoch: {num_epochs} | distance: {best_d_all:.4f} | diff: {diff_best_d_all:.4f}"
        f" | beta1={beta1:.4f} | lr_g={lr_g:.6f} | lr_d={lr_d:.6f}, "
        f" | lambda_sc={lambda_sc:.2f} | dropout={dropout_p:.2f}"
        f" | update_d_every={update_d_every}\n"
    ) 
    
    print(f"[Trial {trial.number + 1}]  (Best epoch: {best_epoch})  distance: {best_distance:.2f}  diff: {best_diff:.2f}")
    # LOG: scrive risultati in un file generale e nel trial 
    global_log_path = os.path.join(result_path, "best_param.txt")
    with open(global_log_path, "a") as f:
        f.write(log_line)

    global_log_path_2 = os.path.join(result_path, "best_distance.txt")
    with open(global_log_path_2, "a") as f:
        f.write(log_line_2)    

     # Log locale nel trial 
    with open(os.path.join(trial_dir, "trial_log.txt"), "w") as f:
        f.write(log_line)   
        f.write(log_line_2)       
    
    

    print(f"Saved trial log → {trial_dir}")


    # Optuna cerca di minimizzare l’output dell’objective
    return best_param

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for DysarthricGAN")
    parser.add_argument("--n_trials", type=int, default=30, help="Numero di combinazioni (try) da testare")
    parser.add_argument("--study_name", type=str, default="DysarthricGAN_F02_Optuna", help="Nome dello studio Optuna")
    args = parser.parse_args()

    # Definisci database Optuna
    db_path = os.path.join(result_path, f"{args.study_name}.db")
    storage = f"sqlite:///{db_path}"

    # Crea o ricarica lo studio 
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",   
        load_if_exists=True,
    )

    print(f" Avvio studio Optuna: {args.study_name}")
    print(f" Risultati salvati in: {result_path}")
    print(f" Numero di trial richiesti: {args.n_trials}\n")

    # Lancia la ricerca
    study.optimize(objective, n_trials=args.n_trials)

    # Mostra risultati migliori 
    best_trial = study.best_trial # Ottiene l'oggetto completo del miglior trial

    # Mostra risultati migliori 
    print("\n Migliori iperparametri trovati:")
    print(f"  Valore minimo (loss): {study.best_value:.6f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Salva un file riassuntivo 
    summary_path = os.path.join(result_path, "optuna_best_trial.txt")
    with open(summary_path, "w") as f:
        f.write(f"Best trial ID: {best_trial.number + 1}\n")
        f.write(f"Best trial value: {study.best_value:.6f}\n")
        f.write("Best parameters:\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")

    print(f"\n📝 Risultati migliori salvati in → {summary_path}")


if __name__ == "__main__":
    main()    
