import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import dct 
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pandas as pd
import torch.nn.functional as F
from losses import LogSpectralConvLoss, LogSTFTMagnitudeLoss



# Costanti per l'analisi Centroidi MFCC
# Queste costanti definiscono la trasformazione che omogenizza i dati per il calcolo del centroide
TARGET_MFCC_DIM = 12 # Da 80 a 12 coefficienti
TARGET_TIME_LENGTH = 89 



#Funzioni di Trasformazione
def log_mel_to_mfcc(log_mel_tensor, target_mfcc_dim=12):
    """
    Applica la DCT Type-II a un Log-Mel Spectrogram e seleziona i primi N coefficienti.
    Risultato: [1, 12, 178]
    """
    if log_mel_tensor.dim() > 3:
        log_mel_tensor = log_mel_tensor.squeeze(0)  # Rimuovi dimensione batch se presente
    # log_mel_tensor: [1, 80, T]
    log_mel_np = log_mel_tensor.squeeze(0).cpu().numpy()  # -> [80, T]

    # DCT-II lungo asse mel (0)
    mfcc_np = dct(log_mel_np, type=2, axis=0, norm='ortho')  # -> [80, T]

    # Prendi solo i primi coefficienti (MFCC più importanti)
    mfcc_np = mfcc_np[:target_mfcc_dim, :]  # -> [12, T]

    # Torna a tensor + batch dimension
    mfcc_tensor = torch.from_numpy(mfcc_np).float().unsqueeze(0)  # -> [1, 12, T]

    return mfcc_tensor

def reduce_time_dimension(mfcc_tensor):
    """
    Applica un pooling (media) adattivo lungo la dimensione temporale 
    per raggiungere una lunghezza fissa di 89 frame.
    
    mfcc_tensor: [1, Features, Time] (e.g., [1, 12, 178] o [1, 12, 300]). 
    Risultato: [89, 12] (trasposto per la PCA).
    """
    
    # 1. Prepara i dati: Rimuovi la dimensione batch e converti in NumPy
    # data: [Features (12), Time (L)]
    data = mfcc_tensor.squeeze(0).numpy()
    
    num_features, L = data.shape
    
    # 2. Inizializza l'array risultante [Features (12), TARGET_TIME_LENGTH (89)]
    reduced_data = np.zeros((num_features, TARGET_TIME_LENGTH))
    
    # 3. Calcola i confini dei 89 "bin" (finestre temporali)
    # np.linspace crea 90 punti uniformemente distribuiti tra 0 e L
    # Questo definisce gli 89 intervalli (finestre)
    bin_edges = np.linspace(0, L, TARGET_TIME_LENGTH + 1)
    
    # 4. Loop per calcolare la media in ciascuno degli 89 bin
    for i in range(TARGET_TIME_LENGTH):
        # Definisce l'inizio e la fine del bin corrente (arrotondati all'intero più vicino)
        start = int(np.round(bin_edges[i]))
        end = int(np.round(bin_edges[i+1]))
        
        # Estrai la finestra [12, dimensione_finestra]
        window = data[:, start:end]
        
        # Gestisce il caso di finestra vuota (non dovrebbe accadere con L>89)
        if window.shape[1] > 0:
            # Calcola la media lungo l'asse temporale (asse 1)
            reduced_data[:, i] = np.mean(window, axis=1)
        else:
            # Se la finestra è vuota (es. se L è molto piccolo) usa il valore precedente o zero
            reduced_data[:, i] = reduced_data[:, i-1] if i > 0 else 0
            
    # 5. **TRASPOSIZIONE FONDAMENTALE PER LA PCA:**
    # Trasforma da [Feature, Frame] a [Frame, Feature]
    # Risultato: [89, 12] (89 esempi/righe, 12 feature/colonne)
    return reduced_data.T

# funzione di utility
def log_spec(writer, img, label, epoch):
    """
    Log MELSpectrograms as images into tensorboard
    """
    spec_np = img.detach().cpu().numpy() # detach from gradient chain, move to cpu and transform to ndarray
    spec_np = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-8) # scaling, requested by matplotlib
    spec_np = np.flipud(spec_np) # put origin at lower
    colormap = matplotlib.colormaps['magma']
    spec_color = colormap(spec_np)[..., :3]  # shape [H, W, 3], RGB
    spec_tensor = torch.tensor(spec_color).permute(2, 0, 1)
    writer.add_image(label, spec_tensor, epoch)


def train_dcgan(
        device,
        lr_g,
        lr_d,
        beta1,
        netD,
        netG,
        train_loader,
        val_loader,
        lambda_l1,
        lambda_sc,
        result_path,
        num_epochs,
        update_d_every, 
        use_reduce_on_plateau = False, #può essere True o False
        writer = None
    ):

    # img_path = os.path.join(result_path, "generated_images")
    # os.makedirs(img_path, exist_ok=True)
    tensor_path = os.path.join(result_path, "generated_tensors")
    os.makedirs(tensor_path, exist_ok=True)
    checkpoint_path = os.path.join(result_path, "best_generator.pth")

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_2 = LogSpectralConvLoss().to(device)
    # mse_loss = torch.nn.MSELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))



    # LR Scheduler
    if use_reduce_on_plateau:
        schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=5)
        schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=5)
    else:
        schedulerD = CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min= lr_d * 0.5)
        schedulerG = CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min= lr_g * 0.5)    

    best_val_loss = float('inf')
    best_single_val_loss = float('inf')
    train_losses_D, train_losses_G, val_losses, accuracies_D = [], [], [], []
    val_loss_history = []

    early_stop = False

    best_distance = 1000.0
    best_diff = 1000.0
    best_d_all = 1000.0
    diff_best_d_all = 1000.0

    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        running_loss_D, running_loss_G, running_acc_D, D_bce_real_loss, D_bce_fake_loss, G_bce_loss, G_sc_loss, G_l1_loss, G_mel_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


        for i, (batch_sano, batch_disartrico, _) in enumerate(train_loader):
            # step_count += 1
            batch_sano = batch_sano.to(device)
            batch_disartrico = batch_disartrico.to(device)
            batch_size = batch_sano.size(0)

            optimizerD.zero_grad()

            # DISCRIMINATORE
            label_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)  # label smoothing
            label_fake = torch.full((batch_size,), 0.1, dtype=torch.float, device=device)

            # out_real = netD(batch_disartrico)
            out_real = netD(torch.cat((batch_sano, batch_disartrico), dim=1))
            out_real = out_real.view(-1) # i disartrici reali vengono passati al D
            loss_real = criterion(out_real, label_real) # calcolo loss dei disartrici reali con la label (1)

            fake_data = netG(batch_sano) # i sani vengono passati al G per generare i disartrici fake
            # out_fake = netD(fake_data.detach())
            out_fake = netD(torch.cat((batch_sano, fake_data.detach()), dim=1))
            out_fake = out_fake.view(-1) # i disartrici fake generati vengono passati al D
            loss_fake = criterion(out_fake, label_fake) # calcolo loss dei disartrici fake con la label (0)

            # Accumula loss (non dividere arbitrariamente)
            loss_D_batch = (loss_real + loss_fake) * 0.5

            if i % update_d_every == 0: # update discriminator every "update_d_every" iterations
                loss_D_batch.backward()
                # if grad_clip is not None:
                #     torch.nn.utils.clip_grad_norm_(netD.parameters(), grad_clip)
                optimizerD.step()

            running_loss_D += loss_D_batch.item()
            D_bce_real_loss += loss_real.item()
            D_bce_fake_loss += loss_fake.item()

            # GENERATORE
            optimizerG.zero_grad()
            label_gen = torch.full((batch_size,), 0.9, dtype=torch.float, device=device) 
            # out_gen = netD(fake_data)
            out_gen = netD(torch.cat((batch_sano, fake_data), dim=1))
            out_gen = out_gen.view(-1) # passa i disartrici fake generati al D
            adv_loss = criterion(out_gen, label_gen) #applica la label (1) per ingannare il D

            # Mask L1
            mask = (batch_disartrico != 0).float()
            if mask.sum() == 0:
                l1_loss = torch.abs(fake_data - batch_disartrico).mean()
            else:
                l1_loss = (torch.abs(fake_data - batch_disartrico) * mask).sum() / (mask.sum() + 1e-8)

            sc_loss = criterion_2(fake_data, batch_disartrico)    

            loss_G = adv_loss + lambda_sc * sc_loss
            loss_G.backward()
            optimizerG.step()

            running_loss_G += loss_G.item()
            G_bce_loss += adv_loss.item()
            # G_fm_loss += fm_loss
            G_l1_loss += l1_loss.item()
            G_sc_loss += sc_loss.item()


        
        # reduce LR on plateau step
        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)
        schedulerD.step()
        schedulerG.step()

        # per tensorboard
        writer.add_scalar('D_BCE_REAL_LOSS',      (D_bce_real_loss/len(train_loader)), epoch)
        writer.add_scalar('D_BCE_FAKE_LOSS',      (D_bce_fake_loss/len(train_loader)), epoch)
        writer.add_scalar('G_BCE_LOSS',           (G_bce_loss/len(train_loader)), epoch)
        writer.add_scalar('G_L1_LOSS',            (G_l1_loss/len(train_loader)), epoch)
        writer.add_scalar('G_SC_LOSS',            (G_sc_loss/len(train_loader)), epoch)

        current_lr_d = optimizerD.param_groups[0]['lr']  # prendi il learning rate del primo param_group
        writer.add_scalar('Learning_Rate_D', current_lr_d, epoch)
        current_lr_g = optimizerG.param_groups[0]['lr']  # prendi il learning rate del primo param_group   
        writer.add_scalar('Learning_Rate_G', current_lr_g, epoch)

        # === VALIDAZIONE ===
        val_loss, val_mel_loss, val_adv_loss, val_l1_loss, val_fm_loss, total_val_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        output_0 = [] #per i sani input
        output_1 = [] #per i disartrici
        output_2 = [] #per i sani generati
        
        if val_loader is not None:
            netG.eval()

            with torch.no_grad():
                for val_sano, val_dis, val_sano_in in val_loader:
                    val_sano = val_sano.float().to(device)
                    val_dis = val_dis.float().to(device)
                    val_sano_in = val_sano_in.float().to(device)

                    fake_val = netG(val_sano)
                    output_val = netD(torch.cat((val_sano, fake_val), dim=1))
                    output_val = output_val.view(-1)
                    labels_val = torch.full((val_sano.size(0),), 0.9, dtype=torch.float, device=device)

                    # ADVERSARIAL LOSS
                    adv_loss = criterion(output_val, labels_val)
                    val_adv_loss += adv_loss.item()
                     
                    sc_loss = criterion_2(output_val, labels_val) 
                    # L1 LOSS (mascherata)
                    mask = (val_dis != 0).float()
                    if mask.sum() == 0:
                        l1_loss = torch.abs(fake_val - val_dis).mean()
                    else:
                        l1_loss = (torch.abs(fake_val - val_dis) * mask).sum() / (mask.sum() + 1e-8)
                    val_l1_loss += l1_loss.item()

                    # === TOTAL ===
                    total_val_loss = adv_loss + lambda_sc * sc_loss
                    val_loss += total_val_loss.item()

                    # Sano Input
                    mfcc_sano_in = log_mel_to_mfcc(val_sano_in)
                    reduced_sano_in = reduce_time_dimension(mfcc_sano_in)
                    reduced_t_sano_in = torch.from_numpy(reduced_sano_in).float()
                    output_0.append(reduced_t_sano_in)

                    # Disartrico Target
                    mfcc_dis = log_mel_to_mfcc(val_dis)
                    reduced_dis = reduce_time_dimension(mfcc_dis)
                    reduced_t_dis = torch.from_numpy(reduced_dis).float()
                    output_1.append(reduced_t_dis)
                    
                    # Sano Generato
                    mfcc_gen = log_mel_to_mfcc(fake_val)
                    reduced_gen = reduce_time_dimension(mfcc_gen)
                    reduced_t_gen = torch.from_numpy(reduced_gen).float()
                    output_2.append(reduced_t_gen)
            
            
            # Fai la media sulle dimensioni del dataset
            n = len(val_loader.dataset)
            val_loss /= n
            val_losses.append(val_loss)

            # aggiorna la lista storica delle val_loss
            val_loss_history.append(val_loss)

        writer.add_scalar('Validation_loss', val_loss, epoch)

        tot_output_0 = torch.cat(output_0, dim=0)
        tot_output_1 = torch.cat(output_1, dim=0)
        tot_output_2 = torch.cat(output_2, dim=0)     


        labels_0 = np.zeros(tot_output_0.shape[0])
        labels_1 = np.ones(tot_output_1.shape[0])
        labels_2 = np.full(tot_output_2.shape[0], 2)    

        all_embeddings = torch.cat([tot_output_0, tot_output_1, tot_output_2], dim=0)
        all_labels = np.concatenate([labels_0, labels_1, labels_2], axis=0)   
        
        X = all_embeddings.numpy() 
        if len(X.shape) == 3 and X.shape[1] == 1:
            X = X.squeeze(axis=1)

        # Usiamo la dimensione effettiva degli MFCC
        mfcc_dim = X.shape[1] 
        df_features = pd.DataFrame(X, columns=[f'MFCC{i}' for i in range(1, mfcc_dim + 1)])

        # Aggiungiamo le etichette, assicurandoci che siano convertite in int/float compatibili se necessario
        df_features['Target'] = all_labels.astype(int) 

        # Calcolo dei Centroidi (Media) per ogni classe nello spazio a D dimensioni MFCC
        # I centroidi rappresentano il "punto medio" del cluster nel dominio delle feature
        centroids_Dd = df_features.groupby('Target').mean()

        # Estrazione dei vettori Centroidi utilizzando le etichette numeriche (0, 1, 2)
        mu_sano_in = centroids_Dd.loc[0].values
        mu_disartrico = centroids_Dd.loc[1].values
        mu_sano_gen = centroids_Dd.loc[2].values

        V_target = mu_sano_gen - mu_disartrico

        V_input = mu_sano_gen - mu_sano_in 

        # Vettore di Spostamento desiderato (Sano Input -> Disartrico)
        V_native = mu_disartrico - mu_sano_in

        # Distanza euclidea (Magnitudo dell'errore residuo)
        distance_to_target = np.linalg.norm(V_target)

        # Distanza euclidea (Magnitudo dell'errore residuo)
        distance_from_input = np.linalg.norm(V_input) 

        distance_native = np.linalg.norm(V_native)

        diff = abs(distance_from_input - distance_native)
         # === LOGGING TENSORBOARD DELLE METRICHE DEI CENTROIDI ===
        writer.add_scalar('Validation_Metrics/d1 (gen-dis)', distance_to_target, epoch) # Base
        writer.add_scalar('Validation_Metrics/Centroid_Isosceles_Diff', diff, epoch)
        writer.add_scalar('Validation_Metrics/d2 (gen-sano)', distance_from_input, epoch)
        writer.add_scalar('Validation_Metrics/d3 (dis-sano)', distance_native, epoch)

        if distance_to_target < best_distance and diff < best_diff:
            best_epoch = epoch
            best_distance = distance_to_target
            best_diff = diff
            best_gen_path = os.path.join(result_path, f"best_generator.pth")
            torch.save(netG.state_dict(), best_gen_path)
            print(f"best Generator salvato: {best_gen_path}")
        elif distance_to_target < best_d_all:
            epoch_d_all = epoch
            best_d_all = distance_to_target
            diff_best_d_all = diff
            best_d_all_path = os.path.join(result_path, f"best_dist_gen.pth")
            torch.save(netG.state_dict(), best_d_all_path)
            print(f"best distance gen salvato: {best_d_all_path}")
       

        #writer.flush()  
        if epoch == 0:
            fixed_sano, fixed_disartrico, _ = next(iter(val_loader))
            fixed_sano = fixed_sano.float().to(device)
            fixed_disartrico = fixed_disartrico.float().to(device)
            log_spec(writer, img=fixed_sano[0,0], label='Input_MELSpec', epoch=epoch)
            log_spec(writer, img=fixed_disartrico[0,0], label='Target_MELSpec', epoch=epoch)

        netG.eval()
        with torch.no_grad():
            fixed_fake = netG(fixed_sano)
        log_spec(writer, img=fixed_fake[0,0], label='Generated_MELSpec', epoch=epoch)
        writer.flush()
        
        if epoch % 10 == 0:
            ex_generated = os.path.join(tensor_path, f'generated_melspec_{epoch}')
            os.makedirs(ex_generated, exist_ok=True)
            for i, (s, _, _) in enumerate(val_loader):
                if i >= 5:
                    break
                fake = netG(s.to(device)) 
                 
                torch.save(fake.detach().cpu().squeeze(), os.path.join(ex_generated, f'generated_melspec_{i}.pth'))
            print(f'>> Generated MELSpec saved')

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | Val_Loss: {val_loss:.4f}  | Dist_Target: {distance_to_target:.4f} | Diff_Isoscele: {diff:.4f} ")
        # if early_stop:                
        #     break
        
    writer.close()    

    

    return checkpoint_path, best_epoch, best_distance, best_diff, epoch_d_all, best_d_all, diff_best_d_all, 
