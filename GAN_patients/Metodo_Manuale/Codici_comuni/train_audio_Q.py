import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import dct 
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pandas as pd
import torch.nn.functional as F
import soundfile as sf  # Assicurati di averlo installato: pip install soundfile
from losses import LogSpectralConvLoss, LogSTFTMagnitudeLoss, MRSTFTLoss_Mel



# Costanti per l'analisi Centroidi MFCC
# Queste costanti definiscono la trasformazione che omogenizza i dati per il calcolo del centroide
TARGET_MFCC_DIM = 12 # Da 80 a 12 coefficienti
TARGET_TIME_LENGTH = 89 

# --- Parametri WaveGlow ---
SR_ORIG = 22050
MU_SPK = -9.1232
SIGMA_SPK = 3.5412


def log_mel_to_mfcc(log_mel_tensor, target_mfcc_dim=12):
    """
    Applica la DCT Type-II su un batch di Log-Mel Spectrogram.
    Input: [B, C, 80, T] -> [B, 1, 80, T]
    Output: [B, C, target_mfcc_dim, T] -> [B, 1, 12, T]
    """
    
    # Rimuove dimensione canale se è 1, e sposta sul CPU per l'operazione numpy/scipy
    # input_data: [B, 80, T]
    input_data = log_mel_tensor.squeeze(1).detach().cpu().numpy() 

    # Applicazione DCT Type-II
    # Applica la DCT lungo l'asse delle Mel-Features (asse 1)
    # output_dct: [B, 80, T]
    mfcc_np = dct(input_data, type=2, axis=1, norm='ortho')

    # Selezione dei primi coefficienti
    # mfcc_np_reduced: [B, 12, T]
    mfcc_np_reduced = mfcc_np[:, :target_mfcc_dim, :] 

    # Ritorna a tensor e ripristina dimensione canale
    # mfcc_tensor: [B, 1, 12, T]. Manteniamo il batch e il canale.
    mfcc_tensor = torch.from_numpy(mfcc_np_reduced).float().unsqueeze(1)

    return mfcc_tensor.to(log_mel_tensor.device) # Riporta sul device originale

def reduce_time_dimension(mfcc_tensor):
    """
    Applica un Adaptive Average Pooling lungo la dimensione temporale per 
    raggiungere una lunghezza fissa di 89 frame in modo vettorizzato.
    
    mfcc_tensor: [B, C, Features, Time] (e.g., [B, 1, 12, T]). 
    Output: [B * TARGET_TIME_LENGTH, Features] -> [B * 89, 12] (trasposto per la PCA/Centroidi).
    """
    
    # Preparazione
    # Rimuovi la dimensione canale (C=1)
    # data_t: [B, Features, Time] -> [B, 12, T]
    data_t = mfcc_tensor.squeeze(1) 
    
    # Adaptive Pooling
    # Applica il pooling su tutti i batch contemporaneamente.
    # pooled_data: [B, Features, TARGET_TIME_LENGTH] -> [B, 12, 89]
    pooled_data = F.adaptive_avg_pool1d(data_t, output_size=TARGET_TIME_LENGTH)
    
    # Trasformazione e Riorganizzazione per il Centroid Embedding
    # Vogliamo [B * 89, 12] (Frame, Feature)
    
    # pooled_data trasposto: [B, TARGET_TIME_LENGTH, Features] -> [B, 89, 12]
    pooled_data = pooled_data.transpose(1, 2)
    
    # Rimodella (Flatten) il batch e il tempo in un'unica dimensione
    # final_output: [B * 89, 12]
    B, L, F_dim = pooled_data.shape 
    final_output = pooled_data.reshape(B * L, F_dim)
    
    # Ritorna come Tensor (non Numpy)
    return final_output.float()

# funzione di utility
def log_spec(writer, img, label, epoch):
    """
    Log MELSpectrograms as images into tensorboard
    """
    spec_np = img.detach().cpu().numpy() 
    # Riporta in scala log-mel per una visualizzazione più naturale
    spec_np = (spec_np * SIGMA_SPK) + MU_SPK
    spec_np = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-8) # scaling
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
        lambda_mr,
        result_path,
        num_epochs,
        update_d_every, 
        use_reduce_on_plateau = False, #può essere True o False
        writer = None
    ):

    tensor_path = os.path.join(result_path, "generated_tensors")
    os.makedirs(tensor_path, exist_ok=True)
    checkpoint_path = os.path.join(result_path, "best_generator.pth")

    # CARICAMENTO WAVEGLOW PER SINTESI 
    print(">> Caricamento WaveGlow...")
    waveglow = None
    try:
        waveglow = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_waveglow',
            model_math='fp32',
            trust_repo=True
        ).to(device).eval()
        
        waveglow = waveglow.remove_weightnorm(waveglow)
        from denoiser import Denoiser 
        denoiser = Denoiser(waveglow).to(device)
        print("WaveGlow caricato e pronto per la sintesi.")
    except Exception as e:
        print(f"Errore caricamento WaveGlow: {e}")

    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_2 = LogSpectralConvLoss().to(device)
    criterion_3 = MRSTFTLoss_Mel(
        mu=MU_SPK,      
        sigma=SIGMA_SPK
    ).to(device)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))



    # LR Scheduler
    if use_reduce_on_plateau:
        schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=5)
        schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=5)
    else:
        schedulerD = CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min= lr_d * 0.1)
        schedulerG = CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min= lr_g * 0.1)    

    best_val_loss = float('inf')
    best_single_val_loss = float('inf')
    train_losses_D, train_losses_G, val_losses, accuracies_D = [], [], [], []
    val_loss_history = []

    early_stop = False

    
    best_d_all = 1000.0
    diff_best_d_all = 1000.0

    output_0 = [] #per i sani input
    output_1 = [] #per i disartrici
    output_2 = [] #per i sani generati


    with torch.no_grad(): 
        print(">> Calcolo dei Centroidi Statici (Sano Input e Disartrico Target)...")
        for batch_sano, batch_disartrico, batch_sano_in in train_loader:
            
            # Sposta i dati sul device (CPU o GPU)
            batch_disartrico = batch_disartrico.to(device)
            batch_sano_in = batch_sano_in.to(device)

            # Trasformazione MFCC e Riduzione Temporale
            # Sano Input
            mfcc_sano_in = log_mel_to_mfcc(batch_sano_in, target_mfcc_dim=TARGET_MFCC_DIM)
            reduced_sano_in = reduce_time_dimension(mfcc_sano_in)
            output_0.append(reduced_sano_in)

            # Disartrico Target
            mfcc_dis = log_mel_to_mfcc(batch_disartrico, target_mfcc_dim=TARGET_MFCC_DIM)
            reduced_dis = reduce_time_dimension(mfcc_dis)
            output_1.append(reduced_dis)

    # Aggrega e Calcola i Centroidi Finali Fissi
    tot_output_0 = torch.cat(output_0, dim=0)
    tot_output_1 = torch.cat(output_1, dim=0)

    # Crea i DataFrame/Numpy per calcolare i Centroidi in D-dimensioni (MFCC)
    X_0 = tot_output_0.cpu().numpy() # [N*89, 12]
    X_1 = tot_output_1.cpu().numpy() # [N*89, 12]

    mfcc_dim = X_0.shape[1] 
    mfcc_cols = [f'MFCC{i}' for i in range(1, mfcc_dim + 1)]

    # DataFrame per Sano Input
    df_sano = pd.DataFrame(X_0, columns=mfcc_cols)
    mu_sano_in = df_sano.mean().values # Il centroide [12] è la media delle 89*N righe

    # DataFrame per Disartrico Target
    df_dis = pd.DataFrame(X_1, columns=mfcc_cols)
    mu_disartrico = df_dis.mean().values # Il centroide [12]

    # Calcola il Vettore Nativo (Fisso)
    V_native = mu_disartrico - mu_sano_in
    distance_native = np.linalg.norm(V_native)

    # Rilascia la memoria non necessaria
    del output_0, output_1, tot_output_0, tot_output_1, df_sano, df_dis, X_0, X_1
    print(">> Centroidi Statici Calcolati con Successo.")



    # Estrazione di 5 campioni fissi per il logging e il salvataggio
    try:
        # Estrae un batch (che DEVE contenere almeno 5 campioni, altrimenti riduci il numero)
        batch_sano_fixed, batch_disartrico_fixed, _ = next(iter(train_loader))
        
        # Seleziona i PRIMI 5 campioni
        fixed_sano_input_set = batch_sano_fixed[:5].float().to(device) # [5, 1, 80, T]
        fixed_dis_target_set = batch_disartrico_fixed[:5].float().to(device) # [5, 1, 80, T]
        
        #  Definisce il singolo campione per il LOGGING su TensorBoard
        fixed_sano = fixed_sano_input_set[0].unsqueeze(0) # [1, 1, 80, T]
        fixed_disartrico = fixed_dis_target_set[0].unsqueeze(0) # [1, 1, 80, T]
        
        # Log Statico di TensorBoard (solo il primo campione all'epoca 0)
        log_spec(writer, img=fixed_sano[0,0], label='Input_MELSpec_Fixed', epoch=0) 
        log_spec(writer, img=fixed_disartrico[0,0], label='Target_MELSpec_Fixed', epoch=0)
        
        # SALVA I 5 INPUT SANI E I 5 TARGET DISARTRICI NELLA CARTELLA DEI TENSORI
        fixed_path = os.path.join(tensor_path, 'fixed_samples')
        os.makedirs(fixed_path, exist_ok=True)
        
        for i in range(fixed_sano_input_set.size(0)):
            # Salva l'Input Sano
            torch.save(fixed_sano_input_set[i].detach().cpu().squeeze(), 
                       os.path.join(fixed_path, f'input_sano_{i}.pth'))
            # Salva il Target Disartrico
            torch.save(fixed_dis_target_set[i].detach().cpu().squeeze(), 
                       os.path.join(fixed_path, f'target_dis_{i}.pth'))
        
        print(f">> 5 Campioni Fissi (Sani e Disartrici) salvati in {fixed_path}")
        
        FIXED_SAMPLE_EXISTS = True
    except Exception as e:
        print(f"Errore nell'estrazione/salvataggio del campione fisso: {e}")
        fixed_sano, fixed_disartrico, fixed_sano_input_set, fixed_dis_target_set = None, None, None, None
        FIXED_SAMPLE_EXISTS = False

    for epoch in range(num_epochs):
        netG.train()
        netD.train()
        running_loss_D, running_loss_G, running_acc_D, D_bce_real_loss, D_bce_fake_loss, G_bce_loss, G_sc_loss, G_l1_loss, G_MR_STFT_loss,  G_fm_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        output_2 = []

        for i, (batch_sano, batch_disartrico, batch_sano_in) in enumerate(train_loader):
            batch_sano = batch_sano.to(device)
            batch_disartrico = batch_disartrico.to(device)
            batch_sano_in = batch_sano_in.to(device)
            batch_size = batch_sano.size(0)

            optimizerD.zero_grad()

            # DISCRIMINATORE
            label_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)  # label smoothing  
            out_real= netD(torch.cat((batch_sano, batch_disartrico), dim=1))    
            out_real = out_real.view(-1) # i disartrici reali vengono passati al D
            loss_real = criterion(out_real, label_real) # calcolo loss dei disartrici reali con la label (1)
            
          
            label_fake = torch.full((batch_size,), 0.1, dtype=torch.float, device=device)
            fake_data = netG(batch_sano) # i sani vengono passati al G per generare i disartrici fake
            out_fake= netD(torch.cat((batch_sano, fake_data.detach()), dim=1))
            out_fake = out_fake.view(-1) # i disartrici fake generati vengono passati al D
            loss_fake = criterion(out_fake, label_fake) # calcolo loss dei disartrici fake con la label (0)

            # Accumula loss
            loss_D_batch = (loss_real + loss_fake) * 0.5

            if i % update_d_every == 0: # update discriminator ogni "update_d_every" iterationi
                loss_D_batch.backward()
                optimizerD.step()

            running_loss_D += loss_D_batch.item()
            D_bce_real_loss += loss_real.item()
            D_bce_fake_loss += loss_fake.item()

            # GENERATORE
            optimizerG.zero_grad()
            label_gen = torch.full((batch_size,), 0.9, dtype=torch.float, device=device) 
            out_gen= netD(torch.cat((batch_sano, fake_data), dim=1))
            out_gen = out_gen.view(-1) # passa i disartrici fake generati al D
            adv_loss = criterion(out_gen, label_gen) #applica la label (1) per ingannare il D

            # Mask L1
            mask = (batch_disartrico != 0).float()
            if mask.sum() == 0:
                l1_loss = torch.abs(fake_data - batch_disartrico).mean()
            else:
                l1_loss = (torch.abs(fake_data - batch_disartrico) * mask).sum() / (mask.sum() + 1e-8)
  
            fake_data_masked = fake_data * mask
            sc_loss, MR_STFT_loss = criterion_3(fake_data_masked, batch_disartrico)

            loss_G = adv_loss + lambda_mr * MR_STFT_loss + lambda_sc * sc_loss 
            loss_G.backward()
            optimizerG.step()

            running_loss_G += loss_G.item()
            G_bce_loss += adv_loss.item()
            G_l1_loss += l1_loss.item()
            G_sc_loss += sc_loss.item()
            G_MR_STFT_loss += MR_STFT_loss.item()
  
            # Sano Generato
            mfcc_gen = log_mel_to_mfcc(fake_data, target_mfcc_dim=TARGET_MFCC_DIM)
            reduced_gen = reduce_time_dimension(mfcc_gen)
            output_2.append(reduced_gen)


        
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
        writer.add_scalar('G_MR_STFT_LOSS',            (G_MR_STFT_loss/len(train_loader)), epoch)

        current_lr_d = optimizerD.param_groups[0]['lr']  # prende il learning rate del primo param_group
        writer.add_scalar('Learning_Rate_D', current_lr_d, epoch)
        current_lr_g = optimizerG.param_groups[0]['lr']  # prende il learning rate del primo param_group   
        writer.add_scalar('Learning_Rate_G', current_lr_g, epoch)


        tot_output_2 = torch.cat(output_2, dim=0)     
        X_2 = tot_output_2.cpu().numpy() # [N*89, 12]

        del tot_output_2
        mfcc_dim = X_2.shape[1] 
        mfcc_cols = [f'MFCC{i}' for i in range(1, mfcc_dim + 1)]
        

        # DataFrame per Sano Input
        df_gen = pd.DataFrame(X_2, columns=mfcc_cols)
        mu_dis_gen = df_gen.mean().values # Il centroide [12] è la media delle 89*N righe

        V_target = mu_dis_gen - mu_disartrico

        V_input = mu_dis_gen - mu_sano_in 

        # Distanza euclidea (Magnitudo dell'errore residuo)
        distance_to_target = np.linalg.norm(V_target)

        # Distanza euclidea (Magnitudo dell'errore residuo)
        distance_from_input = np.linalg.norm(V_input) 

        diff = abs(distance_from_input - distance_native)

        del X_2, df_gen


        # === LOGGING TENSORBOARD DELLE METRICHE DEI CENTROIDI ===
        writer.add_scalar('T_Metrics/D1 (gen-dis)', distance_to_target, epoch) # Base
        writer.add_scalar('T_Metrics/Centroid_Isosceles_Diff', diff, epoch)
        writer.add_scalar('T_Metrics/D2 (gen-sano)', distance_from_input, epoch)
        writer.add_scalar('T_Metrics/D3 (dis-sano)', distance_native, epoch)



        if distance_to_target < best_d_all:
            epoch_d_all = epoch
            best_d_all = distance_to_target
            diff_best_d_all = diff
            best_d_all_path = os.path.join(result_path, f"best_dist_gen.pth")           
            torch.save(netG.state_dict(), best_d_all_path)
            
            print(f"best distance gen salvato: {best_d_all_path}")
        
        
        # LOGGING DELLO SPETTROGRAMMA GENERATO FISSO SU TENSORBOARD 
        if FIXED_SAMPLE_EXISTS:
            netG.eval()
            with torch.no_grad():
                # Genera il SINGOLO campione per TensorBoard
                fixed_fake = netG(fixed_sano) 
                
                # Log del singolo campione generato
                log_spec(writer, img=fixed_fake[0,0], label='Generated_MELSpec_Fixed', epoch=epoch)
                writer.flush()
                
                # SALVATAGGIO DEI 5 TENSOR GENERATI FISSI 
                if epoch % 15 == 0:
                    
                    ex_generated = os.path.join(tensor_path, f'generated_melspec_{epoch}')
                    os.makedirs(ex_generated, exist_ok=True)
                    
                    fake_set = netG(fixed_sano_input_set) # Output [5, 1, 80, T]
                    gen_4_eval = os.path.join(ex_generated, f"generator_{epoch}.pth")
                    torch.save(netG.state_dict(), gen_4_eval)
                    # Salva ogni campione generato separatamente
                    for i in range(fake_set.size(0)):
                        # Salvataggio Tensore .pth
                        mel_tensor = fake_set[i].detach().cpu().squeeze()
                        torch.save(mel_tensor, os.path.join(ex_generated, f'generated_melspec_{i}.pth'))

                        # Sintesi Audio 
                        if waveglow is not None:
                            try:
                                # Sposta sul device e denormalizza
                                mel_denorm = (mel_tensor.to(device) * SIGMA_SPK) + MU_SPK
                                
                                # WaveGlow vuole [1, 80, T]
                                mel_input_wg = mel_denorm.unsqueeze(0)
                                
                                with torch.no_grad():
                                    audio_gen = waveglow.infer(mel_input_wg)
                                    audio_gen = denoiser(audio_gen, strength=0.05)
                                    
                                    # Post-processing
                                    audio_np = audio_gen[0, 0, :].cpu().numpy().astype(np.float32)
                                    
                                    # Normalizzazione clipping
                                    max_val = np.max(np.abs(audio_np))
                                    if max_val > 1.0:
                                        audio_np = audio_np / max_val
                                    
                                    # Salvataggio WAV
                                    sf.write(os.path.join(ex_generated, f'audio_{i}.wav'), audio_np, SR_ORIG)
                                    
                            except Exception as e:
                                print(f"❌ Errore sintesi audio {i}: {e}")
                                print(f'>> [Epoch {epoch}] Salva 5 .pth e 5 .wav in {ex_generated}')
                            
                    print(f'>> 5 Generated MELSpec saved (Fixed Set)')



        print(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | D1: {distance_to_target:.4f} | Diff: {diff:.4f} ")
        
    writer.close()    

    

    return checkpoint_path, epoch_d_all, best_d_all, diff_best_d_all, 

