import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSpectralConvLoss(nn.Module):
    """
    log-Spectral convergence loss module.
    """

    def __init__(self):
        super(LogSpectralConvLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm((y_mag - x_mag), p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """
    Log STFT magnitude loss module.
    see https://arxiv.org/pdf/1808.06719.pdf
    """

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        x_mag = torch.clamp(x_mag, min=1e-6, max=1e3) # avoid underflows & overflows
        y_mag = torch.clamp(y_mag, min=1e-6, max=1e3)
        return F.l1_loss(torch.log(x_mag), torch.log(y_mag))

class MRSTFTLoss_Mel(nn.Module):
    # def __init__(self, fft_sizes=[16, 32, 64, 128], hop_sizes=[4, 8, 16, 32], win_lengths=[16, 32, 64, 128], 
    #              mu=-9.6236, sigma=3.3116): # Usa i tuoi valori esatti
    # def __init__(self, fft_sizes=[16, 32, 64, 128], hop_sizes=[4, 8, 16, 32], win_lengths=[16, 32, 64, 128], 
    #              mu=-9.7897, sigma=3.2147): # Usa i tuoi valori esatti    
    def __init__(self, fft_sizes=[8, 16, 32, 64], hop_sizes=[2, 4, 8, 16], win_lengths=[8, 16, 32, 64], 
                 mu=-9.1232, sigma=3.5412): # Usa i tuoi valori esatti   
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.mu = mu
        self.sigma = sigma
        
        for i, w in enumerate(win_lengths):
            self.register_buffer(f"window_{i}", torch.hann_window(w))

    def forward(self, x, y):
        # DE-NORMALIZZAZIONE (Z-score -> Log-Mel)
        x_logmel = (x * self.sigma) + self.mu
        y_logmel = (y * self.sigma) + self.mu
        
        # 2 CONVERSIONE IN AMPIEZZA 
        x_linear = torch.exp(x_logmel)
        y_linear = torch.exp(y_logmel)
        
        B, C, F_mel, T = x_linear.shape
        x_flat = x_linear.view(-1, T)
        y_flat = y_linear.view(-1, T)
        
        loss_sc, loss_mag = 0.0, 0.0

        for i, (fft_size, hop_size, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")
            
            x_stft = torch.stft(x_flat, n_fft=fft_size, hop_length=hop_size, 
                                win_length=win_length, window=window, 
                                center=True, return_complex=True)
            y_stft = torch.stft(y_flat, n_fft=fft_size, hop_length=hop_size, 
                                win_length=win_length, window=window, 
                                center=True, return_complex=True)

            x_mag = torch.abs(x_stft) + 1e-7
            y_mag = torch.abs(y_stft) + 1e-7

            # Spectral Convergence
            sc = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
            
            # Magnitude Loss
            mag = F.l1_loss(torch.log(x_mag), torch.log(y_mag))

            loss_sc += sc
            loss_mag += mag

        return loss_sc / len(self.fft_sizes), loss_mag / len(self.fft_sizes)
    

       
