import torch
import torch.nn as nn
import sys

class ResidualCatBlock(nn.Module):
    """
    Residual block che concatena i flussi lungo la dimensione del canale
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.cat([out, residual], dim=1)
        return self.relu(out)

class ResidualSumBlock(nn.Module):
    """
    Residual block che somma i flussi
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, num_res_blocks=3, residual_mode='sum'): # residual_mode = 'sum' or 'concat'
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        if residual_mode=='concat':
            res_blocks = []
            channels = base_channels
            for _ in range(num_res_blocks):
                res_blocks.append(ResidualCatBlock(channels))
                channels *= 2
            self.res_blocks = nn.Sequential(*res_blocks)
            self.output_conv = nn.Conv2d(channels, in_channels, 3, 1, 1)

        elif residual_mode=='sum':
            self.res_blocks = nn.Sequential(*[ResidualSumBlock(base_channels) for _ in range(num_res_blocks)])
            self.output_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        out = self.input_conv(x)
        out = self.res_blocks(out)
        out = self.output_conv(out)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, dropout_p=0.3):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2, 2),  # dimezza dimensioni

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),           
            nn.Dropout2d(dropout_p),
            #nn.MaxPool2d(2, 2),  # dimezza dimensioni

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            #nn.MaxPool2d(2, 2),  # dimezza dimensioni
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p)        
        )
        
        # Global Average Pooling: riduce [B, C, F, T] -> [B, C, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.readout_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
            # niente Sigmoid se usi LSGAN o WGAN
        )

    def forward(self, x, return_features=False):
        features = self.conv_layers(x)           # [B, 128, F', T']
        pooled = self.global_pool(features)      # [B, 128, 1, 1]
        x_flat = torch.flatten(pooled, 1)        # [B, 128]
        out = self.readout_layers(x_flat)        # [B, 1]

        if return_features:
            return out, features
        else:
            return out
import torch
import torch.nn as nn
import sys

class ResidualCatBlock(nn.Module):
    """
    Residual block che concatena i flussi lungo la dimensione del canale
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.cat([out, residual], dim=1)
        return self.relu(out)

class ResidualSumBlock(nn.Module):
    """
    Residual block che somma i flussi
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class Generator(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, num_res_blocks=3, residual_mode='sum'): # residual_mode = 'sum' or 'concat'
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        if residual_mode=='concat':
            res_blocks = []
            channels = base_channels
            for _ in range(num_res_blocks):
                res_blocks.append(ResidualCatBlock(channels))
                channels *= 2
            self.res_blocks = nn.Sequential(*res_blocks)
            self.output_conv = nn.Conv2d(channels, in_channels, 3, 1, 1)

        elif residual_mode=='sum':
            self.res_blocks = nn.Sequential(*[ResidualSumBlock(base_channels) for _ in range(num_res_blocks)])
            self.output_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        out = self.input_conv(x)
        out = self.res_blocks(out)
        out = self.output_conv(out)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, dropout_p=0.3):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2, 2),  # dimezza dimensioni

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),           
            nn.Dropout2d(dropout_p),
            #nn.MaxPool2d(2, 2),  # dimezza dimensioni

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            #nn.MaxPool2d(2, 2),  # dimezza dimensioni
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p)        
        )
        
        # Global Average Pooling: riduce [B, C, F, T] -> [B, C, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.readout_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
            # niente Sigmoid se usi LSGAN o WGAN
        )

    def forward(self, x, return_features=False):
        features = self.conv_layers(x)           # [B, 128, F', T']
        pooled = self.global_pool(features)      # [B, 128, 1, 1]
        x_flat = torch.flatten(pooled, 1)        # [B, 128]
        out = self.readout_layers(x_flat)        # [B, 1]

        if return_features:
            return out, features
        else:
            return out


############################################################
        # self.flatten = nn.Flatten()

        # self.readout_layers = nn.Sequential(
        #     nn.Linear(2048, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )

    # def forward(self, x, return_features=False):
    #     features = self.conv_layers(x)
    #     x_flat = self.flatten(features)

    #     # print(f'XFLAT shape: {x_flat.shape}')
    #     # sys.exit()
    #     out = self.readout_layers(x_flat)

    #     if return_features:
    #         return out, features
    #     else:
    #         return out
##############################################################

class DysarthricGAN:
    def __init__(self, in_channels, device, residual_mode, dropout_p):
        # inizializza generator e discriminator
        self.netG = Generator(in_channels=in_channels, residual_mode=residual_mode).to(device) # specifica qui residual_mode = 'sum' or 'concat'
        self.netD = Discriminator(in_channels=2*in_channels, dropout_p=dropout_p).to(device)

    def get_models(self):
        return self.netG, self.netD


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    my_DysGAN = DysarthricGAN(in_channels=1, device=device, residual_mode='sum')
    my_G, my_D = my_DysGAN.get_models()

    dummy_sample = torch.randn(((32, 1, 80, 178))).to(device)

    # Test del Generatore
    out_G = my_G(dummy_sample)
    print(f'out_G shape: {out_G.shape}')

    # Test del Discriminatore
    out_D = my_D(torch.cat((dummy_sample, dummy_sample), dim=1))
    print(f'out_D shape: {out_D.shape}')

