import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class PairedMelSpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.pairs = []

        # Scansione delle sottocartelle
        for subfolder in self.root_dir.iterdir():
            if subfolder.is_dir():
                sano_path = subfolder / "sano.pth"
                sano_in_path = subfolder / "sano_in.pth"
                disartrico_path = subfolder / "disartrico.pth"
                if sano_path.exists() and disartrico_path.exists() and sano_in_path.exists():
                    self.pairs.append((sano_path, disartrico_path, sano_in_path ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sano_path, disartrico_path , sano_in_path = self.pairs[idx]
        
        sano_tensor = torch.load(sano_path, weights_only=True)#.unsqueeze(0)
        sano_in_tensor = torch.load(sano_in_path, weights_only=True)#.unsqueeze(0)
        disartrico_tensor = torch.load(disartrico_path, weights_only=True)#.unsqueeze(0)

        return sano_tensor, disartrico_tensor, sano_in_tensor


if __name__ == "__main__":
    # Dataset test
    dataset = PairedMelSpectrogramDataset(root_dir="/home/deepfake/DysarthricGAN/M14/M14_MEL_SPEC")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for idx, (s, d) in enumerate(dataloader):
        print(f'{idx} - Sano shape: {s.shape} - Disartrico shape: {d.shape}')
        if idx==10:
            break
