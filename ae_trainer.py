import torch
import torch.nn as nn

class AETrainer:
    """Handles autoencoder training with balanced mini-batches from LiveDataset."""

    def __init__(self, autoencoder, lr=0.0001, device='cuda'):
        self.autoenc = autoencoder
        self.device = device
        self.optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, human_sigs):
        """Single gradient step on a batch of human signals. Returns loss value."""
        
        human_sigs = human_sigs.to(self.device)
        reconstructed = self.autoenc(human_sigs)
        
        loss = self.loss_fn(human_sigs, reconstructed)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
