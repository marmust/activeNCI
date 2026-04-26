import torch
import torch.nn as nn

class AETrainer:
    """
    Handles autoencoder training with balanced mini-batches from LiveDataset.

    Args:
        autoencoder (nn.Module): the autoencoder to train.
        lr (float): AdamW learning rate.
        weight_decay (float): AdamW weight decay.
        device (str or torch.device): device for training.
    """

    def __init__(self, autoencoder, lr=0.0001, weight_decay=0.01, device='cuda'):
        self.autoenc = autoencoder
        self.device = device
        self.optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

    def train_step(self, human_sigs):
        """
        Runs a single gradient step on a batch of human signals.

        Args:
            human_sigs (torch.Tensor): batch of signal tensors.

        Returns:
            float: reconstruction loss value.
        """
        human_sigs = human_sigs.to(self.device)
        reconstructed = self.autoenc(human_sigs)

        loss = self.loss_fn(human_sigs, reconstructed)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
