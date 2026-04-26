import torch
import random
from collections import deque

class LiveDataset:
    """
    Per-class queue storage for live training.

    Stores (latent, human_sig, label) tuples in separate fixed-size queues, one per
    OKM class. Balances AE training batches across classes and prevents any single
    gesture from dominating the buffer.

    Args:
        num_classes (int): number of queues, one per OKM centroid.
        queue_size_per_class (int): max entries per queue; oldest are evicted when full.
    """

    def __init__(self, num_classes, queue_size_per_class):
        self.num_classes = num_classes
        self.queue_size = queue_size_per_class
        self.queues = [deque(maxlen=queue_size_per_class) for _ in range(num_classes)]

    def apply_datapoint(self, AE_latent, human_sig, class_index, label):
        """
        Adds a (latent, human_sig, label) tuple to the queue for class_index.

        Args:
            AE_latent (torch.Tensor): encoded latent vector.
            human_sig (torch.Tensor): raw input signal.
            class_index (int): OKM class this sample belongs to.
            label (int or None): ground truth label index, or None if unlabeled.
        """
        self.queues[class_index].append((
            AE_latent.detach().cpu(),
            human_sig.detach().cpu(),
            label,
        ))

    def get_random_latents(self, num_latents):
        """
        Samples latents from all queues (unbalanced).

        Args:
            num_latents (int): max samples to return.

        Returns:
            torch.Tensor or None: stacked latents, or None if all queues are empty.
        """
        all_latents = [latent for q in self.queues for latent, _, _ in q]
        if not all_latents:
            return None
        if num_latents >= len(all_latents):
            return torch.stack(all_latents)
        return torch.stack(random.sample(all_latents, num_latents))

    def get_random_human_sigs(self, num_sigs):
        """
        Returns a class-balanced batch of human signals for AE training.

        Args:
            num_sigs (int): target batch size (approximately honored).

        Returns:
            torch.Tensor or None: stacked signals, or None if all queues are empty.
        """
        non_empty = [q for q in self.queues if len(q) > 0]
        if not non_empty:
            return None

        per_class = max(1, num_sigs // len(non_empty))
        samples = []
        for q in non_empty:
            entries = list(q)
            chosen = random.sample(entries, min(per_class, len(entries)))
            samples.extend(sig for _, sig, _ in chosen)

        return torch.stack(samples)

    def get_queue_sizes(self):
        """
        Returns:
            list[int]: current entry count for each class queue.
        """
        return [len(q) for q in self.queues]

    def total_size(self):
        """
        Returns:
            int: total entries across all queues.
        """
        return sum(len(q) for q in self.queues)
