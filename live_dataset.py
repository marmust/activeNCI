import torch
import random
from collections import deque

class LiveDataset:
    """
    Per-class queue storage for live training.
    Stores (latent, human_sig, label) tuples in separate queues per OKM class,
    preventing class starvation and enabling balanced AE training.
    Label is optional (None when no ground truth is available).
    """

    def __init__(self, num_classes, queue_size_per_class):
        self.num_classes = num_classes
        self.queue_size = queue_size_per_class
        # each queue stores (latent, human_sig, label) tuples
        self.queues = [deque(maxlen=queue_size_per_class) for _ in range(num_classes)]

    def apply_datapoint(self, AE_latent, human_sig, class_index, label):
        """Add a (latent, human_sig, label) tuple to the specified class queue."""
        self.queues[class_index].append((
            AE_latent.detach().cpu(),
            human_sig.detach().cpu(),
            label,
        ))

    def get_random_latents(self, num_latents):
        """Sample latents from all queues (unbalanced). Returns stacked tensor or None."""
        all_latents = []
        for q in self.queues:
            for latent, _, _ in q:
                all_latents.append(latent)
        if not all_latents:
            return None
        if num_latents >= len(all_latents):
            return torch.stack(all_latents)
        return torch.stack(random.sample(all_latents, num_latents))

    def get_random_human_sigs(self, num_sigs):
        """Balanced sampling of human sigs across classes for AE training."""
        non_empty = [q for q in self.queues if len(q) > 0]
        if not non_empty:
            return None

        per_class = max(1, num_sigs // len(non_empty))
        samples = []
        for q in non_empty:
            entries = list(q)
            k = min(per_class, len(entries))
            chosen = random.sample(entries, k)
            samples.extend(sig for _, sig, _ in chosen)

        return torch.stack(samples)

    def get_queue_sizes(self):
        """Returns a list of current sizes for each class queue."""
        return [len(q) for q in self.queues]

    def total_size(self):
        """Total number of datapoints across all queues."""
        return sum(len(q) for q in self.queues)
