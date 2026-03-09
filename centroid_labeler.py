import torch


class CentroidLabeler:
    """
    Derives centroid→label mappings by counting ground truth labels
    stored in a LiveDataset. No internal state — recomputes on demand.

    LiveDataset queues are indexed by centroid/class index, and each
    entry carries an optional ground truth label. This class aggregates
    those counts and exposes label + confidence metadata.
    """

    def __init__(self, num_centroids, num_labels):
        self.num_centroids = num_centroids
        self.num_labels = num_labels

    def _build_counts(self, dataset):
        """Count label occurrences per centroid from the dataset queues."""
        counts = torch.zeros(self.num_centroids, self.num_labels)
        for centroid_idx, queue in enumerate(dataset.queues):
            for _latent, _human_sig, label in queue:
                if label is not None:
                    counts[centroid_idx, label] += 1.0
        return counts

    def get_label_map(self, dataset):
        """
        Returns a dict of {centroid_idx: label_idx} for all centroids.
        Value is None if no labeled data exists for that centroid.
        """
        counts = self._build_counts(dataset)
        return {
            i: (counts[i].argmax().item() if counts[i].sum() >= 1e-6 else None)
            for i in range(self.num_centroids)
        }

    def get_confidence(self, dataset, centroid_idx):
        """
        Returns confidence (0-1) for a centroid's label assignment.
        High confidence = one label dominates. Low = ambiguous.
        """
        counts = self._build_counts(dataset)
        row = counts[centroid_idx]
        total = row.sum()
        if total < 1e-6:
            return 0.0
        return (row.max() / total).item()

    def get_confidence_map(self, dataset):
        """Returns a dict of {centroid_idx: confidence} for all centroids."""
        counts = self._build_counts(dataset)
        result = {}
        for i in range(self.num_centroids):
            row = counts[i]
            total = row.sum()
            result[i] = (row.max() / total).item() if total >= 1e-6 else 0.0
        return result
