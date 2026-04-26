import torch

class CentroidLabeler:
    """
    Derives and caches centroid-to-label mappings from ground truth in a LiveDataset.

    Call update(dataset) whenever new labeled data arrives to recompute the mapping.
    Use predict(class_idx) to read the current cached assignment.

    Args:
        num_centroids (int): number of OKM centroids.
        num_labels (int): number of possible label classes.
    """

    def __init__(self, num_centroids, num_labels):
        self.num_centroids = num_centroids
        self.num_labels = num_labels
        self._label_map = {i: None for i in range(num_centroids)}

    def update(self, dataset):
        """
        Recomputes and caches the label map from the current dataset state.

        Args:
            dataset (LiveDataset): dataset to scan.
        """
        counts = self._build_counts(dataset)
        self._label_map = {
            i: (counts[i].argmax().item() if counts[i].sum() >= 1e-6 else None)
            for i in range(self.num_centroids)
        }

    def predict(self, class_idx):
        """
        Returns the cached label index for a centroid.

        Args:
            class_idx (int): centroid index to query.

        Returns:
            int or None: label index, or None if the centroid is unlabeled.
        """
        return self._label_map.get(class_idx)

    @property
    def label_map(self):
        """dict: current cached {centroid_idx: label_idx or None} mapping."""
        return self._label_map

    def _build_counts(self, dataset):
        """
        Counts label occurrences per centroid from dataset queues.

        Args:
            dataset (LiveDataset): dataset to scan.

        Returns:
            torch.Tensor: (num_centroids, num_labels) count matrix.
        """
        counts = torch.zeros(self.num_centroids, self.num_labels)
        for centroid_idx, queue in enumerate(dataset.queues):
            for _latent, _human_sig, label in queue:
                if label is not None:
                    counts[centroid_idx, label] += 1.0
        return counts

    def get_confidence(self, dataset, centroid_idx):
        """
        Returns label confidence for a single centroid (0 = ambiguous, 1 = unanimous).

        Args:
            dataset (LiveDataset): dataset to scan.
            centroid_idx (int): centroid to query.

        Returns:
            float: fraction of votes for the winning label.
        """
        counts = self._build_counts(dataset)
        row = counts[centroid_idx]
        total = row.sum()
        if total < 1e-6:
            return 0.0
        return (row.max() / total).item()

    def get_confidence_map(self, dataset):
        """
        Returns label confidence for every centroid.

        Args:
            dataset (LiveDataset): dataset to scan.

        Returns:
            dict: {centroid_idx: float} confidence values.
        """
        counts = self._build_counts(dataset)
        result = {}
        for i in range(self.num_centroids):
            row = counts[i]
            total = row.sum()
            result[i] = (row.max() / total).item() if total >= 1e-6 else 0.0
        return result
