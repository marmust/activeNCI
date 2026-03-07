import torch


class OnlineKmeans:
    """
    Online k-means clustering operating on externally-stored latent vectors.
    No local data storage — reads latents from a LiveDataset via retrain().
    """

    def __init__(self, num_centroids, device, retrain_iters=10, convergence_atol=1e-6):
        self.num_centroids = num_centroids
        self.device = device
        self.retrain_iters = retrain_iters
        self.convergence_atol = convergence_atol

        self.centroids = None
        self.initialized = False

    def retrain(self, latents):
        """
        Run Lloyd's algorithm on the provided latents tensor.
        Handles dead centroids by reinitializing them to the farthest point.
        """
        if latents is None or len(latents) < self.num_centroids:
            return

        data = latents.to(self.device)

        # first-time centroid seeding
        if not self.initialized:
            indices = torch.randperm(data.size(0))[:self.num_centroids]
            self.centroids = data[indices].clone()
            self.initialized = True

        for _ in range(self.retrain_iters):
            dists = torch.cdist(data, self.centroids)
            labels = dists.argmin(dim=1)

            new_centroids = self.centroids.clone()
            for i in range(self.num_centroids):
                mask = labels == i
                if mask.any():
                    new_centroids[i] = data[mask].mean(dim=0)
                else:
                    # dead centroid: reinitialize to the point farthest from any centroid
                    min_dists = dists.min(dim=1).values
                    farthest_idx = min_dists.argmax()
                    new_centroids[i] = data[farthest_idx].clone()

            if torch.allclose(new_centroids, self.centroids, atol=self.convergence_atol):
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

    def classify(self, latent):
        """Classify a single latent vector. Returns (class_index, dists) or (None, None)."""
        if not self.initialized:
            return None, None

        point = latent.to(self.device).unsqueeze(0)
        dists = torch.cdist(point, self.centroids).squeeze(0)
        class_index = dists.argmin().item()

        return class_index, dists
